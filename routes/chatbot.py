from flask import Blueprint, request, jsonify, session, current_app,logging, url_for, redirect,  render_template, url_for,Response,Response, stream_with_context
from models import Chatbot,Feedback, Ticket, TicketResponse,QuestionAnalytics
from werkzeug.utils import secure_filename
from extensions import db
from utils.nlp_utils import preprocess_text, get_general_answer
from utils.file_utils import extract_text_from_pdf, read_text_file,extract_folder_content,extract_text_from_url
from utils.api_utils import fetch_real_time_data
import json
import uuid
import os
from functools import wraps
from transformers import pipeline
import logging
import speech_recognition as sr
import tempfile
from sqlalchemy import desc
from flask_cors import cross_origin, CORS
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from email.mime.text import MIMEText
import base64
from models import Chatbot, GmailIntegration, Escalation, EscalationMessage
from extensions import db
from routes.analytics import track_question_helper
from datetime import datetime
from flask_socketio import SocketIO, emit, join_room, leave_room
import threading
from collections import deque
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool


from pusher import Pusher
from dotenv import load_dotenv
import os

load_dotenv()

pusher_client = Pusher(
    app_id=os.getenv('PUSHER_APP_ID'),
    key=os.getenv('PUSHER_KEY'),
    secret=os.getenv('PUSHER_SECRET'),
    cluster=os.getenv('PUSHER_CLUSTER'),
    ssl=True
)


logging.basicConfig(level=logging.ERROR)

chatbot_bp = Blueprint('chatbot', __name__)

CORS(chatbot_bp, supports_credentials=True)

recognizer = sr.Recognizer()



def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function

def handle_errors(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            current_app.logger.error(f"Error in {f.__name__}: {str(e)}")
            return jsonify({"error": "An unexpected error occurred"}), 500
    return decorated_function



@chatbot_bp.route('/create_chatbot', methods=['POST'])
@login_required
@handle_errors
def create_chatbot():
    data = request.json
    name = data.get('name')

    new_chatbot = Chatbot(
        id=str(uuid.uuid4()),
        name=name,
        user_id=session['user_id'],
        data=json.dumps([])
    )
    db.session.add(new_chatbot)
    db.session.commit()

    return jsonify({"message": "Chatbot created successfully", "chatbot_id": new_chatbot.id}), 201


@chatbot_bp.route('/chatbots', methods=['GET'])
@login_required
@handle_errors
def get_chatbots():
    try:
        user_id = session.get('user_id')  # Use .get() instead of direct access
        if not user_id:
            return jsonify({"error": "Unauthorized"}), 401
            
        chatbots = Chatbot.query.filter_by(user_id=user_id).all()
        chatbot_list = [{"id": c.id, "name": c.name} for c in chatbots]
        
        return jsonify(chatbot_list), 200
    except Exception as e:
        current_app.logger.error(f"Error in get_chatbots: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500




def transcribe_audio():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand that.")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None



@chatbot_bp.route('/train_chatbot/<chatbot_id>', methods=['POST'])
@handle_errors
def train_chatbot(chatbot_id):
    try:
        chatbot = Chatbot.query.get(chatbot_id)
        if not chatbot or chatbot.user_id != session['user_id']:
            return jsonify({"error": "Chatbot not found or unauthorized"}), 404
        
        file = request.files.get('file')
        api_url = request.form.get('api_url')
        folder_path = request.form.get('folder_path')
        website_url = request.form.get('website_url')
        
        pdf_data = []
        db_data = []
        folder_data = []
        web_data = []
        
        if file:
            filename = secure_filename(file.filename)
            upload_folder = current_app.config['UPLOAD_FOLDER']
            
            os.makedirs(upload_folder, exist_ok=True)
            
            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)
            
            file_extension = os.path.splitext(filename)[1].lower()
            
            try:
                if file_extension == '.pdf':
                    pdf_text = extract_text_from_pdf(filepath)
                    pdf_data.append(pdf_text)
                    print(f"data:{pdf_data}")
                elif file_extension in ['.txt', '.md', '.rst']:
                    raw_text = read_text_file(filepath)
                    pdf_data.append({'page': 'file', 'text': raw_text})
                    
                else:
                    return jsonify({"error": f"Unsupported file type: {file_extension}"}), 400
            except Exception as e:
                current_app.logger.error(f"Error processing file {filename}: {str(e)}")
                return jsonify({"error": f"Error processing file {filename}: {str(e)}"}), 500
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)
        
        if api_url:
            try:
                real_time_data = fetch_real_time_data(api_url)
                if real_time_data:
                    real_time_text = json.dumps(real_time_data, indent=2)
                    db_data.append({'page': 'real_time', 'text': real_time_text})
            except Exception as e:
                current_app.logger.error(f"Error fetching data from API {api_url}: {str(e)}")
                return jsonify({"error": f"Error fetching data from API: {str(e)}"}), 500
        
        if folder_path:
            try:
                folder_data = extract_folder_content(folder_path)
            except Exception as e:
                current_app.logger.error(f"Error extracting content from folder {folder_path}: {str(e)}")
                return jsonify({"error": f"Error extracting content from folder: {str(e)}"}), 500
        
        if website_url:
            try:
                extracted_data = extract_text_from_url(website_url)
                if isinstance(extracted_data, list) and extracted_data and extracted_data[0].get('tag') == 'error':
                    return jsonify({"error": extracted_data[0]['text']}), 400
                web_data = extracted_data
            except Exception as e:
                current_app.logger.error(f"Error extracting text from URL {website_url}: {str(e)}")
                return jsonify({"error": f"Error extracting text from URL: {str(e)}"}), 500
        
        if not pdf_data and not db_data and not folder_data and not web_data:
            return jsonify({"error": "No data provided. Please upload a file, provide a folder path, provide an API URL, provide a website URL, or ensure MongoDB has data."}), 400
       
        new_data = {
            "pdf_data": pdf_data,
            "db_data": db_data,
            "folder_data": folder_data,
            "web_data": web_data
        }
        
        # If chatbot.data is empty, initialize it as a list
        if not chatbot.data:
            chatbot.data = []
        
        # If chatbot.data is a string, parse it first
        if isinstance(chatbot.data, str):
            chatbot.data = json.loads(chatbot.data)
        
        # Append the new data to the existing data
        if isinstance(chatbot.data, list):
            chatbot.data.append(new_data)
        else:
            chatbot.data = [chatbot.data, new_data]
        
        # Convert the entire data structure back to a JSON string
        chatbot.data = json.dumps(chatbot.data)
        
        db.session.commit()
        
        return jsonify({"message": "Chatbot trained successfully"}), 200
    
    except Exception as e:
        current_app.logger.error(f"Unexpected error in train_chatbot: {str(e)}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500



@chatbot_bp.route('/chatbot/<chatbot_id>', methods=['GET'])
def get_chatbot_data(chatbot_id):
    chatbot = Chatbot.query.get(chatbot_id)
    if not chatbot or chatbot.user_id != session['user_id']:
        return jsonify({"error": "Chatbot not found or unauthorized"}), 404
    
   

    return jsonify({"id": chatbot.id, "name": chatbot.name, "data": chatbot.data}), 200


@chatbot_bp.route('/chatbot/<chatbot_id>', methods=['PUT'])
def update_chatbot_data(chatbot_id):
    chatbot = Chatbot.query.get(chatbot_id)
    if not chatbot or chatbot.user_id != session['user_id']:
        return jsonify({"error": "Chatbot not found or unauthorized"}), 404

    data = request.json
    chatbot.name = data.get('name', chatbot.name)
    chatbot.data = data.get('data', chatbot.data)

    db.session.commit()

    return jsonify({"message": "Chatbot updated successfully"}), 200


# @chatbot_bp.route('/delete_chatbot/<chatbot_id>', methods=['DELETE'])
# @handle_errors
# def delete_chatbot(chatbot_id):
#     chatbot = Chatbot.query.get(chatbot_id)
#     if not chatbot or chatbot.user_id != session['user_id']:
#         return jsonify({"error": "Chatbot not found or unauthorized"}), 404
    
#     db.session.delete(chatbot)
#     db.session.commit()
    
#     return jsonify({"message": "Chatbot deleted successfully"}), 200



@chatbot_bp.route('/delete_chatbot/<chatbot_id>', methods=['DELETE'])
@handle_errors
def delete_chatbot(chatbot_id):
    # First check if the chatbot exists and belongs to the current user
    chatbot = Chatbot.query.get(chatbot_id)
    if not chatbot or chatbot.user_id != session['user_id']:
        return jsonify({"error": "Chatbot not found or unauthorized"}), 404
    
    # Delete all related question_analytics records first
    QuestionAnalytics.query.filter_by(chatbot_id=chatbot_id).delete()
    
    # Then delete the chatbot
    db.session.delete(chatbot)
    db.session.commit()
    
    return jsonify({"message": "Chatbot deleted successfully"}), 200




@chatbot_bp.route('/chatbots/<chatbot_id>', methods=['GET'])
@handle_errors
def get_chatbot(chatbot_id):
    user_id = session['user_id']
    chatbots = Chatbot.query.filter_by(user_id=user_id).all()

    chatbot = Chatbot.query.get_or_404(chatbot_id)
    
    # Ensure the chatbot belongs to the current user
    if chatbot.user_id != user_id:
        return jsonify({"error": "Unauthorized access"}), 403

    chatbot_list = [{"id": c.id, "name": c.name} for c in chatbots]
    
    return jsonify({
        "chatbot": {
            "id": chatbot.id, 
            "name": chatbot.name
        },
        "chatbot_list": chatbot_list
    }), 200

import time
# @chatbot_bp.route('/chatbot/<chatbot_id>/ask', methods=['POST'])
# def chatbot_ask(chatbot_id):
    
#     start_time = time.time()     # Start timing the request
    
#     chatbot = Chatbot.query.get(chatbot_id)
#     if not chatbot:
#         return jsonify({"error": "Chatbot not found"}), 404

#     if request.content_type == 'application/json':
#         # Handle JSON input
#         data = request.json
#         question = data.get('question')
#     else:
#         return jsonify({"error": "Unsupported content type"}), 415

#     if not question:
#         return jsonify({"error": "No question provided"}), 400

#     try:
#         chatbot_data_str = chatbot.data if isinstance(chatbot.data, str) else json.dumps(chatbot.data)
#         chatbot_data = json.loads(chatbot_data_str)
        
#         current_app.logger.debug(f"Chatbot data type: {type(chatbot_data)}")
#         current_app.logger.debug(f"Chatbot data: {chatbot_data}")
        
#         if isinstance(chatbot_data, list):
#             if len(chatbot_data) > 0:
#                 chatbot_data = chatbot_data[-1]
#             else:
#                 raise ValueError("Chatbot data list is empty")
#         elif not isinstance(chatbot_data, dict):
#             raise ValueError(f"Invalid chatbot data format: {type(chatbot_data)}")
        
#         current_app.logger.debug(f"Processed chatbot data type: {type(chatbot_data)}")
#         current_app.logger.debug(f"Processed chatbot data: {chatbot_data}")

       
       
#         answer = get_general_answer(json.dumps(chatbot_data), question)
        
#         # Calculate processing time
#         processing_time = time.time() - start_time
        
        
#        # Track analytics
#         analytics_data = {
#             "question": question,
#             "answer": answer,
#             "question_metadata": {
#                 "processing_time": processing_time,
#                 "keywords_matched": [
#                     keyword for keyword in ["proce", "inventory", "stock", "available", "category", "type"]
#                     if keyword in question.lower()
#                 ]
#             }
#         }
#         track_question_helper(chatbot_id, analytics_data)
        
#         return jsonify({
#             "question": question,
#             "answer": answer,
#             "processing_time": round(time() - start_time, 3)
#         })

#     except json.JSONDecodeError as e:
#         current_app.logger.error(f"JSON decode error: {str(e)}")
#         return jsonify({"error": "Invalid chatbot data format"}), 500
#     except ValueError as e:
#         current_app.logger.error(f"Value error: {str(e)}")
#         return jsonify({"error": str(e)}), 500
#     except Exception as e:
#         current_app.logger.error(f"Error in chatbot_ask: {str(e)}")
#         return jsonify({"error": "An unexpected error occurred"}), 500



@chatbot_bp.route('/chatbot/<chatbot_id>/ask', methods=['POST'])
def chatbot_ask(chatbot_id):
    start_time = time.time()  # Start timing the request

    chatbot = Chatbot.query.get(chatbot_id)
    if not chatbot:
        return jsonify({"error": "Chatbot not found"}), 404

    if request.content_type == 'application/json':
        # Handle JSON input
        data = request.json
        question = data.get('question')
    else:
        return jsonify({"error": "Unsupported content type"}), 415

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        chatbot_data_str = chatbot.data if isinstance(chatbot.data, str) else json.dumps(chatbot.data)
        chatbot_data = json.loads(chatbot_data_str)

        current_app.logger.debug(f"Chatbot data type: {type(chatbot_data)}")
        current_app.logger.debug(f"Chatbot data: {chatbot_data}")

        if isinstance(chatbot_data, list):
            if len(chatbot_data) > 0:
                chatbot_data = chatbot_data[-1]
            else:
                raise ValueError("Chatbot data list is empty")
        elif not isinstance(chatbot_data, dict):
            raise ValueError(f"Invalid chatbot data format: {type(chatbot_data)}")

        current_app.logger.debug(f"Processed chatbot data type: {type(chatbot_data)}")
        current_app.logger.debug(f"Processed chatbot data: {chatbot_data}")

        answer = get_general_answer(json.dumps(chatbot_data), question)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Track analytics
        analytics_data = {
            "question": question,
            "answer": answer,
            "question_metadata": {
                "processing_time": processing_time,
                "keywords_matched": [
                    keyword for keyword in ["proce", "inventory", "stock", "available", "category", "type"]
                    if keyword in question.lower()
                ]
            }
        }
        track_question_helper(chatbot_id, analytics_data)

        return jsonify({
            "question": question,
            "answer": answer,
            "processing_time": round(processing_time, 3)
        })

    except json.JSONDecodeError as e:
        current_app.logger.error(f"JSON decode error: {str(e)}")
        return jsonify({"error": "Invalid chatbot data format"}), 500
    except ValueError as e:
        current_app.logger.error(f"Value error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        current_app.logger.error(f"Error in chatbot_ask: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500






def transcribe_audio_file(file_path):
    r = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    return None


@chatbot_bp.route('/chatbot/<chatbot_id>/feedback', methods=['POST', 'OPTIONS'])
def submit_feedback(chatbot_id):
    if request.method == 'OPTIONS':
        return '', 204
    
    chatbot = Chatbot.query.get(chatbot_id)
    if not chatbot:
        return jsonify({"error": "Chatbot not found"}), 404

    data = request.json
    feedback_text = data.get('feedback')
    # user_id = request.headers.get('User-ID')
    user_id = request.headers.get('User-ID', '4269')  # Retrieve the user ID

    # user_id = data.get('user_id')  # Assume user provides their user_id in the request
    if not user_id:
        # If no user_id is provided, create a temporary user_id (or a placeholder, like "guest")
        user_id = '4269'
    if not feedback_text:
        return "Feedback is missing", 400

    if not feedback_text:
        return jsonify({"error": "No feedback provided"}), 400
    if not user_id:
        return jsonify({"error": "User ID is missing"}), 400  # Ensure User-ID is present

    try:
        new_feedback = Feedback(
            chatbot_id=chatbot_id,
            user_id=4269,  # Ensure this is an integer or correct type as per your DB schema
            feedback=feedback_text,
            created_at=datetime.utcnow()
        )
        db.session.add(new_feedback)
        db.session.commit()
        return jsonify({"message": "Feedback submitted successfully"}), 200
    except SQLAlchemyError as e:
        db.session.rollback()
        current_app.logger.error(f"Database error in submit_feedback: {str(e)}")
        return jsonify({"error": "An error occurred while saving the feedback"}), 500



@chatbot_bp.route('/chatbot/<chatbot_id>/feedback', methods=['GET'])
@login_required
@handle_errors
def get_chatbot_feedback(chatbot_id):
    chatbot = Chatbot.query.get(chatbot_id)
    if not chatbot:
        return jsonify({"error": "Chatbot not found"}), 404
    
    # Query all feedback for the specified chatbot
    feedback_list = Feedback.query.filter_by(chatbot_id=chatbot_id).order_by(desc(Feedback.created_at)).all()
    
    # Prepare the response data as a string
    feedback_strings = []
    for feedback in feedback_list:
        feedback_str = (
            f"Feedback ID: {feedback.id}\n"
            f"User ID: {feedback.user_id}\n"
            f"Feedback: {feedback.feedback}\n"
            f"Created At: {feedback.created_at.isoformat()}\n"
            f"------------------------"
        )
        feedback_strings.append(feedback_str)
    
    # Join all feedback strings with newlines
    combined_feedback = "\n".join(feedback_strings)
    
    return jsonify({
        "chatbot_name": chatbot.name,
        "feedback": combined_feedback
    }), 200






@chatbot_bp.route('/chatbot/all-feedback', methods=['GET'])
def get_all_chatbots_feedback():
    # Query all chatbots
    chatbots = Chatbot.query.all()
    
    if not chatbots:
        return jsonify({"error": "No chatbots found"}), 404
    
    response_data = []
    
    for chatbot in chatbots:
        # Query all feedback for each chatbot
        feedback_list = Feedback.query.filter_by(chatbot_id=chatbot.id)\
                              .order_by(desc(Feedback.created_at)).all()
        
        # Prepare feedback strings for this chatbot
        feedback_strings = []
        for feedback in feedback_list:
            feedback_str = (
                f"Feedback ID: {feedback.id}\n"
                f"User ID: {feedback.user_id}\n"
                f"Feedback: {feedback.feedback}\n"
                f"Created At: {feedback.created_at.isoformat()}\n"
                f"------------------------"
            )
            feedback_strings.append(feedback_str)
        
        # Add chatbot data to response
        chatbot_data = {
            "chatbot_id": chatbot.id,
            "chatbot_name": chatbot.name,
            "feedback": "\n".join(feedback_strings) if feedback_strings else "No feedback available"
        }
        response_data.append(chatbot_data)
    
    return jsonify({
        "total_chatbots": len(chatbots),
        "chatbots": response_data
    }), 200




# @chatbot_bp.route('/get_chatbot_script/<chatbot_id>')
# @login_required
# @handle_errors
# def get_chatbot_script(chatbot_id):
#     chatbot = Chatbot.query.get(chatbot_id)
#     if not chatbot:
#         return jsonify({"error": "Chatbot not found"}), 404
    
#     # Generate URLs for the API endpoints
#     ask_url = url_for('chatbot.chatbot_ask', chatbot_id=chatbot_id, _external=True)
#     feedback_url = url_for('chatbot.submit_feedback', chatbot_id=chatbot_id, _external=True)
#     sentiment_url = url_for('analytics.submit_sentiment', chatbot_id=chatbot_id, _external=True)
#     widget_url = url_for('static', filename='js/widget.js', _external=True)
#     ticket_url = url_for('chatbot.create_ticket', chatbot_id=chatbot_id, _external=True)
#     theme_color = ""
#     avatar = ""
#     escalate_url=url_for('chatbot.create_escalation', chatbot_id=chatbot_id, _external=True)
    
#     integration_code = f'''<!-- you can change the color of the theme used in this chat toggle to be integrated into your website with hexadecimal color -->
#     <!-- you can also change the default image of the avatar used in the chatbot widget by changing the data-avatar variable an image   -->
#     <script 
#         src="{widget_url}"
#         data-chatbot-id="{chatbot_id}"
#         data-name="{chatbot.name}"
#         data-ask-url="{ask_url}"
#         data-feedback-url="{feedback_url}"
#         data-sentiment-url="{sentiment_url}"
#         data-ticket-url="{ticket_url}"
#         data-theme-color="{theme_color}"
#         data-avatar="{avatar}"
#         data-escalate-url="{escalate_url}"
#     ></script>'''
    
#     return jsonify({
#         'integration_code': integration_code,
#         'preview': integration_code
#     })

# @chatbot_bp.route('/get_chatbot_script/<chatbot_id>')
# @login_required
# @handle_errors
# def get_chatbot_script(chatbot_id):
#     chatbot = Chatbot.query.get(chatbot_id)
#     if not chatbot:
#         return jsonify({"error": "Chatbot not found"}), 404

#     # Get customization settings
#     customization = {"theme_color": "", "avatar_url": ""}
#     if chatbot.data:
#         try:
#             chatbot_data = json.loads(chatbot.data) if isinstance(chatbot.data, str) else chatbot.data
#             if isinstance(chatbot_data, list) and chatbot_data:
#                 chatbot_data = chatbot_data[-1]
#             if isinstance(chatbot_data, dict):
#                 stored_custom = chatbot_data.get('customization', {})
#                 customization.update(stored_custom)
#         except json.JSONDecodeError:
#             pass

#     # Generate compact script
#     script = f'''<script src="{url_for('static',filename='js/widget.js',_external=True)}"data-id="{chatbot_id}"data-name="{chatbot.name}"data-theme="{customization['theme_color']}"data-avatar="{customization['avatar_url']}"data-api="{request.url_root}"></script>'''

#     return jsonify({
#         'integration_code': script,
#         'preview': script
#     })



@chatbot_bp.route('/get_chatbot_script/<chatbot_id>')
@login_required
@handle_errors
def get_chatbot_script(chatbot_id):
    chatbot = Chatbot.query.get(chatbot_id)
    if not chatbot:
        return jsonify({"error": "Chatbot not found"}), 404

    # Get customization settings with defaults
    customization = {
        "theme_color": "",
        "avatar_url": "",
        "pusher_key": "43bd6f1835e5bb8165d8",  # Add default for pusher_key
        "pusher_cluster": "us3"  # Add default for pusher_cluster
    }
    if chatbot.data:
        try:
            chatbot_data = json.loads(chatbot.data) if isinstance(chatbot.data, str) else chatbot.data
            if isinstance(chatbot_data, list) and chatbot_data:
                chatbot_data = chatbot_data[-1]
            if isinstance(chatbot_data, dict):
                stored_custom = chatbot_data.get('customization', {})
                customization.update(stored_custom)
        except json.JSONDecodeError:
            pass

    # Generate compact script with Pusher key and cluster
    script = (
        f'''<script src="{url_for('static', filename='js/widget.js', _external=True)}" data-id="{chatbot_id}" data-name="{chatbot.name}" data-theme="{customization['theme_color']}" data-avatar="{customization['avatar_url']}" data-api="{request.url_root}" data-pusher-key="{customization['pusher_key']}" data-pusher-cluster="{customization['pusher_cluster']}"></script>'''
    )

    return jsonify({
        'integration_code': script,
        'preview': script
    })

@chatbot_bp.route('/ticket/create/<chatbot_id>', methods=['POST'])
def create_ticket(chatbot_id):
    data = request.json

    # Check required fields
    if not all(field in data for field in ['subject', 'description', 'account_details']):
        return jsonify({"error": "Missing required fields"}), 400

    # Create the ticket with default user_id
    new_ticket = Ticket(
        user_id=4269,  # Always use the default user
        chatbot_id=chatbot_id,
        subject=data['subject'],
        description=data['description'],
        priority=data.get('priority', 'medium'),
        account_details=data['account_details']
    )

    try:
        db.session.add(new_ticket)
        db.session.commit()
        return jsonify({
            "message": "Ticket created successfully",
            "ticket_id": new_ticket.id
        }), 201
    except SQLAlchemyError as e:
        db.session.rollback()
        current_app.logger.error(f"Database error in create_ticket: {str(e)}")
        return jsonify({"error": "An error occurred while creating the ticket"}), 500



@chatbot_bp.route('/tickets/<chatbot_id>', methods=['GET'])
@login_required
@handle_errors
def list_tickets1(chatbot_id):
    # Query tickets filtered by chatbot_id
    user_tickets = Ticket.query.filter_by(chatbot_id=chatbot_id).all()
    
    # Return the filtered tickets
    return jsonify({
        "tickets": [{
            "id": ticket.id,
            "subject": ticket.subject,
            "status": ticket.status,
            "priority": ticket.priority,
            "created_at": ticket.created_at.isoformat(),
            "chatbot_id": ticket.chatbot_id,
            "user_id": ticket.user_id,
            "account_details": ticket.account_details
        } for ticket in user_tickets]
    }), 200


@chatbot_bp.route('/tickets', methods=['GET'])
# @login_required
def list_tickets():
    
    user_tickets = Ticket.query.all()
    return jsonify({
        "tickets": [{
            "id": ticket.id,
            "subject": ticket.subject,
            "status": ticket.status,
            "priority": ticket.priority,
            "created_at": ticket.created_at.isoformat()
        } for ticket in user_tickets]
    }), 200


@chatbot_bp.route('/ticket/<ticket_id>', methods=['GET'])
# @login_required
# @handle_errors
def get_ticket(ticket_id):
    ticket = Ticket.query.get_or_404(ticket_id)
    responses = TicketResponse.query.filter_by(ticket_id=ticket_id).all()
    
    return jsonify({
        "ticket": {
            "id": ticket.id,
            "subject": ticket.subject,
            "description": ticket.description,
            "status": ticket.status,
            "priority": ticket.priority,
            "account_details": ticket.account_details,
            "created_at": ticket.created_at.isoformat()
        },
        "responses": [{
            "id": response.id,
            "message": response.message,
            "user_id": response.user_id,
            "created_at": response.created_at.isoformat()
        } for response in responses]
    }), 200



@chatbot_bp.route('/ticket/<ticket_id>/update-status', methods=['PATCH'])
@login_required
@handle_errors
def update_ticket_status(ticket_id):
    data = request.json
    if 'status' not in data:
        return jsonify({"error": "Status is required"}), 400
    
    ticket = Ticket.query.get_or_404(ticket_id)
    ticket.status = data['status']
    db.session.commit()
    
    return jsonify({
        "message": "Ticket status updated successfully",
        "ticket": {
            "id": ticket.id,
            "status": ticket.status
        }
    }), 200



@chatbot_bp.route('/ticket/delete/<ticket_id>', methods=['DELETE'])
@login_required
@handle_errors
def delete_ticket(ticket_id):
    ticket = Ticket.query.get(ticket_id)
    if not ticket:
        return jsonify({"error": "Ticket not found"}), 404

    db.session.delete(ticket)
    db.session.commit()
    return jsonify({"message": "Ticket deleted successfully"}), 200



# @chatbot_bp.route('/escalate', methods=['POST'])
# @cross_origin()
# def create_escalation():
#     try:
#         # Validate request
#         if not request.is_json:
#             return jsonify({"error": "Request must be JSON"}), 400

#         data = request.get_json()
#         chatbot_id = data.get('chatbot_id')
#         user_id = request.headers.get('User-ID', '4269')

#         # Validate chatbot_id
#         if not chatbot_id:
#             return jsonify({"error": "Missing chatbot_id"}), 400

#         # Check if chatbot exists
#         chatbot = Chatbot.query.get(chatbot_id)
#         if not chatbot:
#             return jsonify({"error": "Chatbot not found"}), 404

#         # Check for existing active escalation
#         existing_escalation = Escalation.query.filter_by(
#             chatbot_id=chatbot_id,
#             user_id=user_id,
#             status='pending'
#         ).first()

#         if existing_escalation:
#             return jsonify({
#                 "escalation_id": existing_escalation.id,
#                 "status": "existing",
#                 "status_url": f"/escalation/{existing_escalation.id}/status",
#                 "send_url": f"/escalation/{existing_escalation.id}/send",
#                 "messages_url": f"/escalation/{existing_escalation.id}/messages"
#             }), 200

#         # Create new escalation
#         escalation = Escalation(
#             chatbot_id=chatbot_id,
#             user_id=user_id,
#             status='pending',
#             created_at=datetime.utcnow()
#         )

#         db.session.add(escalation)
#         db.session.commit()

#         # Return success with URLs
#         return jsonify({
#             "escalation_id": escalation.id,
#             "status": "created",
#             "status_url": f"/escalation/{escalation.id}/status",
#             "send_url": f"/escalation/{escalation.id}/send",
#             "messages_url": f"/escalation/{escalation.id}/messages"
#         }), 201

#     except SQLAlchemyError as e:
#         db.session.rollback()
#         current_app.logger.error(f"Database error in escalation: {str(e)}")
#         return jsonify({"error": "Database error occurred"}), 500
#     except Exception as e:
#         current_app.logger.error(f"Escalation error: {str(e)}")
#         return jsonify({"error": "Internal server error"}), 500




# @chatbot_bp.route('/escalation/<escalation_id>/messages', methods=['GET'])
# def get_messages(escalation_id):
#     last_id = request.args.get('last_id', 0)
#     messages = EscalationMessage.query.filter(
#         EscalationMessage.escalation_id == escalation_id,
#         EscalationMessage.id > last_id
#     ).order_by(EscalationMessage.id.asc()).all()

#     return jsonify([{
#         "id": msg.id,
#         "sender": "agent" if msg.sender_id == 0 else "user",
#         "message": msg.message,
#         "timestamp": msg.timestamp.isoformat()
#     } for msg in messages]), 200


# @chatbot_bp.route('/escalation/<escalation_id>/send', methods=['POST'])
# def send_message(escalation_id):
#     data = request.json
#     message = data.get('message')
#     user_id = request.headers.get('User-ID', '4269')

#     escalation = Escalation.query.get(escalation_id)
#     if not escalation:
#         return jsonify({"error": "Escalation not found"}), 404

#     sender_id = user_id if escalation.user_id == user_id else 0  # 0 represents agent

#     new_message = EscalationMessage(
#         escalation_id=escalation_id,
#         sender_id=sender_id,
#         message=message
#     )
#     db.session.add(new_message)

#     if escalation.status == 'pending':
#         escalation.status = 'in_progress'

#     db.session.commit()

#     return jsonify({"status": "success"}), 200


# @chatbot_bp.route('/escalation/<escalation_id>/status', methods=['GET'])
# def check_escalation_status(escalation_id):
#     escalation = Escalation.query.get(escalation_id)
#     if not escalation:
#         return jsonify({"error": "Escalation not found"}), 404

#     return jsonify({
#         "status": escalation.status,
#         "agent_joined": escalation.agent_id is not None
#     }), 200

# @chatbot_bp.route('/agent/escalations', methods=['GET'])
# def get_pending_escalations():
#     escalations = Escalation.query.filter_by(status='pending').all()
#     return jsonify([{
#         "id": esc.id,
#         "chatbot_id": esc.chatbot_id,
#         "user_id": esc.user_id,
#         "created_at": esc.created_at.isoformat()
#     } for esc in escalations]), 200

# @chatbot_bp.route('/agent/escalation/<escalation_id>/join', methods=['POST'])
# def join_escalation(escalation_id):
#     escalation = Escalation.query.get(escalation_id)
#     if not escalation:
#         return jsonify({"error": "Escalation not found"}), 404
        
#     escalation.status = 'in_progress'
#     escalation.agent_id = 0  # Replace with actual agent ID from session
#     db.session.commit()
    
#     return jsonify({"message": "Escalation joined successfully"}), 200




# @chatbot_bp.route('/escalation/<escalation_id>/events', methods=['GET'])
# def escalation_events(escalation_id):
#     def event_stream():
#         escalation = Escalation.query.get(escalation_id)
#         if not escalation:
#             yield "data: Escalation not found\n\n"
#             return

#         last_id = request.args.get('last_id', 0)
#         while True:
#             # Check for new messages
#             messages = EscalationMessage.query.filter(
#                 EscalationMessage.escalation_id == escalation_id,
#                 EscalationMessage.id > last_id
#             ).order_by(EscalationMessage.id.asc()).all()

#             for message in messages:
#                 yield f"""data: {json.dumps({
#                     'id': message.id,
#                     'sender': 'agent' if message.sender_id == 0 else 'user',
#                     'message': message.message,
#                     'timestamp': message.timestamp.isoformat()
#                 })}\n\n"""
#                 last_id = message.id

#             # Check for status changes
#             escalation = Escalation.query.get(escalation_id)
#             if escalation.status == 'in_progress' and escalation.agent_id is not None:
#                 yield f"""data: {json.dumps({
#                     'type': 'status',
#                     'status': escalation.status,
#                     'agent_joined': True
#                 })}\n\n"""

#             time.sleep(1)  # Adjust the sleep time as needed

#     return Response(stream_with_context(event_stream()), content_type='text/event-stream')





from flask import Blueprint, request, jsonify, Response, stream_with_context
from models import Chatbot, Escalation, EscalationMessage
from extensions import db
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError
import json



# @chatbot_bp.route('/escalate', methods=['POST'])
# @cross_origin()
# def create_escalation():
#     try:
#         # Validate request
#         if not request.is_json:
#             return jsonify({"error": "Request must be JSON"}), 400

#         data = request.get_json()
#         chatbot_id = data.get('chatbot_id')
#         user_id = request.headers.get('User-ID', '4269')

#         # Validate chatbot_id
#         if not chatbot_id:
#             return jsonify({"error": "Missing chatbot_id"}), 400

#         # Check if chatbot exists
#         chatbot = Chatbot.query.get(chatbot_id)
#         if not chatbot:
#             return jsonify({"error": "Chatbot not found"}), 404

#         # Check for existing active escalation
#         existing_escalation = Escalation.query.filter_by(
#             chatbot_id=chatbot_id,
#             user_id=user_id,
#             status='pending'
#         ).first()

#         if existing_escalation:
#             return jsonify({
#                 "escalation_id": existing_escalation.id,
#                 "status": "existing",
#                 "status_url": f"/escalation/{existing_escalation.id}/status",
#                 "send_url": f"/escalation/{existing_escalation.id}/send",
#                 "messages_url": f"/escalation/{existing_escalation.id}/messages"
#             }), 200

#         # Create new escalation
#         escalation = Escalation(
#             chatbot_id=chatbot_id,
#             user_id=user_id,
#             status='pending',
#             created_at=datetime.utcnow()
#         )

#         db.session.add(escalation)
#         db.session.commit()

#         # Return success with URLs
#         return jsonify({
#             "escalation_id": escalation.id,
#             "status": "created",
#             "status_url": f"/escalation/{escalation.id}/status",
#             "send_url": f"/escalation/{escalation.id}/send",
#             "messages_url": f"/escalation/{escalation.id}/messages"
#         }), 201

#     except SQLAlchemyError as e:
#         db.session.rollback()
#         current_app.logger.error(f"Database error in escalation: {str(e)}")
#         return jsonify({"error": "Database error occurred"}), 500
#     except Exception as e:
#         current_app.logger.error(f"Escalation error: {str(e)}")
#         return jsonify({"error": "Internal server error"}), 500




@chatbot_bp.route('/escalate', methods=['POST'])
@cross_origin()
def create_escalation():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        chatbot_id = data.get('chatbot_id')
        user_id = request.headers.get('User-ID', '4269')

        if not chatbot_id:
            return jsonify({"error": "Missing chatbot_id"}), 400

        chatbot = Chatbot.query.get(chatbot_id)
        if not chatbot:
            return jsonify({"error": "Chatbot not found"}), 404

        existing_escalation = Escalation.query.filter_by(
            chatbot_id=chatbot_id,
            user_id=user_id,
            status='pending'
        ).first()

        if existing_escalation:
            return jsonify({
                "escalation_id": existing_escalation.id,
                "status": "existing",
                "status_url": f"/escalation/{existing_escalation.id}/status",
                "send_url": f"/escalation/{existing_escalation.id}/send",
                "messages_url": f"/escalation/{existing_escalation.id}/messages"
            }), 200

        escalation = Escalation(
            chatbot_id=chatbot_id,
            user_id=user_id,
            status='pending',
            created_at=datetime.utcnow()
        )
        db.session.add(escalation)
        db.session.commit()

        # Publish new escalation event to Pusher
        pusher_client.trigger(
            f'chatbot-{chatbot_id}-escalations',
            'escalation-update',
            {
                'type': 'escalations_update',
                'escalations': [{
                    "id": escalation.id,
                    "chatbot_id": escalation.chatbot_id,
                    "user_id": escalation.user_id,
                    "status": escalation.status,
                    "created_at": escalation.created_at.isoformat(),
                    "messages_count": 0
                }]
            }
        )

        return jsonify({
            "escalation_id": escalation.id,
            "status": "created",
            "status_url": f"/escalation/{escalation.id}/status",
            "send_url": f"/escalation/{escalation.id}/send",
            "messages_url": f"/escalation/{escalation.id}/messages"
        }), 201

    except SQLAlchemyError as e:
        db.session.rollback()
        current_app.logger.error(f"Database error in escalation: {str(e)}")
        return jsonify({"error": "Database error occurred"}), 500
    except Exception as e:
        current_app.logger.error(f"Escalation error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@chatbot_bp.route('/escalation/<escalation_id>/messages', methods=['GET'])
def get_messages(escalation_id):
    try:
        last_id = request.args.get('last_id', 0, type=int)
        
        # Use context manager for database session
        with db.session() as session:
            messages = session.query(EscalationMessage).filter(
                EscalationMessage.escalation_id == escalation_id,
                EscalationMessage.id > last_id
            ).order_by(EscalationMessage.id.asc()).all()

            return jsonify([{
                "id": msg.id,
                "sender": "agent" if msg.sender_id == 0 else "user",
                "message": msg.message,
                "timestamp": msg.timestamp.isoformat()
            } for msg in messages]), 200

    except Exception as e:
        current_app.logger.error(f"Error fetching messages: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500




# @chatbot_bp.route('/escalation/<escalation_id>/send', methods=['POST'])
# def send_message(escalation_id):
#     try:
#         data = request.json
#         message = data.get('message')
#         user_id = request.headers.get('User-ID', '4269')

#         if not message:
#             return jsonify({"error": "Message is required"}), 400

#         escalation = Escalation.query.get(escalation_id)
#         if not escalation:
#             return jsonify({"error": "Escalation not found"}), 404

#         sender_id = user_id if escalation.user_id == user_id else 0  # 0 represents agent

#         new_message = EscalationMessage(
#             escalation_id=escalation_id,
#             sender_id=sender_id,
#             message=message
#         )
#         db.session.add(new_message)

#         # if escalation.status == 'pending':
#         #     escalation.status = 'in_progress'

#         db.session.commit()

#         return jsonify({"status": "success"}), 200

#     except SQLAlchemyError as e:
#         db.session.rollback()
#         current_app.logger.error(f"Database error in send_message: {str(e)}")
#         return jsonify({"error": "Database error occurred"}), 500
#     except Exception as e:
#         current_app.logger.error(f"Error sending message: {str(e)}")
#         return jsonify({"error": "An unexpected error occurred"}), 500


# @chatbot_bp.route('/escalation/<escalation_id>/send', methods=['POST'])
# def send_message(escalation_id):
#     try:
#         data = request.json
#         message = data.get('message')
#         user_id = request.headers.get('User-ID', '4269')

#         if not message:
#             return jsonify({"error": "Message is required"}), 400

#         escalation = Escalation.query.get(escalation_id)
#         if not escalation:
#             return jsonify({"error": "Escalation not found"}), 404

#         sender_id = user_id if escalation.user_id == user_id else 0  # 0 represents agent

#         new_message = EscalationMessage(
#             escalation_id=escalation_id,
#             sender_id=sender_id,
#             message=message
#         )
#         db.session.add(new_message)

#         if escalation.status == 'pending':
#             escalation.status = 'in_progress'

#         db.session.commit()

#         # Publish new message event
#         pusher_client.trigger(
#             f'escalation-{escalation_id}',
#             'new-message',
#             {
#                 'type': 'message',
#                 'id': new_message.id,
#                 'sender': 'agent' if sender_id == 0 else 'user',
#                 'message': message,
#                 'timestamp': new_message.timestamp.isoformat()
#             }
#         )

#         # Publish status update if changed
#         if escalation.status == 'in_progress':
#             pusher_client.trigger(
#                 f'escalation-{escalation_id}',
#                 'status-update',
#                 {
#                     'type': 'status',
#                     'status': escalation.status,
#                     'agent_joined': escalation.agent_id is not None
#                 }
#             )

#         return jsonify({"status": "success"}), 200

#     except SQLAlchemyError as e:
#         db.session.rollback()
#         current_app.logger.error(f"Database error in send_message: {str(e)}")
#         return jsonify({"error": "Database error occurred"}), 500
#     except Exception as e:
#         current_app.logger.error(f"Error sending message: {str(e)}")
#         return jsonify({"error": "An unexpected error occurred"}), 500

@chatbot_bp.route('/escalation/<escalation_id>/send', methods=['POST'])
def send_message(escalation_id):
    try:
        data = request.json
        message = data.get('message')
        user_id_str = request.headers.get('User-ID', '4269')  # String from header

        if not message:
            return jsonify({"error": "Message is required"}), 400

        escalation = Escalation.query.get(escalation_id)
        if not escalation:
            return jsonify({"error": "Escalation not found"}), 404

        # Convert user_id to integer
        try:
            user_id = int(user_id_str)
        except ValueError:
            user_id = 4269  # Fallback if invalid

        # Determine sender (0 for agent, user_id for user)
        sender_id = user_id if escalation.user_id == user_id else 0

        new_message = EscalationMessage(
            escalation_id=escalation_id,
            sender_id=sender_id,
            message=message
        )
        db.session.add(new_message)

        if escalation.status == 'pending':
            escalation.status = 'in_progress'

        db.session.commit()

        # Publish new message event with explicit sender
        pusher_client.trigger(
            f'escalation-{escalation_id}',
            'new-message',
            {
                'type': 'message',
                'id': new_message.id,
                'sender': 'agent' if sender_id == 0 else 'user',
                'message': message,
                'timestamp': new_message.timestamp.isoformat()
            }
        )

        # Publish status update if changed
        if escalation.status == 'in_progress':
            pusher_client.trigger(
                f'escalation-{escalation_id}',
                'status-update',
                {
                    'type': 'status',
                    'status': escalation.status,
                    'agent_joined': escalation.agent_id is not None
                }
            )

        return jsonify({"status": "success"}), 200

    except SQLAlchemyError as e:
        db.session.rollback()
        current_app.logger.error(f"Database error in send_message: {str(e)}")
        return jsonify({"error": "Database error occurred"}), 500
    except Exception as e:
        current_app.logger.error(f"Error sending message: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500



@chatbot_bp.route('/escalation/<escalation_id>/status', methods=['GET'])
def check_escalation_status(escalation_id):
    try:
        escalation = Escalation.query.get(escalation_id)
        if not escalation:
            return jsonify({"error": "Escalation not found"}), 404

        return jsonify({
            "status": escalation.status,
            "agent_joined": escalation.agent_id is not None
        }), 200

    except Exception as e:
        current_app.logger.error(f"Error checking escalation status: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500


# @chatbot_bp.route('/agent/escalations', methods=['GET'])
# def get_pending_escalations():
#     try:
#         escalations = Escalation.query.filter_by(status='pending').all()
#         return jsonify([{
#             "id": esc.id,
#             "chatbot_id": esc.chatbot_id,
#             "user_id": esc.user_id,
#             "created_at": esc.created_at.isoformat()
#         } for esc in escalations]), 200

#     except Exception as e:
#         current_app.logger.error(f"Error fetching pending escalations: {str(e)}")
#         return jsonify({"error": "An unexpected error occurred"}), 500


# @chatbot_bp.route('/agent/escalations<chatbot_id>', methods=['GET'])
# def get_pending_escalations():
#     try:
#         # Fetch escalations with status 'pending' or 'in_progress'
#         escalations = Escalation.query.filter(
#             Escalation.status.in_(['pending', 'in_progress'])
#         ).order_by(desc(Escalation.created_at)).all()

#         # Prepare the response data
#         escalations_data = [{
#             "id": esc.id,
#             "chatbot_id": esc.chatbot_id,
#             "user_id": esc.user_id,
#             "status": esc.status,
#             "created_at": esc.created_at.isoformat()
#         } for esc in escalations]

#         # Disable caching to ensure fresh data
#         response = jsonify(escalations_data)
#         response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
#         response.headers['Pragma'] = 'no-cache'
#         response.headers['Expires'] = '0'

#         return response, 200

#     except Exception as e:
#         current_app.logger.error(f"Error fetching escalations: {str(e)}")
#         return jsonify({"error": "An unexpected error occurred"}), 500 

@chatbot_bp.route('/agent/escalations/<chatbot_id>', methods=['GET'])
def get_chatbot_escalations(chatbot_id):
    try:
        # Verify chatbot exists
        chatbot = Chatbot.query.get(chatbot_id)
        if not chatbot:
            return jsonify({"error": "Chatbot not found"}), 404

        # Fetch escalations with status 'pending' or 'in_progress' for specific chatbot
        escalations = Escalation.query.filter(
            (Escalation.chatbot_id == chatbot_id) &
            (Escalation.status.in_(['pending', 'in_progress', 'resolved', 'closed']))
        ).order_by(desc(Escalation.created_at)).all()

        # Prepare the response data
        escalations_data = [{
            "id": esc.id,
            "chatbot_id": esc.chatbot_id,
            "user_id": esc.user_id,
            "status": esc.status,
            "created_at": esc.created_at.isoformat(),
            "messages_count": EscalationMessage.query.filter_by(escalation_id=esc.id).count()
        } for esc in escalations]

        # Disable caching to ensure fresh data
        response = jsonify({
            "chatbot_name": chatbot.name,
            "total_escalations": len(escalations_data),
            "escalations": escalations_data
        })
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'

        return response, 200

    except SQLAlchemyError as e:
        db.session.rollback()
        current_app.logger.error(f"Database error in get_chatbot_escalations: {str(e)}")
        return jsonify({"error": "Database error occurred"}), 500
    except Exception as e:
        current_app.logger.error(f"Error fetching escalations: {str(e)}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500





# @chatbot_bp.route('/agent/escalation/<escalation_id>/join', methods=['POST'])
# def join_escalation(escalation_id):
#     try:
#         escalation = Escalation.query.get(escalation_id)
#         if not escalation:
#             return jsonify({"error": "Escalation not found"}), 404

#         escalation.status = 'in_progress'
#         escalation.agent_id = 0  # Replace with actual agent ID from session
#         db.session.commit()

#         return jsonify({"message": "Escalation joined successfully"}), 200

#     except SQLAlchemyError as e:
#         db.session.rollback()
#         current_app.logger.error(f"Database error in join_escalation: {str(e)}")
#         return jsonify({"error": "Database error occurred"}), 500
#     except Exception as e:
#         current_app.logger.error(f"Error joining escalation: {str(e)}")
#         return jsonify({"error": "An unexpected error occurred"}), 500


@chatbot_bp.route('/agent/escalation/<escalation_id>/join', methods=['POST'])
def join_escalation(escalation_id):
    try:
        escalation = Escalation.query.get(escalation_id)
        if not escalation:
            return jsonify({"error": "Escalation not found"}), 404

        escalation.status = 'in_progress'
        escalation.agent_id = 0  # Replace with actual agent ID from session
        db.session.commit()

        # Publish status update
        pusher_client.trigger(
            f'escalation-{escalation_id}',
            'status-update',
            {
                'type': 'status',
                'status': escalation.status,
                'agent_joined': True
            }
        )

        return jsonify({"message": "Escalation joined successfully"}), 200

    except SQLAlchemyError as e:
        db.session.rollback()
        current_app.logger.error(f"Database error in join_escalation: {str(e)}")
        return jsonify({"error": "Database error occurred"}), 500
    except Exception as e:
        current_app.logger.error(f"Error joining escalation: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500


# @chatbot_bp.route('/escalation/<escalation_id>/events', methods=['GET'])
# def escalation_events(escalation_id):
#     def event_stream():
#         try:
#             escalation = Escalation.query.get(escalation_id)
#             if not escalation:
#                 yield "data: Escalation not found\n\n"
#                 return
            
#             last_id = request.args.get('last_id', 0, type=int)
#             last_status = None  # Add this to track status changes
            
#             while True:
#                 # Check for new messages
#                 messages = EscalationMessage.query.filter(
#                     EscalationMessage.escalation_id == escalation_id,
#                     EscalationMessage.id > last_id
#                 ).order_by(EscalationMessage.id.asc()).all()
                
#                 for message in messages:
#                     yield f"""data: {json.dumps({
#                         'type': 'message',
#                         'id': message.id,
#                         'sender': 'agent' if message.sender_id == 0 else 'user',
#                         'message': message.message,
#                         'timestamp': message.timestamp.isoformat()
#                     })}\n\n"""
#                     last_id = message.id
                
#                 # Check for status changes
#                 escalation = Escalation.query.get(escalation_id)
                
#                 if (escalation.status in ['in_progress', 'pending']):
#                     if escalation.status != last_status:  # Only send if status changed
#                         print(f"Status update triggered: {escalation.status}")
#                         yield f"""data: {json.dumps({
#                             'type': 'status',
#                             'status': escalation.status,
#                             'agent_joined': True
#                         })}\n\n"""
#                         last_status = escalation.status
                
#                 time.sleep(1)
                
#         except Exception as e:
#             current_app.logger.error(f"Error in event_stream: {str(e)}")
#             yield "data: An error occurred\n\n"
    
#     return Response(stream_with_context(event_stream()), content_type='text/event-stream')





from flask import Response, stream_with_context
import json
import time

# @chatbot_bp.route('/escalation/<escalation_id>/events', methods=['GET'])
# def escalation_events(escalation_id):
#     def event_stream():
#         try:
#             escalation = Escalation.query.get(escalation_id)
#             if not escalation:
#                 yield "data: Escalation not found\n\n"
#                 return
            
#             last_id = request.args.get('last_id', 0, type=int)
#             last_status = None  # Add this to track status changes
            
#             while True:
#                 # Check for new messages
#                 messages = EscalationMessage.query.filter(
#                     EscalationMessage.escalation_id == escalation_id,
#                     EscalationMessage.id > last_id
#                 ).order_by(EscalationMessage.id.asc()).all()
                
#                 for message in messages:
#                     yield f"""data: {json.dumps({
#                         'type': 'message',
#                         'id': message.id,
#                         'sender': 'agent' if message.sender_id == 0 else 'user',
#                         'message': message.message,
#                         'timestamp': message.timestamp.isoformat()
#                     })}\n\n"""
#                     last_id = message.id
                
#                 # Check for status changes
#                 escalation = Escalation.query.get(escalation_id)
                
#                 if (escalation.status in ['in_progress', 'pending']):
#                     if escalation.status != last_status:  # Only send if status changed
#                         print(f"Status update triggered: {escalation.status}")
#                         yield f"""data: {json.dumps({
#                             'type': 'status',
#                             'status': escalation.status,
#                             'agent_joined': True
#                         })}\n\n"""
#                         last_status = escalation.status
                
#                 time.sleep(1)
                
#         except Exception as e:
#             current_app.logger.error(f"Error in event_stream: {str(e)}")
#             yield "data: An error occurred\n\n"
    
#     return Response(stream_with_context(event_stream()), content_type='text/event-stream')


# @chatbot_bp.route('/escalation/<escalation_id>/events', methods=['GET'])
# def escalation_events(escalation_id):
#     def event_stream():
#         session = None
#         try:
#             # Create a new session using the session factory
#             session = db.session()  # Changed from create_scoped_session()
            
#             escalation = session.query(Escalation).get(escalation_id)
#             if not escalation:
#                 yield "data: Escalation not found\n\n"
#                 return
            
#             last_id = request.args.get('last_id', 0, type=int)
#             last_status = None
            
#             while True:
#                 try:
#                     # Query with the session
#                     messages = session.query(EscalationMessage).filter(
#                         EscalationMessage.escalation_id == escalation_id,
#                         EscalationMessage.id > last_id
#                     ).order_by(EscalationMessage.id.asc()).all()
                    
#                     for message in messages:
#                         yield f"""data: {json.dumps({
#                             'type': 'message',
#                             'id': message.id,
#                             'sender': 'agent' if message.sender_id == 0 else 'user',
#                             'message': message.message,
#                             'timestamp': message.timestamp.isoformat()
#                         })}\n\n"""
#                         last_id = message.id
                    
#                     # Refresh escalation object
#                     session.refresh(escalation)
                    
#                     if escalation.status in ['in_progress']:
#                         if escalation.status != last_status:
#                             yield f"""data: {json.dumps({
#                                 'type': 'status',
#                                 'status': escalation.status,
#                                 'agent_joined': True
#                             })}\n\n"""
#                             last_status = escalation.status
                    
#                     # Explicitly commit
#                     session.commit()
                    
#                     time.sleep(1)
                    
#                 except Exception as e:
#                     if session:
#                         session.rollback()
#                     current_app.logger.error(f"Error in event stream loop: {str(e)}")
#                     continue
                    
#         except Exception as e:
#             current_app.logger.error(f"Error in event_stream: {str(e)}")
#             yield "data: An error occurred\n\n"
#         finally:
#             if session:
#                 session.close()
    
#     return Response(
#         stream_with_context(event_stream()),
#         content_type='text/event-stream',
#         headers={
#             'Cache-Control': 'no-cache',
#             'Connection': 'keep-alive',
#             'X-Accel-Buffering': 'no'
#         }
#     )


@chatbot_bp.route('/escalation/<escalation_id>', methods=['DELETE'])
def delete_escalation(escalation_id):
    try:
        escalation = Escalation.query.get(escalation_id)
        if not escalation:
            return jsonify({"error": "Escalation not found"}), 404

        db.session.delete(escalation)
        db.session.commit()

        return jsonify({"message": "Escalation deleted successfully"}), 200

    except SQLAlchemyError as e:
        db.session.rollback()
        current_app.logger.error(f"Database error in delete_escalation: {str(e)}")
        return jsonify({"error": "Database error occurred"}), 500
    except Exception as e:
        current_app.logger.error(f"Error in delete_escalation: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500


# @chatbot_bp.route('/escalation/<escalation_id>/status', methods=['PUT'])
# def update_escalation_status(escalation_id):
#     try:
#         # Validate request
#         if not request.is_json:
#             return jsonify({"error": "Request must be JSON"}), 400

#         data = request.get_json()
#         new_status = data.get('status')

#         # Validate status
#         if new_status not in ['resolved', 'closed']:
#             return jsonify({"error": "Invalid status. Must be 'resolved' or 'closed'"}), 400

#         # Check if escalation exists
#         escalation = Escalation.query.get(escalation_id)
#         if not escalation:
#             return jsonify({"error": "Escalation not found"}), 404

#         # Update status
#         escalation.status = new_status
#         db.session.commit()

#         return jsonify({
#             "escalation_id": escalation.id,
#             "status": escalation.status,
#             "message": f"Escalation status updated to {new_status}"
#         }), 200

#     except SQLAlchemyError as e:
#         db.session.rollback()
#         current_app.logger.error(f"Database error in update_escalation_status: {str(e)}")
#         return jsonify({"error": "Database error occurred"}), 500
#     except Exception as e:
#         current_app.logger.error(f"Error updating escalation status: {str(e)}")
#         return jsonify({"error": "An unexpected error occurred"}), 500
    
@chatbot_bp.route('/escalation/<escalation_id>/status', methods=['PUT'])
def update_escalation_status(escalation_id):
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        new_status = data.get('status')

        if new_status not in ['resolved', 'closed']:
            return jsonify({"error": "Invalid status. Must be 'resolved' or 'closed'"}), 400

        escalation = Escalation.query.get(escalation_id)
        if not escalation:
            return jsonify({"error": "Escalation not found"}), 404

        escalation.status = new_status
        db.session.commit()

        # Publish status update
        pusher_client.trigger(
            f'escalation-{escalation_id}',
            'status-update',
            {
                'type': 'status',
                'status': escalation.status,
                'agent_joined': escalation.agent_id is not None
            }
        )

        return jsonify({
            "escalation_id": escalation.id,
            "status": escalation.status,
            "message": f"Escalation status updated to {new_status}"
        }), 200

    except SQLAlchemyError as e:
        db.session.rollback()
        current_app.logger.error(f"Database error in update_escalation_status: {str(e)}")
        return jsonify({"error": "Database error occurred"}), 500
    except Exception as e:
        current_app.logger.error(f"Error updating escalation status: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500

# @chatbot_bp.route('/agent/escalations/<chatbot_id>/events', methods=['GET'])
# def chatbot_escalations_events(chatbot_id):
#     def event_stream():
#         try:
#             # Verify chatbot exists
#             chatbot = Chatbot.query.get(chatbot_id)
#             if not chatbot:
#                 yield "data: {\"error\": \"Chatbot not found\"}\n\n"
#                 return

#             last_check = datetime.utcnow()

#             while True:
#                 # Fetch new or updated escalations since last check
#                 escalations = Escalation.query.filter(
#                     (Escalation.chatbot_id == chatbot_id) &
#                     (Escalation.updated_at >= last_check)
#                 ).all()

#                 if escalations:
#                     # Prepare escalations data
#                     escalations_data = [{
#                         "id": esc.id,
#                         "chatbot_id": esc.chatbot_id,
#                         "user_id": esc.user_id,
#                         "status": esc.status,
#                         "created_at": esc.created_at.isoformat(),
#                         "messages_count": EscalationMessage.query.filter_by(escalation_id=esc.id).count()
#                     } for esc in escalations]

#                     # Create the update event data
#                     event_data = {
#                         "type": "escalations_update",
#                         "escalations": escalations_data
#                     }
                    
#                     # Properly format the SSE data
#                     message = "data: " + json.dumps(event_data) + "\n\n"
#                     yield message

#                     last_check = datetime.utcnow()

#                 time.sleep(2)  # Check every 2 seconds

#         except Exception as e:
#             current_app.logger.error(f"Error in escalations event stream: {str(e)}")
#             error_data = json.dumps({"error": str(e)})
#             yield f"data: {error_data}\n\n"

#     return Response(
#         stream_with_context(event_stream()),
#         mimetype='text/event-stream',
#         headers={
#             'Cache-Control': 'no-cache',
#             'Connection': 'keep-alive',
#             'X-Accel-Buffering': 'no'  # Added for Nginx compatibility
#         }
#     )

@chatbot_bp.route('/chatbot/<chatbot_id>/customize', methods=['GET'])
@cross_origin(supports_credentials=True)
@handle_errors
def get_chatbot_customization(chatbot_id):
    try:
        chatbot = Chatbot.query.get(chatbot_id)
        if not chatbot:
            return jsonify({"error": "Chatbot not found"}), 404

        # Initialize default customization
        customization = {
            "theme_color": "#0084ff",  # Default blue color
            "avatar_url": "",  # Default empty avatar URL
            "enable_tickets": True,  # Default to enabled
            "enable_escalation": True  # Default to enabled
        }

        # Get customization settings if they exist
        if chatbot.data:
            try:
                # Handle string data
                if isinstance(chatbot.data, str):
                    chatbot_data = json.loads(chatbot.data)
                else:
                    chatbot_data = chatbot.data

                # Handle list data (take the last item if it's a list)
                if isinstance(chatbot_data, list):
                    if chatbot_data:
                        chatbot_data = chatbot_data[-1]
                    else:
                        chatbot_data = {}

                # Get customization from dictionary
                if isinstance(chatbot_data, dict):
                    stored_customization = chatbot_data.get('customization', {})
                    if stored_customization:
                        customization.update(stored_customization)

            except json.JSONDecodeError:
                current_app.logger.error("Invalid JSON in chatbot data")
                pass  # Use default customization

        return jsonify(customization), 200

    except Exception as e:
        current_app.logger.error(f"Error in get_chatbot_customization: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500

@chatbot_bp.route('/chatbot/<chatbot_id>/customize', methods=['PUT'])
@cross_origin(supports_credentials=True)
@handle_errors
def customize_chatbot(chatbot_id):
    try:
        chatbot = Chatbot.query.get(chatbot_id)
        if not chatbot:
            return jsonify({"error": "Chatbot not found"}), 404

        data = request.json
        theme_color = data.get('theme_color')
        avatar_url = data.get('avatar_url')
        enable_tickets = data.get('enable_tickets')
        enable_escalation = data.get('enable_escalation')

        # Validate theme color (hex format)
        if theme_color:
            if not isinstance(theme_color, str) or not theme_color.startswith('#') or len(theme_color) != 7:
                return jsonify({"error": "Invalid theme color format. Use hex format (e.g., #FF0000)"}), 400

        # Validate avatar URL
        if avatar_url and not isinstance(avatar_url, str):
            return jsonify({"error": "Invalid avatar URL format"}), 400

        # Validate boolean values
        if enable_tickets is not None and not isinstance(enable_tickets, bool):
            return jsonify({"error": "enable_tickets must be a boolean value"}), 400

        if enable_escalation is not None and not isinstance(enable_escalation, bool):
            return jsonify({"error": "enable_escalation must be a boolean value"}), 400

        # Initialize or get existing data
        try:
            if chatbot.data:
                if isinstance(chatbot.data, str):
                    chatbot_data = json.loads(chatbot.data)
                else:
                    chatbot_data = chatbot.data
            else:
                chatbot_data = {}

            # If data is a list, convert to dict or initialize new dict
            if isinstance(chatbot_data, list):
                chatbot_data = chatbot_data[-1] if chatbot_data else {}
            
            # Ensure we have a dictionary
            if not isinstance(chatbot_data, dict):
                chatbot_data = {}

            # Initialize customization if it doesn't exist
            if 'customization' not in chatbot_data:
                chatbot_data['customization'] = {}

            # Update the customization values
            if theme_color:
                chatbot_data['customization']['theme_color'] = theme_color
            if avatar_url:
                chatbot_data['customization']['avatar_url'] = avatar_url
            if enable_tickets is not None:
                chatbot_data['customization']['enable_tickets'] = enable_tickets
            if enable_escalation is not None:
                chatbot_data['customization']['enable_escalation'] = enable_escalation

            # Store the updated data
            chatbot.data = json.dumps(chatbot_data)
            db.session.commit()

            return jsonify({
                "message": "Chatbot customization updated successfully",
                "customization": chatbot_data['customization']
            }), 200

        except json.JSONDecodeError:
            current_app.logger.error("Invalid JSON in chatbot data")
            # Initialize new data structure with all settings
            new_data = {
                'customization': {
                    'theme_color': theme_color if theme_color else "#0084ff",
                    'avatar_url': avatar_url if avatar_url else "",
                    'enable_tickets': enable_tickets if enable_tickets is not None else True,
                    'enable_escalation': enable_escalation if enable_escalation is not None else True
                }
            }
            chatbot.data = json.dumps(new_data)
            db.session.commit()
            
            return jsonify({
                "message": "Chatbot customization initialized successfully",
                "customization": new_data['customization']
            }), 200

    except Exception as e:
        current_app.logger.error(f"Error in customize_chatbot: {str(e)}")
        db.session.rollback()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


import os
from werkzeug.utils import secure_filename
from PIL import Image
import io

UPLOAD_FOLDER = 'static/uploads/avatars'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@chatbot_bp.route('/chatbot/<chatbot_id>/upload-avatar', methods=['POST'])
@cross_origin(supports_credentials=True)
@handle_errors
def upload_avatar(chatbot_id):
    try:
        if 'avatar' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files['avatar']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        if file and allowed_file(file.filename):
            # Create upload directory if it doesn't exist
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            
            # Process and compress image
            image = Image.open(file)
            
            # Convert to RGB if necessary
            if image.mode in ('RGBA', 'P'):
                image = image.convert('RGB')
                
            # Resize image while maintaining aspect ratio
            max_size = (400, 400)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Generate unique filename
            filename = secure_filename(f"{chatbot_id}_{int(time.time())}.jpg")
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            
            # Save compressed image
            image.save(filepath, 'JPEG', quality=85, optimize=True)
            
            # Generate URL for the saved image
            avatar_url = url_for('static', 
                               filename=f'uploads/avatars/{filename}', 
                               _external=True)
            
            return jsonify({"avatar_url": avatar_url}), 200
            
        return jsonify({"error": "Invalid file type"}), 400
        
    except Exception as e:
        current_app.logger.error(f"Error uploading avatar: {str(e)}")
        return jsonify({"error": "Failed to upload image"}), 500


def get_db_session():
    """Get a database session and ensure proper cleanup"""
    session = db.session()
    try:
        yield session
    finally:
        session.close()

@chatbot_bp.route('/ticket/<ticket_id>/priority', methods=['PUT'])
@handle_errors
def update_ticket_priority(ticket_id):
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        new_priority = data.get('priority')

        # Validate priority
        valid_priorities = ['low', 'medium', 'high']
        if not new_priority or new_priority not in valid_priorities:
            return jsonify({"error": f"Invalid priority. Must be one of: {', '.join(valid_priorities)}"}), 400

        # Check if ticket exists
        ticket = Ticket.query.get(ticket_id)
        if not ticket:
            return jsonify({"error": "Ticket not found"}), 404

        # Update priority
        ticket.priority = new_priority
        db.session.commit()

        return jsonify({
            "ticket_id": ticket.id,
            "priority": ticket.priority,
            "message": f"Ticket priority updated to {new_priority}"
        }), 200

    except SQLAlchemyError as e:
        db.session.rollback()
        current_app.logger.error(f"Database error in update_ticket_priority: {str(e)}")
        return jsonify({"error": "Database error occurred"}), 500
    except Exception as e:
        current_app.logger.error(f"Error updating ticket priority: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500
