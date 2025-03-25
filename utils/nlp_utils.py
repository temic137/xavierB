import cohere
import json
import logging
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import json
import re
from pymongo import MongoClient      
import torch
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import logging
import faiss
import numpy as np
import torch
from huggingface_hub import InferenceClient
import datetime
import random

nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

import os
from dotenv import load_dotenv

load_dotenv()

huggingface_token = os.getenv('HUGGINGFACE_API_TOKEN')

cohere_token =  os.getenv('COHERE_CLIENT')

# Initialize Cohere client (you'll replace this with your actual API key)
cohere_client = cohere.Client(api_key=cohere_token)


# Replace the Cohere client initialization
huggingface_client = InferenceClient(api_key=huggingface_token)

# Get Groq API key from environment
groq_token = os.getenv('GROQ_API_KEY')

def preprocess_text(text):
    # The existing preprocessing remains the same
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.isalpha()]
    return ' '.join(tokens)

def preprocess_data(pdf_data, folder_data, web_data):
    # Maintain the existing preprocess_data function structure
    structured_data = []

    logging.debug(f"Preprocessing PDF data: {len(pdf_data)} items")
    logging.debug(f"Preprocessing folder data: {len(folder_data)} items")
    logging.debug(f"Preprocessing web data: {bool(web_data)}")

    # Process PDF and folder data
    for item in pdf_data + folder_data:
        if isinstance(item, dict) and 'text' in item:
            sentences = nltk.sent_tokenize(item['text'])
            structured_data.extend([{'type': 'text', 'content': sent, 'source': 'pdf/folder'} for sent in sentences])

    # Process web data (existing logic remains the same)
    if web_data:
        logging.debug(f"Web data keys: {web_data.keys()}")
        if isinstance(web_data, list) and len(web_data) > 0:
            web_data = web_data[0]
        
        if isinstance(web_data, dict):
            if 'title' in web_data:
                structured_data.append({'type': 'title', 'content': web_data['title'], 'source': 'web'})
            
            if 'sections' in web_data:
                for section in web_data['sections']:
                    if isinstance(section, dict):
                        if 'heading' in section:
                            structured_data.append({'type': 'heading', 'content': section['heading'], 'source': 'web'})
                        if 'content' in section:
                            structured_data.extend([{'type': 'web_content', 'content': item, 'source': 'web'} for item in section['content']])
            
            if 'sub_pages' in web_data:
                for sub_page in web_data['sub_pages']:
                    structured_data.extend(preprocess_data([], [], [sub_page]))

    logging.info(f"Preprocessed data: {len(structured_data)} items")
    return structured_data


# def generate_answer(question, relevant_info, max_length=150):
#     # Combine all relevant information into a single context
#     context = " ".join(relevant_info)
    
#     try:
#         # Use Cohere's chat generation with web search
#         response = cohere_client.chat(
#             model='command-r-plus',
#             message=f"Context: {context}\n\nQuestion: {question}",
#             temperature=0.3,
#             connectors=[{"id": "web-search"}]
#         )
        
#         return response.text
#     except Exception as e:
#         logging.error(f"Error generating answer: {str(e)}")
#         return "I apologize, but I encountered an issue while processing your question."






from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict
import logging
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except:
    pass

def prepare_bm25_index(documents: List[str]) -> BM25Okapi:
    """
    Prepare BM25 index from a list of documents.
    """
    tokenized_documents = [word_tokenize(doc.lower()) for doc in documents]
    return BM25Okapi(tokenized_documents)

def get_relevant_passages(question: str, documents: List[str], bm25: BM25Okapi, top_k: int = 3) -> List[str]:
    """
    Retrieve the most relevant passages using BM25 scoring.
    """
    tokenized_question = word_tokenize(question.lower())
    scores = bm25.get_scores(tokenized_question)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [documents[i] for i in top_indices]





# def generate_answer(question: str, documents: List[str], max_length: int = 500, context_threshold: int = 4000) -> str:
#     """
#     Generate an answer based on the question and context, with dynamic response length depending on complexity.
    
#     Args:
#         question: The question to answer
#         documents: List of document passages
#         max_length: Maximum length of the generated answer
#         context_threshold: Character threshold above which to use passage retrieval
#     """
#     try:
#         # Calculate total context length
#         total_context_length = sum(len(doc) for doc in documents)
        
#         # Determine whether to use passage retrieval
#         if total_context_length > context_threshold:
#             # Use BM25 for large contexts
#             bm25 = prepare_bm25_index(documents)
#             relevant_info = get_relevant_passages(question, documents, bm25)
#             context = " ".join(relevant_info)
#         else:
#             # Use full context for smaller documents
#             context = " ".join(documents)
        
#         print(f"context: {context}")
        
#         # Keywords that might indicate need for detailed response
#         detail_indicators = [
#             # "explain", "describe", "elaborate", "details", "how does", "how do",
#         ]
        
#         # Check if question requires detailed response
#         needs_detail = any(indicator in question.lower() for indicator in detail_indicators)
        
#         system_content = """You are a helpful assistant that provides clear, natural responses. Follow these guidelines:
# 1. Adapt your response length to the question:
#    - For simple questions, be brief and direct
#    - For complex questions, provide comprehensive explanations
# 2. When detailed information is requested:
#    - Include relevant examples
#    - Break down complex concepts
#    - Provide step-by-step explanations when appropriate
# 3. For basic questions:
#    - Keep responses concise
#    - Focus on key points only
# 4. Always:
#    - Use natural language
#    - Stay relevant to the question
#    - Organize information logically
#    - Avoid unnecessary repetition"""
        
#         messages = [
#             {
#                 "role": "system",
#                 "content": system_content
#             },
#             {
#                 "role": "user",
#                 "content": f"""Context: {context}

# Question: {question}

# Response type: {"detailed" if needs_detail else "concise"}"""
#             }
#         ]
        
#         response = huggingface_client.chat.completions.create(
#             model="mistralai/Mistral-7B-Instruct-v0.2",
#             messages=messages,
#             temperature=0.5,
#             max_tokens=500
#         )
        
#         return response.choices[0].message.content
    
#     except Exception as e:
#         logging.error(f"Error generating answer: {str(e)}")
#         return "Sorry, I ran into an issue processing your question."
    


# def generate_answer(question: str, documents: List[str], max_length: int = 500, context_threshold: int = 4000) -> str:
#     """
#     Generate an answer based on the question and context, using passage retrieval only for large contexts.

#     Args:
#         question: The question to answer
#         documents: List of document passages
#         max_length: Maximum length of the generated answer
#         context_threshold: Character threshold above which to use passage retrieval
#     """
#     try:
        
#         total_context_length = sum(len(doc) for doc in documents)

#         # Determine whether to use passage retrieval
#         if total_context_length > context_threshold:
#             # Use BM25 for large contexts
#             bm25 = prepare_bm25_index(documents)
#             relevant_info = get_relevant_passages(question, documents, bm25)
#             context = " ".join(relevant_info)
#         else:
#             # Use full context for smaller documents
#             context = " ".join(documents)

#         messages = [
#             {
#                 "role": "system",
#                 "content": """You are a customer service representative providing helpful, direct, and concise responses. Follow these guidelines:

# Response Structure:
# - Do not include the word 'Response:' or similar phrases in your answers
# - Keep answers short and clear
# - Focus only on what is necessary to answer the question
# - Use natural, conversational language
# - Directly address the question first
# - Skip unnecessary pleasantries and marketing language
# - Avoid lengthy introductions, provide answers immediately

# Additional Guidance:
# - If the information is not found in the context, say 'I'm sorry, I don't have that information.'
# - If the response requires follow-up or clarification, suggest the next step clearly."""

#             },
#             {
#                 "role": "user",
#                 "content": f"""Context: {context}
#     Question: {question}

#     Provide a precise and helpful response based on the context above. Only use the information provided. If unsure, say 'I'm sorry, I don't have that information.' Do not include the word 'Response:' in your answer."""
#             }
#         ]

#         response = huggingface_client.chat.completions.create(
#             # model="HuggingFaceH4/zephyr-7b-beta",
#             model="mistralai/Mistral-7B-Instruct-v0.2",
#             messages=messages,
#             temperature=0.5,
#             max_tokens=500
#         )
#         print(response.choices[0].message.content)

#         return response.choices[0].message.content

#     except Exception as e:
#         logging.error(f"Error generating answer: {str(e)}")
#         return "I apologize, but I encountered an issue while processing your question."


def generate_answer(question: str, documents: List[str], max_length: int = 500, context_threshold: int = 4000) -> str:
    """
    Generate an answer based on the question and context using Groq's LLM.

    Args:
        question: The question to answer
        documents: List of document passages
        max_length: Maximum length of the generated answer
        context_threshold: Character threshold above which to use passage retrieval
    """
    try:
        from groq import Groq
        
        # Initialize Groq client with environment variable
        client = Groq(api_key=groq_token)
        
        if not groq_token:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        
        total_context_length = sum(len(doc) for doc in documents)

        # Determine whether to use passage retrieval
        if total_context_length > context_threshold:
            # Use BM25 for large contexts
            bm25 = prepare_bm25_index(documents)
            relevant_info = get_relevant_passages(question, documents, bm25)
            context = " ".join(relevant_info)
        else:
            # Use full context for smaller documents
            context = " ".join(documents)

        chat_completion = client.chat.completions.create(
#             messages=[
#                 {
#                     "role": "system",
#                     "content": """You are a customer service representative providing helpful, direct, and concise responses. Follow these guidelines:

# Response Structure:
# - Do not include the word 'Response:' or similar phrases in your answers
# - Keep answers short and clear
# - Focus only on what is necessary to answer the question
# - Use natural, conversational language
# - Directly address the question first
# - Skip unnecessary pleasantries and marketing language
# - Avoid lengthy introductions, provide answers immediately

# Additional Guidance:
# - If the information is not found in the context, say 'I'm sorry, I don't have that information.'
# - If the response requires follow-up or clarification, suggest the next step clearly."""
#                 },
#                 {
#                     "role": "user",
#                     "content": f"""Context: {context}
# Question: {question}

# Provide a precise and helpful response based on the context above. Only use the information provided. If unsure, say 'I'm sorry, I don't have that information.' Do not include the word 'Response:' in your answer."""
#                 }
#             ],

                messages=[
                {
                    "role": "system",
                    "content": """You are a helpful AI assistant that provides natural, contextually appropriate responses. Scale your responses to match the user's input.

Core Guidelines:
1. Match the User's Style
- Keep responses brief for brief queries
- Be more detailed only when questions require it
- Mirror the user's formality level
- Use natural, conversational language

2. Information Handling
- Only use information from the provided context
- Say "I don't have enough information to answer that question" when needed
- Be direct and straightforward
- Avoid unnecessary elaboration

3. Response Quality
- Answer the main question first
- Add details only if relevant
- Skip unnecessary pleasantries
- Stay focused and on-topic"""
                },
                {
                    "role": "user",
                    "content": f"""Context: {context}

Question: {question}

Provide a response that:
1. Matches the question's scope and complexity
2. Uses only contextual information
3. Is natural and appropriately concise"""
                }
            ],


            model="llama3-8b-8192",
            temperature=0.0,
            max_tokens=max_length
        )
        
        print(f'chatting: ,{chat_completion.choices[0].message.content}')
        return chat_completion.choices[0].message.content

    except Exception as e:
        logging.error(f"Error generating answer: {str(e)}")
        return "I apologize, but I encountered an issue while processing your question."



def get_general_answer(data, question):
    try:
        # Process input data (existing logic remains the same)
        chatbot_data = json.loads(data) if isinstance(data, str) else data
        if isinstance(chatbot_data, list) and chatbot_data:
            chatbot_data = chatbot_data[-1]
        
        if not isinstance(chatbot_data, dict):
            raise ValueError("Invalid data format.")
        
        # Extract data from different sources
        pdf_data = chatbot_data.get('pdf_data', [])
        folder_data = chatbot_data.get('folder_data', [])
        web_data = chatbot_data.get('web_data', {})
        
        # Preprocess and structure the data
        structured_data = preprocess_data(pdf_data, folder_data, web_data)
        text_data = [item['content'] for item in structured_data if 'content' in item]
        
        # # Generate and enhance the answer
        # answer = generate_answer(question, text_data)
        # Generate answer
        answer = generate_answer(question, text_data)
        
        return answer
        
    except Exception as e:
        logging.error(f"Error processing question: {str(e)}")
        return "I apologize, but I ran into an issue while processing your question. Could you try rephrasing it?"

# Keep other existing functions like split_complex_query and post_process_answer unchanged

def preprocess_text(text):
    # Convert text to lowercase and tokenize
    text = text.lower()
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token.isalpha()]

    # Return the cleaned text
    return ' '.join(tokens)


def split_complex_query(query):
    sentences = sent_tokenize(query)
    questions = [s for s in sentences if s.strip().endswith('?')]
    return questions if questions else [query]


def post_process_answer(answer, question):
    if not any(keyword in answer.lower() for keyword in question.lower().split()):
        answer = f"To answer your question about {question.lower()}: {answer}"
    
    if any(keyword in answer.lower() for keyword in ['might', 'maybe', 'possibly', 'not sure']):
        answer += " Please note that this answer is based on the available information and may not be completely certain."
    
    return answer


def get_inventory_rag_answer(data, query):
    try:
        chatbot_data = json.loads(data) if isinstance(data, str) else data
        inventory_data = chatbot_data.get('db_data', [{}])[0].get('text', [])
        inventory_data = json.loads(inventory_data) if isinstance(inventory_data, str) else inventory_data

        questions = split_complex_query(query)
        final_answer = ""

        for question in questions:
            inventory_summary = "\n".join([f"{item['name']}: ${item['price']:.2f}, {item['quantity']} in stock" for item in inventory_data])
            answer = generate_answer(question, [inventory_summary])
            answer = post_process_answer(answer, question)
            final_answer += answer + " "

        return final_answer.strip()
    except Exception as e:
        logging.error(f"Error generating inventory answer: {str(e)}")
        return "I apologize, there seems to be an issue with processing your query."




# import json
# import logging
# import os
# import re
# from typing import List, Dict, Any, Optional, Tuple

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Import necessary libraries
# import numpy as np
# import nltk
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Lazy loading for NLTK resources
# _nltk_resources = {'punkt': False, 'stopwords': False, 'wordnet': False}
# _lemmatizer = None
# _stop_words = None

# # Singleton patterns for API clients
# _groq_client = None
# _huggingface_client = None
# _cohere_client = None

# # Get API keys from environment
# huggingface_token = os.getenv('HUGGINGFACE_API_TOKEN')
# cohere_token = os.getenv('COHERE_CLIENT')
# groq_token = os.getenv('GROQ_API_KEY')

# def get_nltk_resource(resource_name: str) -> bool:
#     """Lazily download NLTK resources only when needed."""
#     global _nltk_resources
#     if not _nltk_resources.get(resource_name, False):
#         try:
#             nltk.download(resource_name, quiet=True)
#             _nltk_resources[resource_name] = True
#             logging.debug(f"Downloaded NLTK resource: {resource_name}")
#             return True
#         except Exception as e:
#             logging.error(f"Failed to download NLTK resource {resource_name}: {str(e)}")
#             return False
#     return True

# def get_lemmatizer() -> WordNetLemmatizer:
#     """Get or initialize the WordNetLemmatizer."""
#     global _lemmatizer
#     if _lemmatizer is None:
#         get_nltk_resource('wordnet')
#         _lemmatizer = WordNetLemmatizer()
#     return _lemmatizer

# def get_stopwords() -> set:
#     """Get or initialize the stopwords set."""
#     global _stop_words
#     if _stop_words is None:
#         get_nltk_resource('stopwords')
#         _stop_words = set(stopwords.words('english'))
#     return _stop_words

# def get_groq_client():
#     """Get or initialize the Groq client."""
#     global _groq_client
#     if _groq_client is None:
#         try:
#             from groq import Groq
#             if not groq_token:
#                 raise ValueError("GROQ_API_KEY environment variable is not set")
#             _groq_client = Groq(api_key=groq_token)
#         except Exception as e:
#             logging.error(f"Error initializing Groq client: {str(e)}")
#             raise
#     return _groq_client

# def get_huggingface_client():
#     """Get or initialize the HuggingFace client."""
#     global _huggingface_client
#     if _huggingface_client is None:
#         try:
#             from huggingface_hub import InferenceClient
#             if not huggingface_token:
#                 raise ValueError("HUGGINGFACE_API_TOKEN environment variable is not set")
#             _huggingface_client = InferenceClient(api_key=huggingface_token)
#         except Exception as e:
#             logging.error(f"Error initializing HuggingFace client: {str(e)}")
#             raise
#     return _huggingface_client

# def get_cohere_client():
#     """Get or initialize the Cohere client."""
#     global _cohere_client
#     if _cohere_client is None:
#         try:
#             import cohere
#             if not cohere_token:
#                 raise ValueError("COHERE_CLIENT environment variable is not set")
#             _cohere_client = cohere.Client(api_key=cohere_token)
#         except Exception as e:
#             logging.error(f"Error initializing Cohere client: {str(e)}")
#             raise
#     return _cohere_client

# def preprocess_text(text: str) -> str:
#     """
#     Preprocess text by converting to lowercase, removing stopwords and lemmatizing.
#     Lazy loads required NLTK resources.
    
#     Args:
#         text: Input text string
        
#     Returns:
#         Preprocessed text string
#     """
#     if not text or not isinstance(text, str):
#         return ""
    
#     # Convert to lowercase
#     text = text.lower()
    
#     # Get resources only when needed
#     get_nltk_resource('punkt')
#     lemmatizer = get_lemmatizer()
#     stop_words = get_stopwords()
    
#     # Tokenize and filter
#     tokens = word_tokenize(text)
#     tokens = [lemmatizer.lemmatize(token) for token in tokens 
#               if token not in stop_words and token.isalpha()]
    
#     # Return the cleaned text
#     return ' '.join(tokens)

# def basic_preprocess_text(text: str) -> str:
#     """
#     Lightweight text preprocessing without lemmatization or stopword removal.
    
#     Args:
#         text: Input text string
        
#     Returns:
#         Lowercase text with only alphabetic tokens
#     """
#     if not text or not isinstance(text, str):
#         return ""
    
#     # Convert to lowercase
#     text = text.lower()
    
#     # Get only the punkt resource (lighter than the full preprocessing)
#     get_nltk_resource('punkt')
    
#     # Tokenize and filter
#     tokens = word_tokenize(text)
#     tokens = [token for token in tokens if token.isalpha()]
    
#     # Return the cleaned text
#     return ' '.join(tokens)

# def split_complex_query(query: str) -> List[str]:
#     """
#     Split a complex query into individual questions.
    
#     Args:
#         query: Input query text
        
#     Returns:
#         List of questions
#     """
#     if not query:
#         return []
    
#     get_nltk_resource('punkt')
#     sentences = sent_tokenize(query)
#     questions = [s for s in sentences if s.strip().endswith('?')]
#     return questions if questions else [query]

# def post_process_answer(answer: str, question: str) -> str:
#     """
#     Post-process an answer to ensure relevance to the question.
    
#     Args:
#         answer: Raw answer text
#         question: Original question
        
#     Returns:
#         Post-processed answer
#     """
#     if not answer or not question:
#         return answer
    
#     # Add question context if not present
#     question_keywords = [w for w in question.lower().split() if len(w) > 3]
#     if question_keywords and not any(keyword in answer.lower() for keyword in question_keywords):
#         answer = f"To answer your question about {question.lower()}: {answer}"
    
#     # Add uncertainty disclaimer if needed
#     uncertainty_markers = ['might', 'maybe', 'possibly', 'not sure', 'could be', 'perhaps']
#     if any(marker in answer.lower() for marker in uncertainty_markers):
#         answer += " Please note that this answer is based on the available information and may not be completely certain."
    
#     return answer

# class BM25Retriever:
#     """Memory-efficient BM25 retrieval implementation."""
    
#     def __init__(self):
#         self.bm25 = None
#         self.documents = []
    
#     def prepare_index(self, documents: List[str], chunk_size: int = 100) -> None:
#         """
#         Prepare BM25 index from documents using chunking for memory efficiency.
        
#         Args:
#             documents: List of document texts
#             chunk_size: Size of chunks to process
#         """
#         from rank_bm25 import BM25Okapi
        
#         self.documents = documents
#         get_nltk_resource('punkt')
        
#         # Process in chunks
#         tokenized_documents = []
#         for i in range(0, len(documents), chunk_size):
#             batch = documents[i:i+chunk_size]
#             # Only process valid strings
#             tokenized_batch = [word_tokenize(doc.lower()) for doc in batch if isinstance(doc, str)]
#             tokenized_documents.extend(tokenized_batch)
#             # Free memory
#             del batch
        
#         self.bm25 = BM25Okapi(tokenized_documents)
    
#     def get_relevant_passages(self, question: str, top_k: int = 3) -> List[str]:
#         """
#         Retrieve relevant passages for a question.
        
#         Args:
#             question: Query text
#             top_k: Number of top results to return
            
#         Returns:
#             List of relevant document passages
#         """
#         if not self.bm25 or not self.documents:
#             return []
        
#         get_nltk_resource('punkt')
#         tokenized_question = word_tokenize(question.lower())
#         scores = self.bm25.get_scores(tokenized_question)
        
#         # Get top k indices, ensuring we don't exceed array bounds
#         top_k = min(top_k, len(self.documents))
#         top_indices = np.argsort(scores)[-top_k:][::-1]
        
#         # Only return valid indices
#         return [self.documents[i] for i in top_indices if i < len(self.documents)]
    
#     def cleanup(self) -> None:
#         """Release resources to free memory."""
#         del self.bm25
#         del self.documents
#         self.bm25 = None
#         self.documents = []
#         # Force garbage collection
#         import gc
#         gc.collect()

# def preprocess_data_chunk(pdf_chunk, folder_chunk, web_data):
#     """Process a chunk of data to avoid memory issues."""
#     structured_data = []
    
#     # Process PDF and folder data
#     for item in pdf_chunk + folder_chunk:
#         if isinstance(item, dict) and 'text' in item:
#             get_nltk_resource('punkt')
#             sentences = sent_tokenize(item['text'])
#             structured_data.extend([{'type': 'text', 'content': sent, 'source': 'pdf/folder'} for sent in sentences])
    
#     return structured_data

# def preprocess_web_data(web_data):
#     """Process web data separately."""
#     structured_data = []
    
#     if isinstance(web_data, list) and len(web_data) > 0:
#         web_data = web_data[0]
    
#     if isinstance(web_data, dict):
#         if 'title' in web_data:
#             structured_data.append({'type': 'title', 'content': web_data['title'], 'source': 'web'})
        
#         if 'sections' in web_data:
#             for section in web_data['sections']:
#                 if isinstance(section, dict):
#                     if 'heading' in section:
#                         structured_data.append({'type': 'heading', 'content': section['heading'], 'source': 'web'})
#                     if 'content' in section:
#                         structured_data.extend([{'type': 'web_content', 'content': item, 'source': 'web'} 
#                                               for item in section['content']])
        
#         if 'sub_pages' in web_data:
#             # Process sub-pages in chunks to avoid deep recursion
#             for sub_page in web_data['sub_pages'][:10]:  # Limit to avoid recursion issues
#                 sub_data = preprocess_web_data([sub_page])
#                 structured_data.extend(sub_data)
    
#     return structured_data

# def preprocess_data(pdf_data, folder_data, web_data):
#     """
#     Memory-efficient data preprocessing function.
#     Processes data in chunks to avoid memory issues.
    
#     Args:
#         pdf_data: List of PDF data objects
#         folder_data: List of folder data objects
#         web_data: Web data object or list
        
#     Returns:
#         List of structured data items
#     """
#     structured_data = []
    
#     logging.debug(f"Preprocessing PDF data: {len(pdf_data)} items")
#     logging.debug(f"Preprocessing folder data: {len(folder_data)} items")
#     logging.debug(f"Preprocessing web data: {bool(web_data)}")
    
#     # Process PDF data in chunks
#     chunk_size = 10  # Smaller chunks to avoid memory spikes
#     for i in range(0, len(pdf_data), chunk_size):
#         pdf_chunk = pdf_data[i:i+chunk_size]
#         chunk_result = preprocess_data_chunk(pdf_chunk, [], {})
#         structured_data.extend(chunk_result)
#         # Free memory
#         del pdf_chunk, chunk_result
    
#     # Process folder data in chunks
#     for i in range(0, len(folder_data), chunk_size):
#         folder_chunk = folder_data[i:i+chunk_size]
#         chunk_result = preprocess_data_chunk([], folder_chunk, {})
#         structured_data.extend(chunk_result)
#         # Free memory
#         del folder_chunk, chunk_result
    
#     # Process web data
#     if web_data:
#         web_result = preprocess_web_data(web_data)
#         structured_data.extend(web_result)
#         # Free memory
#         del web_result
    
#     logging.info(f"Preprocessed data: {len(structured_data)} items")
#     return structured_data

# def get_relevant_passages(question: str, documents: List[str], top_k: int = 3) -> List[str]:
#     """
#     Memory-efficient function to get relevant passages using BM25.
    
#     Args:
#         question: Query text
#         documents: List of documents
#         top_k: Number of passages to return
        
#     Returns:
#         List of relevant passages
#     """
#     retriever = BM25Retriever()
#     retriever.prepare_index(documents)
#     passages = retriever.get_relevant_passages(question, top_k)
#     retriever.cleanup()
#     return passages


# def generate_answer(question: str, documents: List[str], max_length: int = 500, context_threshold: int = 4000) -> str:
#     """
#     Generate an answer based on the question and context using Groq's LLM.
#     Memory-optimized version that processes documents in chunks.

#     Args:
#         question: The question to answer
#         documents: List of document passages
#         max_length: Maximum length of the generated answer
#         context_threshold: Character threshold above which to use passage retrieval
#     """
#     try:
#         # Calculate total context length without loading all at once
#         total_context_length = sum(len(doc) for doc in documents if isinstance(doc, str))
        
#         # Create context based on size
#         if total_context_length > context_threshold:
#             # Use BM25 retriever for large contexts
#             retriever = BM25Retriever()
#             retriever.prepare_index(documents)
#             relevant_docs = retriever.get_relevant_passages(question, top_k=3)
#             context = " ".join(relevant_docs)
#             # Clean up retriever resources
#             retriever.cleanup()
#         else:
#             # For smaller contexts, just join but limit size
#             max_docs = min(len(documents), 50)  # Avoid memory issues with huge arrays
#             context = " ".join(documents[:max_docs])
        
#         # Get the Groq client
#         client = get_groq_client()
        
#         # Choose model based on environment (smaller for free tier)
#         model_name = "llama3-8b-8192"  # Default
#         if os.getenv('RENDER_SERVICE_TYPE') == 'web' and os.getenv('RENDER_SERVICE_PLAN', '').lower() == 'free':
#             model_name = "llama3-8b-8192"  # Use smaller model on free tier
        
#         # Generate response
#         chat_completion = client.chat.completions.create(
#             messages=[
#                 {
#                     "role": "system",
#                     "content": """You are a helpful AI assistant that provides natural, contextually appropriate responses. Scale your responses to match the user's input.

# Core Guidelines:
# 1. Match the User's Style
# - Keep responses brief for brief queries
# - Be more detailed only when questions require it
# - Mirror the user's formality level
# - Use natural, conversational language

# 2. Information Handling
# - Only use information from the provided context
# - Say "I don't have enough information to answer that question" when needed
# - Be direct and straightforward
# - Avoid unnecessary elaboration

# 3. Response Quality
# - Answer the main question first
# - Add details only if relevant
# - Skip unnecessary pleasantries
# - Stay focused and on-topic"""
#                 },
#                 {
#                     "role": "user",
#                     "content": f"""Context: {context}

# Question: {question}

# Provide a response that:
# 1. Matches the question's scope and complexity
# 2. Uses only contextual information
# 3. Is natural and appropriately concise"""
#                 }
#             ],
#             model=model_name,
#             temperature=0.0,
#             max_tokens=max_length
#         )
        
#         # Get response text
#         answer = chat_completion.choices[0].message.content
        
#         # Free up memory
#         del context, chat_completion
        
#         return answer

#     except Exception as e:
#         logging.error(f"Error generating answer: {str(e)}")
#         return "I apologize, but I encountered an issue while processing your question."

# def get_general_answer(data, question):
#     """
#     Process data and generate an answer to a question.
#     Memory-optimized version.
    
#     Args:
#         data: Input data (JSON or dict)
#         question: User question
        
#     Returns:
#         Generated answer
#     """
#     try:
#         # Process input data
#         chatbot_data = json.loads(data) if isinstance(data, str) else data
        
#         # Handle list input
#         if isinstance(chatbot_data, list) and chatbot_data:
#             chatbot_data = chatbot_data[-1]
        
#         if not isinstance(chatbot_data, dict):
#             raise ValueError("Invalid data format.")
        
#         # Extract data from different sources
#         pdf_data = chatbot_data.get('pdf_data', [])
#         folder_data = chatbot_data.get('folder_data', [])
#         web_data = chatbot_data.get('web_data', {})
        
#         # Process data in chunks
#         structured_data = []
        
#         # Process PDF data in chunks
#         chunk_size = 10  # Smaller chunks to avoid memory spikes
#         for i in range(0, len(pdf_data), chunk_size):
#             pdf_chunk = pdf_data[i:i+chunk_size]
#             chunk_result = preprocess_data_chunk(pdf_chunk, [], {})
#             structured_data.extend(chunk_result)
#             # Free memory
#             del pdf_chunk, chunk_result
        
#         # Process folder data in chunks
#         for i in range(0, len(folder_data), chunk_size):
#             folder_chunk = folder_data[i:i+chunk_size]
#             chunk_result = preprocess_data_chunk([], folder_chunk, {})
#             structured_data.extend(chunk_result)
#             # Free memory
#             del folder_chunk, chunk_result
        
#         # Process web data
#         if web_data:
#             web_result = preprocess_web_data(web_data)
#             structured_data.extend(web_result)
#             # Free memory
#             del web_result
        
#         # Get text data in chunks
#         text_data = []
#         for i in range(0, len(structured_data), 100):
#             chunk = structured_data[i:i+100]
#             text_chunk = [item.get('content', '') for item in chunk if isinstance(item, dict) and 'content' in item]
#             text_data.extend(text_chunk)
#             # Free memory
#             del chunk, text_chunk
        
#         # Generate answer
#         answer = generate_answer(question, text_data)
        
#         # Clean up
#         del text_data, structured_data
        
#         return answer
        
#     except Exception as e:
#         logging.error(f"Error processing question: {str(e)}")
#         return "I apologize, but I ran into an issue while processing your question. Could you try rephrasing it?"
