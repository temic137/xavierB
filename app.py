from flask import Flask, send_from_directory, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from config import Config
from extensions import db
from routes.auth import auth_bp
from routes.chatbot import chatbot_bp
from routes.analytics import analytics_bp
import os
from flask_migrate import Migrate
from sqlalchemy import Text
from sqlalchemy.dialects import postgresql
from models import User

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Add session cookie configuration and load secrets from environment variables
    app.config.update(
        SESSION_COOKIE_SECURE=True,
        SESSION_COOKIE_SAMESITE='None',
        SESSION_COOKIE_HTTPONLY=True,
        GOOGLE_CLIENT_ID=os.environ.get('GOOGLE_CLIENT_ID'),  # Load from env
        GOOGLE_CLIENT_SECRET=os.environ.get('GOOGLE_CLIENT_SECRET'),  # Load from env
        GOOGLE_REDIRECT_URI='http://localhost:5000/oauth2callback'  # This can stay hardcoded if it's not sensitive
    )

    # Configure CORS with all necessary settings
    CORS(app, 
         resources={r"/*": {
             "origins": [
                 "https://xavierai-m-2.vercel.app",
                 "https://xavierai.pages.dev",
             ],
             "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
             "allow_headers": ["Content-Type", "Authorization", "X-CSRFToken", "User-ID"],
             "expose_headers": ["Content-Type", "Authorization", "X-CSRFToken"],
             "supports_credentials": True,
             "send_wildcard": False,
             "max_age": 86400
         }})
    
    # Initialize extensions
    db.init_app(app)
    migrate = Migrate(app, db)
    
    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(chatbot_bp)
    app.register_blueprint(analytics_bp)
    
    # Create database tables and default user
    with app.app_context():
        try:
            db.create_all()
            default_user = User.query.get(4269)
            if not default_user:
                default_user = User(
                    id=4269,
                    username='default_ticket_user',
                    password_hash='default_not_used'
                )
                db.session.add(default_user)
                db.session.commit()
                print("Default user created successfully")
            else:
                print("Default user already exists")
        except Exception as e:
            print(f"Database initialization error: {e}")
            db.session.rollback()
    
    return app

# Create the app instance for Gunicorn
app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)