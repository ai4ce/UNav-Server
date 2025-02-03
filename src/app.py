# app.py
from flask import Flask
from flask_socketio import SocketIO
from modules.db import init_db
from modules.config.settings import Config
from modules.socketio_handlers import setup_socketio_handlers
from modules.routes import register_routes

app = Flask(__name__, template_folder='templates')
app.config.from_object(Config)
app.config.update(SESSION_COOKIE_SAMESITE="None", SESSION_COOKIE_SECURE=True)

socketio = SocketIO(app)  # Initialize SocketIO with app at module level
client_frames = {}

def create_app(server=None):
    # Initialize the database
    init_db(app)
    
    if server:
        setup_socketio_handlers(socketio, server, client_frames)
    
    register_routes(app, server, socketio)
    
    return app
