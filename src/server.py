from app import create_app, socketio
from modules.config.settings import load_config
from server_manager import Server
from utils.logging_utils import configure_logging
import threading
import time
from datetime import datetime, timedelta

def find_use_true_feature(feature_dict):
    for key, value in feature_dict.items():
        if isinstance(value, dict) and value.get('use') is True:
            return key
    return None

# Configure logging
logger = configure_logging(socketio)

# Load configuration
config = load_config('/home/unav/Desktop/UNav_socket/hloc.yaml')

feature_global = config.get('feature', {}).get('global', {})
feature = find_use_true_feature(feature_global)

# Create Server instance
server = Server(config, logger, feature)

# Complete app setup with server
app = create_app(server)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001)
