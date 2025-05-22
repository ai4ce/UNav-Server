from flask import request, jsonify
import os
import base64
from PIL import Image
import io
import numpy as np
from utils.time_logger import TimeLogger
import time
import logging
from functools import wraps

time_logger = TimeLogger()

# Standard response format
def api_response(success=True, data=None, message=None, error=None, status_code=200):
    """
    Creates a standardized API response format.
    
    Args:
        success (bool): Whether the API call was successful
        data (dict, optional): The data to return
        message (str, optional): A success message
        error (str, optional): An error message if success is False
        status_code (int): HTTP status code
        
    Returns:
        tuple: (response_json, status_code)
    """
    response = {
        "success": success,
    }
    
    if data is not None:
        response["data"] = data
    
    if message is not None:
        response["message"] = message
        
    if error is not None:
        response["error"] = error
        
    return jsonify(response), status_code

# Exception handling decorator
def handle_exceptions(f):
    """
    Decorator to handle exceptions in route handlers.
    Ensures consistent error responses across all API endpoints.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            logging.error(f"Value error in {f.__name__}: {str(e)}")
            return api_response(success=False, error=str(e), status_code=400)
        except KeyError as e:
            logging.error(f"Key error in {f.__name__}: {str(e)}")
            return api_response(success=False, error=f"Missing required parameter: {str(e)}", status_code=400)
        except Exception as e:
            logging.error(f"Unexpected error in {f.__name__}: {str(e)}", exc_info=True)
            return api_response(success=False, error="An unexpected error occurred", status_code=500)
    return decorated_function

def register_data_routes(app, server, socketio):
    """
    Register all data-related route handlers with the Flask application.
    
    Args:
        app: Flask application instance
        server: Server instance for handling business logic
        socketio: SocketIO instance for real-time communication
    """

    @app.route('/localize', methods=['POST'])
    @handle_exceptions
    def localize():
        """
        Handle localization request by processing the provided image and returning the pose.
        
        Request:
            JSON with 'query_image' (base64 encoded image)
            
        Response:
            Success: {'success': True, 'data': {'pose': [x, y, angle]}}
            Error: {'success': False, 'error': 'Error message'}
        """
        data = request.json
        if not data:
            return api_response(success=False, error="No request data provided", status_code=400)
            
        query_image_base64 = data.get('query_image')
        if not query_image_base64:
            return api_response(success=False, error="No image provided", status_code=400)

        try:
            # Decode the base64 image
            image_data = query_image_base64.split(',')[1] if ',' in query_image_base64 else query_image_base64
            query_image_data = base64.b64decode(image_data)
            query_image = Image.open(io.BytesIO(query_image_data)).convert('RGB')
        except Exception as e:
            logging.error(f"Error decoding image: {str(e)}")
            return api_response(success=False, error="Invalid image format", status_code=400)

        # Localize the image using the server's method
        pose = server.localize(np.array(query_image))
        rounded_pose = [int(coord) for coord in pose] if pose else None

        return api_response(
            success=pose is not None,
            data={"pose": rounded_pose},
            message="Localization successful" if pose else None,
            error="Localization failed" if pose is None else None,
            status_code=200 if pose else 404
        )

    @app.route('/list_images', methods=['GET'])
    @handle_exceptions
    def get_images_list():
        """
        Return testing images list organized by ID.
        
        Response:
            Success: {'success': True, 'data': {'images': {id1: [image1.png, ...], ...}}}
            Error: {'success': False, 'error': 'Error message'}
        """
        config = server.config['location']
        target_place = config.get('place')
        target_building = config.get('building')
        target_floor = config.get('floor')
        
        if not all([target_place, target_building, target_floor]):
            return api_response(success=False, error="Incomplete location configuration", status_code=400)
        
        # Build the path based on the place, building, and floor
        data_path = os.path.join(server.root, "logs", target_place, target_building, target_floor)
        
        if not os.path.exists(data_path):
            return api_response(success=False, error=f"Path not found: {data_path}", status_code=404)
        
        images_dict = {}
        
        # Traverse through directories and collect image names
        for root, ids, _ in os.walk(data_path):
            # We are interested in directories that match ids (like '00150')
            for id in ids:
                image_dir = os.path.join(root, id, 'images')
                try:
                    if os.path.isdir(image_dir):
                        files = sorted(os.listdir(image_dir))
                        images_dict[id] = [f for f in files if f.endswith('.png')]
                except Exception as e:
                    logging.warning(f"Error reading image directory {image_dir}: {str(e)}")
        
        return api_response(success=True, data={"images": images_dict})
    
    @app.route('/get_options', methods=['GET'])
    @handle_exceptions
    def get_options():
        """
        Return a dictionary of available places, buildings, and floors for the client to choose from.
        
        Response:
            Success: {'success': True, 'data': {'options': {place1: {building1: [floor1, ...], ...}, ...}}}
            Error: {'success': False, 'error': 'Error message'}
        """
        data_path = os.path.join(server.root, "data")
        if not os.path.exists(data_path):
            return api_response(success=False, error=f"Data directory not found: {data_path}", status_code=404)
            
        try:
            places = [place for place in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, place))]
            options = {}
            
            for place in places:
                place_path = os.path.join(data_path, place)
                buildings = [building for building in os.listdir(place_path) 
                             if os.path.isdir(os.path.join(place_path, building))]
                options[place] = {}
                
                for building in buildings:
                    building_path = os.path.join(place_path, building)
                    floors = [floor for floor in os.listdir(building_path) 
                              if os.path.isdir(os.path.join(building_path, floor))]
                    options[place][building] = floors
        except Exception as e:
            logging.error(f"Error reading options: {str(e)}")
            return api_response(success=False, error="Error reading location options", status_code=500)

        return api_response(success=True, data={"options": options})

    @app.route('/list_places', methods=['GET'])
    @handle_exceptions
    def list_places():
        """
        List all available places stored on the server.
        
        Response:
            Success: {'success': True, 'data': {'places': [place1, place2, ...]}}
            Error: {'success': False, 'error': 'Error message'}
        """
        data_path = os.path.join(server.root, "data")
        if not os.path.exists(data_path):
            return api_response(success=False, error=f"Data directory not found: {data_path}", status_code=404)
            
        try:
            places = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        except Exception as e:
            logging.error(f"Error listing places: {str(e)}")
            return api_response(success=False, error="Error listing places", status_code=500)
            
        return api_response(success=True, data={"places": places})

    @app.route('/list_buildings/<place>', methods=['GET'])
    @handle_exceptions
    def list_buildings(place):
        """
        List all buildings available within a given place.
        
        Path parameters:
            place (str): The name of the place
            
        Response:
            Success: {'success': True, 'data': {'buildings': [building1, building2, ...]}}
            Error: {'success': False, 'error': 'Error message'}
        """
        place_path = os.path.join(server.root, "data", place)
        if not os.path.exists(place_path):
            return api_response(success=False, error=f"Place not found: {place}", status_code=404)
            
        try:
            buildings = [d for d in os.listdir(place_path) if os.path.isdir(os.path.join(place_path, d))]
        except Exception as e:
            logging.error(f"Error listing buildings for place {place}: {str(e)}")
            return api_response(success=False, error=f"Error listing buildings for {place}", status_code=500)
            
        return api_response(success=True, data={"buildings": buildings})

    @app.route('/list_floors/<place>/<building>', methods=['GET'])
    @handle_exceptions
    def list_floors(place, building):
        """
        List all floors available within a given building in a specified place.
        
        Path parameters:
            place (str): The name of the place
            building (str): The name of the building
            
        Response:
            Success: {'success': True, 'data': {'floors': [floor1, floor2, ...]}}
            Error: {'success': False, 'error': 'Error message'}
        """
        building_path = os.path.join(server.root, "data", place, building)
        if not os.path.exists(building_path):
            return api_response(success=False, error=f"Building '{building}' not found in place '{place}'", status_code=404)
            
        try:
            floors = [d for d in os.listdir(building_path) if os.path.isdir(os.path.join(building_path, d))]
        except Exception as e:
            logging.error(f"Error listing floors for building {building} in place {place}: {str(e)}")
            return api_response(success=False, error=f"Error listing floors for {building}", status_code=500)
            
        return api_response(success=True, data={"floors": floors})

    # @app.route('/get_scale', methods=['POST'])
    # @handle_exceptions
    # def get_scale():
    #     """
    #     Retrieve the scale for a specified place, building, and floor.
    #     
    #     Request:
    #         JSON with 'place', 'building', 'floor', and 'session_id'
    #         
    #     Response:
    #         Success: {'success': True, 'data': {'scale': 0.01234}}
    #         Error: {'success': False, 'error': 'Error message'}
    #     """
    #     data = request.json
    #     if not data:
    #         return api_response(success=False, error="No request data provided", status_code=400)
    #         
    #     place = data.get('place')
    #     building = data.get('building')
    #     floor = data.get('floor')
    #     session_id = data.get('session_id')
    #     
    #     if not all([place, building, floor, session_id]):
    #         return api_response(success=False, error="Missing required parameters", status_code=400)

    #     scale = server.get_scale(place, building, floor, session_id)
    #     return api_response(success=True, data={"scale": scale})

    @app.route('/get_destinations', methods=['POST'])
    @handle_exceptions
    def get_destinations_list():
        """
        Retrieve available destinations for the current location.
        
        Request:
            JSON with 'place', 'building', and 'floor'
            
        Response:
            Success: {'success': True, 'data': {'destinations': [...]}}
            Error: {'success': False, 'error': 'Error message'}
        """
        data = request.json
        if not data:
            return api_response(success=False, error="No request data provided", status_code=400)
            
        place = data.get('place')
        building = data.get('building')
        floor = data.get('floor')
        
        if not building or not floor:
            return api_response(success=False, error="Missing required parameters: building and floor", status_code=400)
        
        try:
            destination_data = server.get_destinations_list(building, floor)
        except Exception as e:
            logging.error(f"Error getting destinations for {building}/{floor}: {str(e)}")
            return api_response(success=False, error="Error retrieving destinations", status_code=500)
            
        return api_response(success=True, data=destination_data)

    @app.route('/get_floorplan', methods=['POST'])
    @handle_exceptions
    def get_floorplan():
        """
        Retrieve the floorplan for the current location.
        
        Request:
            JSON with 'session_id'
            
        Response:
            Success: {'success': True, 'data': {'floorplan': base64_encoded_image}}
            Error: {'success': False, 'error': 'Error message'}
        """
        data = request.json
        if not data:
            return api_response(success=False, error="No request data provided", status_code=400)
            
        session_id = data.get('session_id')
        if not session_id:
            return api_response(success=False, error="Missing required parameter: session_id", status_code=400)
        
        try:
            floorplan_data = server.get_floorplan(session_id)
        except Exception as e:
            logging.error(f"Error getting floorplan for session {session_id}: {str(e)}")
            return api_response(success=False, error="Error retrieving floorplan", status_code=500)
            
        return api_response(success=True, data=floorplan_data)
    
    @app.route('/select_destination', methods=['POST'])
    @handle_exceptions
    def select_destination():
        """
        Handle the selection of a destination by the client.
        
        Request:
            JSON with 'place', 'building', 'floor', 'destination_id', and 'session_id'
            
        Response:
            Success: {'success': True, 'message': 'Destination selected successfully'}
            Error: {'success': False, 'error': 'Error message'}
        """
        data = request.json
        if not data:
            return api_response(success=False, error="No request data provided", status_code=400)
            
        place = data.get('place')
        building = data.get('building')
        floor = data.get('floor')
        destination_id = data.get('destination_id')
        session_id = data.get('session_id')

        if not all([place, building, floor, destination_id, session_id]):
            missing = []
            if not place: missing.append('place')
            if not building: missing.append('building')
            if not floor: missing.append('floor')
            if not destination_id: missing.append('destination_id')
            if not session_id: missing.append('session_id')
            return api_response(success=False, error=f"Missing required parameters: {', '.join(missing)}", status_code=400)

        try:
            server.select_destination(session_id, place, building, floor, destination_id)
        except Exception as e:
            logging.error(f"Error selecting destination {destination_id} for session {session_id}: {str(e)}")
            return api_response(success=False, error="Error selecting destination", status_code=500)
            
        return api_response(success=True, message="Destination selected successfully")

    @app.route('/planner', methods=['POST'])
    @handle_exceptions
    def planner():
        """
        Handle a planning request, providing a path and action list based on the current setup.
        
        Request:
            JSON with 'session_id'
            
        Response:
            Success: {'success': True, 'data': {'trajectory': [...]}}
            Error: {'success': False, 'error': 'Error message'}
        """
        data = request.json
        if not data:
            return api_response(success=False, error="No request data provided", status_code=400)
            
        session_id = data.get('session_id')
        if not session_id:
            return api_response(success=False, error="Missing required parameter: session_id", status_code=400)
        
        try:
            navigation_start_time = time.time()
            trajectory, _ = server.handle_navigation(session_id)
            time_logger.log_navigation_time(navigation_start_time, trajectory)
            
            socketio.emit('planner_update', {'trajectory': trajectory})
            
            return api_response(success=True, data={"trajectory": trajectory})
        except ValueError as e:
            # Re-raise to be caught by the handle_exceptions decorator
            raise
        except Exception as e:
            logging.error(f"Error planning route for session {session_id}: {str(e)}", exc_info=True)
            return api_response(success=False, error="Error planning route", status_code=500)
