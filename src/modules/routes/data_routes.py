from flask import request, jsonify
import os
import base64
from PIL import Image
import io
import numpy as np
from utils.time_logger import TimeLogger
import time

time_logger = TimeLogger()

def register_data_routes(app, server, socketio):

    @app.route('/localize', methods=['POST'])
    def localize():
        """
        Handle localization request by processing the provided image and returning the pose.
        """
        data = request.json
        query_image_base64 = data.get('query_image')

        if not query_image_base64:
            return jsonify({'error': 'No image provided'}), 400

        # Decode the base64 image
        query_image_data = base64.b64decode(query_image_base64.split(',')[1]) if ',' in query_image_base64 else base64.b64decode(query_image_base64)
        query_image = Image.open(io.BytesIO(query_image_data)).convert('RGB')

        # Localize the image using the server's method
        pose = server.localize(np.array(query_image))
        rounded_pose = [int(coord) for coord in pose] if pose else None

        return jsonify({'pose': rounded_pose})

    @app.route('/list_images', methods=['GET'])
    def get_images_list():
        """
        Return testing images list.
        """
        config = server.config['location']
        target_place = config['place']
        target_building = config['building']
        target_floor = config['floor']
        
        # Build the path based on the place, building, and floor
        data_path = os.path.join(server.root, "logs", target_place, target_building, target_floor)
        
        if not os.path.exists(data_path):
            return jsonify({"error": "Invalid path"}), 400
        
        images_dict = {}
        
        # Traverse through directories and collect image names
        for root, ids, _ in os.walk(data_path):
            # We are interested in directories that match ids (like '00150')
            for id in ids:
                image_dir = os.path.join(root, id, 'images')
                try:
                    files = sorted(os.listdir(image_dir))
                    images_dict[id] = [f for f in files if f.endswith('.png')]
                except:
                    pass
        
        return jsonify(images_dict), 200
    
    @app.route('/get_options', methods=['GET'])
    def get_options():
        """
        Return a dictionary of available places, buildings, and floors for the client to choose from.
        """
        data_path = os.path.join(server.root, "data")
        places = [place for place in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, place))]

        options = {}
        for place in places:
            place_path = os.path.join(data_path, place)
            buildings = [building for building in os.listdir(place_path) if os.path.isdir(os.path.join(place_path, building))]
            options[place] = {building: [floor for floor in os.listdir(os.path.join(place_path, building)) if os.path.isdir(os.path.join(place_path, building, floor))] for building in buildings}

        return jsonify(options)

    @app.route('/list_places', methods=['GET'])
    def list_places():
        """
        List all available places stored on the server.
        """
        data_path = os.path.join(server.root, "data")
        places = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        return jsonify({'places': places})

    @app.route('/list_buildings/<place>', methods=['GET'])
    def list_buildings(place):
        """
        List all buildings available within a given place.
        """
        place_path = os.path.join(server.root, "data", place)
        buildings = [d for d in os.listdir(place_path) if os.path.isdir(os.path.join(place_path, d))]
        return jsonify({'buildings': buildings})

    @app.route('/list_floors/<place>/<building>', methods=['GET'])
    def list_floors(place, building):
        """
        List all floors available within a given building in a specified place.
        """
        building_path = os.path.join(server.root, "data", place, building)
        floors = [d for d in os.listdir(building_path) if os.path.isdir(os.path.join(building_path, d))]
        return jsonify({'floors': floors})

    # @app.route('/get_scale', methods=['POST'])
    # def get_scale():
    #     """
    #     Retrieve the scale for a specified place, building, and floor.
    #     """
    #     data = request.json
    #     place = data.get('place')
    #     building = data.get('building')
    #     floor = data.get('floor')
    #     session_id = data.get('session_id')

    #     scale = server.get_scale(place, building, floor, session_id)
    #     return jsonify({'scale': scale})

    @app.route('/get_destinations', methods=['POST'])
    def get_destinations_list():
        """
        Retrieve available destinations for the current location.
        """
        data = request.json
        
        place = data.get('place')
        building = data.get('building')
        floor = data.get('floor')
        
        destination_data = server.get_destinations_list(building, floor)
        return jsonify(destination_data)

    @app.route('/get_floorplan', methods=['POST'])
    def get_floorplan():
        """
        Retrieve the floorplan for the current location.
        """
        data = request.json
        session_id = data.get('session_id')
        
        floorplan_base64 = server.get_floorplan(session_id)
        return jsonify(floorplan_base64)
    
    @app.route('/select_destination', methods=['POST'])
    def select_destination():
        """
        Handle the selection of a destination by the client.
        """
        data = request.json
        
        place = data.get('place')
        building = data.get('building')
        floor = data.get('floor')
        destination_id = data.get('destination_id')
        session_id = data.get('session_id')

        if not destination_id:
            return jsonify({'error': 'Missing destination ID'}), 400

        server.select_destination(session_id, place, building, floor, destination_id)
        return jsonify({'status': 'success'})

    @app.route('/planner', methods=['POST'])
    def planner():
        """
        Handle a planning request, providing a path and action list based on the current setup.
        """
        # try:
        data = request.json
        session_id = data.get('session_id')
        
        navigation_start_time = time.time()
        trajectory,_ = server.handle_navigation(session_id)
        time_logger.log_navigation_time(navigation_start_time, trajectory)
        
        socketio.emit('planner_update', {'trajectory': trajectory})
        return jsonify({'trajectory': trajectory})
        # except ValueError as e:
        #     return jsonify({'error': str(e)}), 400
