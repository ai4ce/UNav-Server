import os
import json
from utils import DataHandler, CacheManager

import time

import socket

from UNav_core.src.track import Coarse_Locator, localization
from UNav_core.src.navigation import Trajectory, actions

import io
import base64

import cv2
from PIL import Image
import numpy as np

COARSE_LOCALIZE_THRESHOLD = 5  # Example threshold for coarse localization failures
TIMEOUT_SECONDS = 600  # Example timeout for coarse localization in seconds

class Server(DataHandler):
    def __init__(self, config, logger, feature):
        super().__init__(config["IO_root"], config['location']['place'], feature)
        self.config = config
        self.logger = logger
        self.root = config["IO_root"]
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((config["server"]["host"], config["server"]["port"]))
        self.sock.listen(5)

        self.load_all_maps = config['hloc']['load_all_maps']
            
        self.coarse_locator = Coarse_Locator(feature, config=self.config)
        self.refine_locator = localization(self.coarse_locator, config=self.config, logger=self.logger)
        
        self.trajectory_maker = Trajectory(self.all_buildings_data, self.all_interwaypoint_connections)
        
        self.cache_manager = CacheManager()
        self.localization_states = {}
        self.destination_states = {}

        with open(os.path.join(self.root, 'data', 'scale.json'), 'r') as f:
            self.scale_data = json.load(f)

        ############################################# test data #################################################
        # # Load and process the specific image for debugging
        # image_path = '/mnt/data/UNav-IO/logs/New_York_City/LightHouse/3_floor/New_test_images/20240925_163410.jpg'
        # image = Image.open(image_path)
        
        # original_width, original_height = image.size

        # new_width = 640
        # new_height = int((new_width / original_width) * original_height)

        # # Resize the image
        # resized_image = image.resize((new_width, new_height))

        # image_rgb = resized_image.convert("RGB")
        
        # self.image_np = np.array(image_rgb)
        ############################################# test data #################################################

    def update_config(self, new_config):
        # Merge the new configuration with the existing one
        
        place = new_config.get('place')
        building = new_config.get('building')
        floor = new_config.get('floor')
        
        new_scale = self.scale_data.get(place, {}).get(building, {}).get(floor, None)
        new_config['scale'] = new_scale
        
        self.config['location'] = new_config
        self.root = self.config["IO_root"]

    def terminate(self, session_id):
        self.logger.info(f"Terminating session of {session_id}...")
        
        segment_id = self.localization_states.get(session_id, {}).get('segment_id', None)
        
        if segment_id:
            connection_data = self.coarse_locator.connection_graph.get(segment_id, {})
            current_neighbors = list(connection_data.get('adjacent_segment'))
            segments_to_release = [segment_id] + current_neighbors
            self.cache_manager.release_segments(session_id, segments_to_release)
            del self.localization_states[session_id]
            
            if session_id in self.destination_states:
                del self.destination_states[session_id]
            
            if session_id in self.trajectory_maker.sessions:
                del self.trajectory_maker.sessions[session_id]
            
        self.logger.info(f"Session of {session_id} terminated successfully.")

    def coarse_localize(self, query_image):
        _, SEGMENTID, success = self.coarse_locator.coarse_vpr(query_image)
        if success:
            return SEGMENTID
        else:
            return None

    def get_floorplan(self, session_id): #, building, floor):
        # Load floorplan data
        building, floor = self.localization_states.get(session_id).get('building'), self.localization_states.get(session_id).get('floor')
        location_config=self.config['location']
        floorplan_url = os.path.join(self.new_root_dir, 'data', location_config['place'], building, floor, 'floorplan.png')
        floorplan = Image.open(floorplan_url).convert("RGB")

        # Convert floorplan image to base64
        buffer = io.BytesIO()
        floorplan.save(buffer, format="PNG")
        floorplan_base64 = base64.b64encode(buffer.getvalue()).decode()

        return {
            'floorplan': floorplan_base64,
        }

    def get_destinations_list(self, building, floor):
        # Load destination data
        destinations = self.all_buildings_data.get(building,{}).get(floor,{}).get('destinations',{})

        destinations_data = [
            {'name': dest_info['name'], 'id': dest_id, 'location': dest_info['location']}
            for dest_id, dest_info in destinations.items()
        ]
        
        destinations_data = sorted(destinations_data, key=lambda x: x['name'])

        return {
            'destinations': destinations_data,
        }

    def select_destination(self, session_id, place, building, floor, destination_id):
        self.destination_states[session_id] = {
            'Place': place,
            'Building': building,
            'Floor': floor,
            'Selected_destination_ID': destination_id
        }
        
        self.trajectory_maker.update_destination_graph(session_id, self.destination_states[session_id])
        
        self.logger.info(f"Selected destination ID set to: {destination_id}")

    def list_images(self):
        base_path = os.path.join(self.root, 'logs', self.config['location']['place'], self.config['location']['building'], self.config['location']['floor'])
        ids = os.listdir(base_path)
        images = {id: os.listdir(os.path.join(base_path, id, 'images')) for id in ids if os.path.isdir(os.path.join(base_path, id, 'images'))}
        return images

    def _split_id(self, segment_id):
        # Load the current segment and its neighbors
        parts = segment_id.split('_')
        building = parts[0]  # Extract building name
        floor = parts[1] + '_' + parts[2]  # Extract floor name (e.g., '6_floor')
        return building, floor

    def _update_next_step(self):
        pass

    def handle_localization(self, session_id, frame):
        """
        Handles the localization process for a given session and frame.
        Returns the pose and segment_id if localization is successful.
        localization is done here
        """
        state = self.localization_states.get(session_id, {'failures': 0, 'last_success_time': time.time(), 'building': None, 'floor': None, 'segment_id': None, 'pose': None})
        pose_update_info = {
            'building': None,
            'floor': None,
            'pose': None,
            # 'floorplan_base64': None
        }

        if self.load_all_maps:
            building = self.config["location"]["building"]
            floor = self.config["location"]["floor"]
            
            if not state['building'] and not state['floor']:
                current_cluster = [key for key in self.coarse_locator.connection_graph if key.startswith(building + '_' + floor)]
                
                print(f"Current cluster: {current_cluster}")
                map_data = self.cache_manager.load_segments(self, session_id, current_cluster)
                self.refine_locator.update_maps(map_data)
            pose, next_segment_id = self.refine_locator.get_location(frame)
            
            if pose:
                pose_update_info['pose'] = pose
                
                state['pose'] = pose
                state['segment_id'] = next_segment_id
                
                state['last_success_time'] = time.time()
                
                building, floor = self._split_id(next_segment_id)
                
                if building != state['building'] or floor != state['floor']:
                    state['building'] = building
                    state['floor'] = floor
                    # pose_update_info['floorplan_base64'] = self.get_floorplan(building, floor).get('floorplan', None)

        else:
            time_since_last_success = time.time() - state['last_success_time']
            previous_segment_id = state['segment_id']
            if state['failures'] >= COARSE_LOCALIZE_THRESHOLD or time_since_last_success > TIMEOUT_SECONDS or not state['segment_id']:
                segment_id = self.coarse_localize(frame) #debug
                                
                if segment_id:
                    building, floor = self._split_id(segment_id)
                        
                    connection_data = self.coarse_locator.connection_graph.get(segment_id, {})
                    current_neighbors = list(connection_data.get(segment_id, set()))
                    
                    current_cluster = [segment_id] + current_neighbors
                    
                    map_data = self.cache_manager.load_segments(self, session_id, current_cluster)
                    
                    self.refine_locator.update_maps(map_data)
                    
                    pose, next_segment_id = self.refine_locator.get_location(frame) #debug
                    
                    if pose:
                        pose_update_info['pose'] = pose
                        
                        state['pose'] = pose
                        state['segment_id'] = segment_id
                        state['failures'] = 0
                        state['last_success_time'] = time.time()
                        
                        # if building != state['building'] or floor != state['floor']:
                        # pose_update_info['floorplan_base64'] = self.get_floorplan(building, floor).get('floorplan', None)
                            
                        state['floor'] = floor
                        state['building'] = building
                        
                        if state['segment_id']:
                            # judge if need switch segments
                            if next_segment_id != state['segment_id']:
                                
                                next_building, next_floor = self._split_id(next_segment_id)
                                state['segment_id'] = next_segment_id
                                
                                # if next_building != state['building'] or next_floor != state['floor']:
                                # pose_update_info['floorplan_base64'] = self.get_floorplan(next_building, next_floor).get('floorplan', None)
                                    
                                # delete old segments in cache
                                next_segment_neighbors = list(self.coarse_locator.connection_graph.get(next_segment_id, {}).get('adjacent_segment', set()))
                                segments_to_release = list(set([next_segment_id] + next_segment_neighbors) - set(current_cluster))
                                self.cache_manager.release_segments(session_id, segments_to_release)
                                
                                    
                                state['building'] = next_building
                                state['floor'] = next_floor
                            
                    else:
                        state['pose'] = None
                        state['segment_id'] = None
                        state['floor'] = None
                        state['building'] = None
                        state['failures'] += 1
                        
                    # Release previous segment and its neighbors if they are no longer in use
                    if previous_segment_id and previous_segment_id != segment_id:
                        previous_neighbors = list(self.coarse_locator.connection_graph.get(previous_segment_id, {}).get('adjacent_segment'), set())
                        segments_to_release = list(set([previous_segment_id] + previous_neighbors) - set(current_cluster))
                        self.cache_manager.release_segments(session_id, segments_to_release)
                else:
                    state['failures'] += 1

            else:      
                # Retrieve the current segment and its neighbors from the cache
                connection_data = self.coarse_locator.connection_graph.get(state['segment_id'], {})
                current_neighbors = list(connection_data.get('adjacent_segment', set()))
                current_cluster = [state['segment_id']] + current_neighbors
                
                map_data = self.cache_manager.load_segments(self, session_id, current_cluster)

                self.refine_locator.update_maps(map_data)
                
                pose, next_segment_id = self.refine_locator.get_location(frame) #debug
                
                if pose:

                    # judge if need switch segments
                    next_building, next_floor = self._split_id(next_segment_id)
                    state['building'] = next_building
                    state['floor'] = next_floor
                    if next_segment_id != state['segment_id']:
                        state['segment_id'] = next_segment_id
                        # if next_building != state['building'] or next_floor != state['floor']:
                        # pose_update_info['floorplan_base64'] = self.get_floorplan(next_building, next_floor).get('floorplan', None)

                        # delete old segments in cache
                        next_segment_neighbors = list(self.coarse_locator.connection_graph.get(next_segment_id, {}).get('adjacent_segment', set()))
                        segments_to_release = list(set([next_segment_id] + next_segment_neighbors) - set(current_cluster))
                        self.cache_manager.release_segments(session_id, segments_to_release)

                    pose_update_info['pose'] = pose
                    
                    # pose_update_info['floorplan_base64'] = self.get_floorplan(next_building, next_floor).get('floorplan', None)
                    state['pose'] = pose
                    state['failures'] = 0
                    state['last_success_time'] = time.time()
                else:
                    state['pose'] = None
                    state['floor'] = None
                    state['building'] = None
                    state['failures'] += 1

        self.localization_states[session_id] = state

        pose_update_info['building'] = state['building']
        pose_update_info['floor'] = state['floor']
        
        return pose_update_info

    def handle_navigation(self, session_id):
        if session_id not in self.destination_states:
            self.logger.error("Selected destination ID is not set.")
            raise ValueError("Selected destination ID is not set.")
        if session_id not in self.localization_states:
            self.logger.error("Please do localization first.")
            raise ValueError("Please do localization first.")
        localization_state = self.localization_states.get(session_id)
        pose = localization_state.get('pose')
        if pose:
            trajectory = self.trajectory_maker.calculate_path(self, session_id, localization_state)
            # action_list = actions(trajectory)
            
            if len(trajectory) > 0:
                return trajectory, None
            else:
                return {}, None
        else:
            return {}, None
    
    def preload_maps(self, place=None, building=None, floor=None):
        """
        Preload map data for specified or default place, building, and floor
        using the same caching mechanism as handle_localization.
        """
        if place is None:
            place = self.config['location']['place']
        if building is None:
            building = self.config['location']['building']
        if floor is None:
            floor = self.config['location']['floor']
        
        # Set location config for loading
        self.config['location']['place'] = place
        self.config['location']['building'] = building 
        self.config['location']['floor'] = floor
        
        # Initialize session ID for preloading
        session_id = "preload_session"
        
        # Initialize localization state
        self.localization_states[session_id] = {
            'place': place,
            'building': building,
            'floor': floor,
            'segment_id': None,
            'pose': None,
            'selected_destination_id': None,
            'failures': 0,
            'last_success_time': time.time()
        }
        
        # Load the map data
        self.logger.info(f"Preloading map data for {place}/{building}/{floor}")
        
        try:
                
            # Define a cache file path for modal-level caching
            cache_dir = os.path.join(self.root, 'cache', place, building, floor)
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, 'map_data_cache.pkl')
            
            # Check if cache exists
            if os.path.exists(cache_file):
                self.logger.info(f"Found cached map data for {building}/{floor}, loading from cache")
                import pickle
                with open(cache_file, 'rb') as f:
                    map_data = pickle.load(f)
                self.logger.info(f"Map data loaded from cache, type: {type(map_data)}")
                
                # Update maps with cached data
                start_time = time.time()
                self.refine_locator.update_maps(map_data)
                update_time = time.time() - start_time
                
                print(f"Using cached data: update_maps took {update_time:.3f}s")
                
            else:
                if self.load_all_maps:
                    # Use the same approach as in handle_localization when load_all_maps is True
                    current_cluster = [key for key in self.coarse_locator.connection_graph 
                                    if key.startswith(f"{building}_{floor}")]
                    
                    self.logger.info(f"Preloading {len(current_cluster)} map segments for {building}/{floor}")
                    
                    # Load segments using cache manager
                    start_time = time.time()
                    map_data = self.cache_manager.load_segments(self, session_id, current_cluster)
                    load_time = time.time() - start_time
                    
                    start_time = time.time()
                    self.refine_locator.update_maps(map_data)
                    update_time = time.time() - start_time
                    
                    print(f"load_segments took {load_time:.3f}s, update_maps took {update_time:.3f}s")
                    
                    self.logger.info(f"Map data type: {type(map_data)}")
                
                    # Save to cache for future use
                    import pickle
                    with open(cache_file, 'wb') as f:
                        pickle.dump(map_data, f)
                    self.logger.info(f"Map data cached to {cache_file}")
                
            
                    self.logger.info(f"Successfully preloaded all maps for {building}/{floor}")
                    
                else:
                    # When not loading all maps, preload key segments that are likely to be used first
                    # Find segments for this floor
                    segments = []
                    for segment_id in self.coarse_locator.connection_graph:
                        seg_building, seg_floor = self._split_id(segment_id)
                        if seg_building == building and seg_floor == floor:
                            segments.append(segment_id)
                    
                    # Load primary segments and their neighbors
                    for segment_id in segments:
                        connection_data = self.coarse_locator.connection_graph.get(segment_id, {})
                        neighbors = list(connection_data.get('adjacent_segment', set()))
                        current_cluster = [segment_id] + neighbors
                        
                        self.logger.info(f"Preloading segment {segment_id} and {len(neighbors)} neighbors")
                        map_data = self.cache_manager.load_segments(self, session_id, current_cluster)
                        self.refine_locator.update_maps(map_data)
                    
                    self.logger.info(f"Successfully preloaded {len(segments)} segments and their neighbors")
                        
                # Mark that maps were preloaded
                self._preloaded_maps = True
                
                self.logger.info(f"Map preloading completed for {place}/{building}/{floor}")
                return True
        except Exception as e:
                self.logger.error(f"Error preloading map data: {e}")
                return False