import os
import json
import time
import socket
import io
import base64
import logging
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any, Union

from utils import DataHandler, CacheManager
from UNav_core.src.track import Coarse_Locator, localization
from UNav_core.src.navigation import Trajectory, actions

# Constants
COARSE_LOCALIZE_THRESHOLD = 5  # Threshold for coarse localization failures
TIMEOUT_SECONDS = 600  # Timeout for coarse localization in seconds

class Server(DataHandler):
    """
    Main server class that handles localization, navigation, and data management for the UNav system.
    
    This class extends DataHandler to leverage its data loading capabilities and adds functionality
    for real-time localization, path planning, and state management across multiple user sessions.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger, feature: str):
        """
        Initialize the Server with configuration, logging, and feature extraction settings.
        
        Args:
            config: Dictionary containing server configuration parameters
            logger: Logger instance for recording server activity
            feature: Feature extraction method to use for localization
            
        Raises:
            ValueError: If critical configuration parameters are missing
            ConnectionError: If socket binding fails
        """
        try:
            # Initialize parent DataHandler class
            super().__init__(config["IO_root"], config['location']['place'], feature)
            
            # Store configuration and setup logging
            self.config = config
            self.logger = logger
            self.root = config["IO_root"]
            
            # Setup socket for network communication
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.bind((config["server"]["host"], config["server"]["port"]))
                self.sock.listen(5)
            except (socket.error, OSError) as e:
                raise ConnectionError(f"Failed to bind socket: {str(e)}")
                
            # Configure localization settings
            self.load_all_maps = config['hloc']['load_all_maps']
                
            # Initialize localization components
            self.logger.info(f"Initializing localization with feature: {feature}")
            self.coarse_locator = Coarse_Locator(feature, config=self.config)
            self.refine_locator = localization(self.coarse_locator, config=self.config, logger=self.logger)
            
            # Initialize trajectory planning
            self.trajectory_maker = Trajectory(self.all_buildings_data, self.all_interwaypoint_connections)
            
            # Initialize cache and state management
            self.cache_manager = CacheManager()
            self.localization_states = {}  # Maps session_id to localization state
            self.destination_states = {}   # Maps session_id to destination state

            # Load scale data for all buildings
            try:
                scale_path = os.path.join(self.root, 'data', 'scale.json')
                with open(scale_path, 'r') as f:
                    self.scale_data = json.load(f)
                self.logger.info("Scale data loaded successfully")
            except (FileNotFoundError, json.JSONDecodeError) as e:
                self.logger.error(f"Failed to load scale data: {str(e)}")
                self.scale_data = {}
        except Exception as e:
            self.logger.error(f"Error initializing server: {str(e)}", exc_info=True)
            raise

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

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update the server configuration with new location settings.
        
        This method updates the location configuration and retrieves the corresponding scale
        for the specified place, building, and floor combination.
        
        Args:
            new_config: Dictionary containing updated configuration parameters
                        Must include 'place', 'building', and 'floor' keys
                        
        Returns:
            None
            
        Logs:
            - Info when configuration is updated successfully
            - Warning when scale data is not found for the location
        """
        try:
            # Extract location parameters
            place = new_config.get('place')
            building = new_config.get('building')
            floor = new_config.get('floor')
            
            if not all([place, building, floor]):
                self.logger.warning(f"Incomplete location data in config update: {new_config}")
            
            # Retrieve scale for the new location
            new_scale = self.scale_data.get(place, {}).get(building, {}).get(floor, None)
            
            if new_scale is None:
                self.logger.warning(f"Scale not found for {place}/{building}/{floor}")
                
            # Update configuration
            new_config['scale'] = new_scale
            self.config['location'] = new_config
            self.root = self.config["IO_root"]
            
            self.logger.info(f"Configuration updated to {place}/{building}/{floor} with scale {new_scale}")
        except Exception as e:
            self.logger.error(f"Error updating configuration: {str(e)}", exc_info=True)
            raise

    def terminate(self, session_id: str) -> None:
        """
        Terminate a user session and release associated resources.
        
        This method cleans up all resources associated with a session, including:
        - Localization state
        - Destination state
        - Trajectory planning state
        - Cached map segments
        
        Args:
            session_id: Unique identifier for the user session to terminate
            
        Returns:
            None
            
        Logs:
            - Info when session termination starts/completes
            - Debug details about resources being released
        """
        try:
            self.logger.info(f"Terminating session of {session_id}...")
            
            # Get the segment_id associated with this session
            segment_id = self.localization_states.get(session_id, {}).get('segment_id', None)
            
            if segment_id:
                # Release map segments from cache
                try:
                    connection_data = self.coarse_locator.connection_graph.get(segment_id, {})
                    current_neighbors = list(connection_data.get('adjacent_segment', []))
                    segments_to_release = [segment_id] + current_neighbors
                    
                    self.logger.debug(f"Releasing segments: {segments_to_release}")
                    self.cache_manager.release_segments(session_id, segments_to_release)
                except Exception as e:
                    self.logger.warning(f"Error releasing segments: {str(e)}")
                
                # Clean up session states
                if session_id in self.localization_states:
                    del self.localization_states[session_id]
                    self.logger.debug(f"Removed localization state for {session_id}")
                
                if session_id in self.destination_states:
                    del self.destination_states[session_id]
                    self.logger.debug(f"Removed destination state for {session_id}")
                
                if session_id in self.trajectory_maker.sessions:
                    del self.trajectory_maker.sessions[session_id]
                    self.logger.debug(f"Removed trajectory state for {session_id}")
            else:
                self.logger.info(f"No active segment found for session {session_id}")
                
            self.logger.info(f"Session of {session_id} terminated successfully.")
        except Exception as e:
            self.logger.error(f"Error terminating session {session_id}: {str(e)}", exc_info=True)
            # Don't re-raise the exception to ensure cleanup continues even if parts fail

    def coarse_localize(self, query_image: np.ndarray) -> Optional[str]:
        """
        Perform coarse localization to determine the segment ID for a query image.
        
        This method attempts to identify which segment (area of a building/floor) the query image belongs to
        using visual place recognition techniques.
        
        Args:
            query_image: Numpy array containing the RGB image data to localize
            
        Returns:
            str: Segment ID if localization is successful, None otherwise
            
        Logs:
            - Debug information about localization attempts
            - Info when localization succeeds or fails
        """
        try:
            self.logger.debug("Performing coarse localization on query image")
            
            if query_image is None or not isinstance(query_image, np.ndarray):
                self.logger.error("Invalid query image provided for localization")
                return None
                
            _, segment_id, success = self.coarse_locator.coarse_vpr(query_image)
            
            if success and segment_id:
                self.logger.info(f"Coarse localization successful. Segment ID: {segment_id}")
                return segment_id
            else:
                self.logger.info("Coarse localization failed")
                return None
        except Exception as e:
            self.logger.error(f"Error in coarse localization: {str(e)}", exc_info=True)
            return None

    def get_floorplan(self, session_id: str) -> Dict[str, str]:
        """
        Retrieve the floorplan image for a user's current location.
        
        The method uses the building and floor information from the user's localization state
        to find and return the appropriate floorplan image encoded in base64.
        
        Args:
            session_id: Unique identifier for the user session
            
        Returns:
            Dict containing the floorplan image encoded in base64
            
        Raises:
            ValueError: If the session ID is invalid or no localization data exists
            FileNotFoundError: If the floorplan image cannot be found
            
        Logs:
            - Error if floorplan retrieval fails
            - Debug information about floorplan path
        """
        try:
            # Get localization state for this session
            loc_state = self.localization_states.get(session_id)
            if not loc_state:
                raise ValueError(f"No localization state found for session {session_id}")
                
            # Extract building and floor from localization state
            building = loc_state.get('building')
            floor = loc_state.get('floor')
            
            if not building or not floor:
                raise ValueError(f"Building or floor not set for session {session_id}")
                
            # Get current location configuration
            location_config = self.config['location']
            place = location_config.get('place')
            
            # Construct path to floorplan image
            floorplan_path = os.path.join(
                self.new_root_dir, 'data', place, building, floor, 'floorplan.png'
            )
            self.logger.debug(f"Loading floorplan from: {floorplan_path}")
            
            # Check if file exists
            if not os.path.exists(floorplan_path):
                raise FileNotFoundError(f"Floorplan not found at {floorplan_path}")
                
            # Load and convert floorplan image
            floorplan = Image.open(floorplan_path).convert("RGB")

            # Convert floorplan image to base64
            buffer = io.BytesIO()
            floorplan.save(buffer, format="PNG")
            floorplan_base64 = base64.b64encode(buffer.getvalue()).decode()

            return {
                'floorplan': floorplan_base64,
            }
        except Exception as e:
            self.logger.error(f"Error retrieving floorplan: {str(e)}", exc_info=True)
            raise

    def localize(self, query_image: np.ndarray) -> Optional[List[int]]:
        """
        Localize a query image and return the pose (x, y, angle).
        
        This is the main entry point for image-based localization, which:
        1. Creates a unique session ID if one doesn't exist
        2. Performs coarse localization to find the segment
        3. Performs fine localization to find the precise pose
        
        Args:
            query_image: Numpy array containing the RGB image data to localize
            
        Returns:
            List of 3 integers [x, y, angle] if localization is successful, None otherwise
            
        Logs:
            - Info about localization attempts and results
            - Performance metrics for localization operations
        """
        try:
            # Generate a unique session ID for this localization
            session_id = f"localize_{int(time.time())}_{id(query_image) % 10000}"
            self.logger.info(f"Starting localization for session {session_id}")
            
            # Check if input is valid
            if query_image is None or not isinstance(query_image, np.ndarray):
                self.logger.error("Invalid query image for localization")
                return None
                
            # Start timing the operation
            start_time = time.time()
            
            # Perform coarse localization
            segment_id = self.coarse_localize(query_image)
            if not segment_id:
                self.logger.info("Localization failed at coarse stage")
                return None
                
            # Initialize localization state for this session
            building, floor = self._split_id(segment_id)
            self.localization_states[session_id] = {
                'building': building,
                'floor': floor,
                'segment_id': segment_id,
                'pose': None,
                'failures': 0,
                'last_success_time': time.time()
            }
            
            # Process the image and get location data
            location_data = self.handle_localization(session_id, query_image)
            pose = location_data.get('pose')
            
            # Log performance metrics
            elapsed_time = time.time() - start_time
            self.logger.info(f"Localization completed in {elapsed_time:.2f} seconds. Result: {pose is not None}")
            
            return pose
        except Exception as e:
            self.logger.error(f"Error in localization: {str(e)}", exc_info=True)
            return None
            
    def get_destinations_list(self, building: str, floor: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve the list of destinations available in a specific building and floor.
        
        Args:
            building: Building identifier
            floor: Floor identifier
            
        Returns:
            Dictionary containing the list of destinations with their IDs, names, and locations
            
        Raises:
            KeyError: If the building or floor data is not found
            
        Logs:
            - Debug information about the requested building/floor
            - Error if destination data cannot be retrieved
        """
        try:
            self.logger.debug(f"Retrieving destinations for {building}/{floor}")
            
            # Load destination data from building data
            building_data = self.all_buildings_data.get(building)
            if not building_data:
                raise KeyError(f"Building {building} not found in data")
                
            floor_data = building_data.get(floor)
            if not floor_data:
                raise KeyError(f"Floor {floor} not found in building {building}")
                
            destinations = floor_data.get('destinations', {})
            
            # Format destinations for response
            destinations_data = [
                {'name': dest_info['name'], 'id': dest_id, 'location': dest_info['location']}
                for dest_id, dest_info in destinations.items()
            ]
            
            # Sort destinations by name for consistency
            destinations_data = sorted(destinations_data, key=lambda x: x['name'])
            
            self.logger.info(f"Retrieved {len(destinations_data)} destinations for {building}/{floor}")
            
            return {
                'destinations': destinations_data,
            }
        except Exception as e:
            self.logger.error(f"Error retrieving destinations: {str(e)}", exc_info=True)
            raise

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
