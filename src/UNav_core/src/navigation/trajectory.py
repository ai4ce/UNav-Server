import numpy as np
from scipy.sparse.csgraph import shortest_path
from collections import defaultdict, deque
import math

class Navigator:
    def __init__(self):
        pass
    
    def compute_distance(self, p1, p2, scale):
        """Compute real-world distance between two points on the floorplan."""
        
        print(f"Scale: {scale} 📏")
        print(f"Scale type : {type(scale)}")
        
        pixel_distance = math.sqrt((p2[0] - p1[0]) ** 2 + (p1[1] - p2[1]) ** 2)  # Adjusted y-axis
        return pixel_distance * scale

    def compute_angle(self, p1, p2):
        """Compute the adjusted angle (in degrees) required to move from p1 to p2."""
        return (math.degrees(math.atan2(p1[1] - p2[1], p2[0] - p1[0])) + 270) % 360  # Adjusted y-axis
    
    def compute_relative_angle(self, current_angle, target_angle):
        """Compute the relative turn angle needed from the current orientation."""
        relative_angle = (target_angle - current_angle) % 360
        if relative_angle > 180:
            relative_angle -= 360
        return relative_angle
    
    def angle_to_clock(self, relative_angle):
        """Convert relative angle to a 12-hour clock format with correct mapping (counterclockwise is positive)."""
        clock_positions = [
            (0, 15, "12 o'clock"), (15, 45, "11 o'clock"), (45, 75, "10 o'clock"),
            (75, 105, "9 o'clock"), (105, 135, "8 o'clock"), (135, 165, "7 o'clock"),
            (165, 195, "6 o'clock"), (-180, -165, "6 o'clock"), (-165, -135, "5 o'clock"),
            (-135, -105, "4 o'clock"), (-105, -75, "3 o'clock"), (-75, -45, "2 o'clock"),
            (-45, -15, "1 o'clock"), (-15, 0, "12 o'clock")
        ]
        for low, high, clock in clock_positions:
            if low <= relative_angle < high:
                return clock
        return "12 o'clock"
    
    def commander(self, current_pose, following_poses, scale, are_instructions_generated=True):
        """Generate navigation instructions based on the current pose and the following path."""
        instructions = []
        x, y, ang = current_pose
        
        for idx, (next_x, next_y) in enumerate(following_poses):
            target_angle = self.compute_angle((x, y), (next_x, next_y))
            relative_turn = self.compute_relative_angle(ang, target_angle)
            clock_direction = self.angle_to_clock(relative_turn)
            distance = self.compute_distance((x, y), (next_x, next_y), scale)
            
            # Generate instruction ensuring forward motion after turn
            if abs(relative_turn) > 10:  # Only mention turns if they are significant
                instruction = f"Turn to {clock_direction}."
                instructions.append(instruction)
                ang = target_angle  # Update current angle after turn
            
            instruction = f"Walk {distance:.1f} meters forward."
            instructions.append(instruction)
            
            # Update current position for next iteration
            x, y = next_x, next_y
        
        if are_instructions_generated:
            return {
                "instructions": instructions,
                "are_instructions_generated": True
            }
        else:
            return {
                "are_instructions_generated": False
            }
            
class Trajectory():
    def __init__(self, all_buildings_data, all_interwaypoint_connections):
        self.all_buildings_data = all_buildings_data
        self.all_interwaypoint_connections = all_interwaypoint_connections
        self.sessions = defaultdict(dict)
        self.precalculated_inter_paths = self._precalculated_inter_paths()
        self.precalculated_global_paths = self._precalculate_global_paths()
        self.navigator = Navigator()

    def _find_all_paths(self, current_building, current_floor, dest_building, dest_floor):
        paths = []

        # Helper function to get valid next waypoints
        def get_next_waypoints(current_waypoint, visited):
            next_waypoints = []
            for inter_id, waypoints in self.all_interwaypoint_connections.items():
                for waypoint in waypoints:
                    # Check if it's a valid transition (same building/floor or connected via interwaypoints)
                    if waypoint['building'] == current_waypoint['building'] and waypoint['floor'] == current_waypoint['floor']:
                        if waypoint['waypoint'] != current_waypoint['waypoint'] and waypoint not in visited:
                            next_waypoints.append(waypoint)
                    elif waypoint['id'] == current_waypoint['id'] and waypoint not in visited:
                        next_waypoints.append(waypoint)
            return next_waypoints
        
        # BFS search for paths
        queue = deque()

        # Enqueue all interwaypoints in the current building and floor
        for inter_id, waypoints in self.all_interwaypoint_connections.items():
            for waypoint in waypoints:
                if waypoint['building'] == current_building and waypoint['floor'] == current_floor:
                    queue.append((waypoint, [waypoint], defaultdict(int)))  # (current_waypoint, path_so_far, id_count)

        while queue:
            current_waypoint, path_so_far, id_count = queue.popleft()

            # Update the count of the current waypoint's id
            id_count[current_waypoint['id']] += 1
            
            # If any id is visited more than twice, skip this path
            if id_count[current_waypoint['id']] > 2:
                continue

            # If we've reached the destination building and floor, check if the path is valid
            if current_waypoint['building'] == dest_building and current_waypoint['floor'] == dest_floor:
                # Ensure each id has been visited exactly twice
                if all(count == 2 for count in id_count.values()):
                    paths.append(path_so_far)
                continue
            
            # Get next valid waypoints
            next_waypoints = get_next_waypoints(current_waypoint, path_so_far)
            for next_waypoint in next_waypoints:
                # Pass along the current id_count dictionary
                new_id_count = id_count.copy()
                queue.append((next_waypoint, path_so_far + [next_waypoint], new_id_count))
        
        return paths

    # Function to precalculate all possible paths between every building and floor combination
    def _precalculate_global_paths(self):
        # Store the precalculated paths
        precalculated_paths = defaultdict(dict)

        # Use keys from self.all_buildings_data to get all building and floor combinations
        for current_building, floors in self.all_buildings_data.items():
            for current_floor, _ in floors.items():
                for dest_building, dest_floors in self.all_buildings_data.items():
                    for dest_floor, _ in dest_floors.items():
                        # Skip if the start and destination are the same
                        if current_building == dest_building and current_floor == dest_floor:
                            continue

                        # Calculate all paths between the start and destination
                        paths = self._find_all_paths(current_building, current_floor, dest_building, dest_floor)

                        # Store the result
                        precalculated_paths[(current_building, current_floor)][(dest_building, dest_floor)] = paths

        return precalculated_paths
            
    def _precalculated_inter_paths(self):
        path_between_interwaypoints = defaultdict(dict)
        
        for building, building_data  in self.all_buildings_data.items():
            for floor, floor_data  in building_data.items():
                
                current_M = floor_data.get('access_graph')
                _, Pr = shortest_path(current_M, directed=True, method='FW', return_predecessors=True)
                
                interwaypoints = floor_data.get('interwaypoints')
                
                interwaypoints_index_in_floor = [interwaypoint.get('index') for interwaypoint in interwaypoints]
                
                anchor_location =  self._form_anchor_points(floor_data.get('destinations'), floor_data.get('waypoints'))
                
                for current_index in interwaypoints_index_in_floor:
                    for target_index in interwaypoints_index_in_floor:
                        if current_index < target_index:
                            path_between_interwaypoints[(building, floor)][(current_index, target_index)] = self._trace_back_path(Pr, anchor_location, current_index, target_index)
                            path_between_interwaypoints[(building, floor)][(target_index, current_index)] = path_between_interwaypoints[(building, floor)][(current_index, target_index)]
                            
        return path_between_interwaypoints
    
    def _form_anchor_points(self, destinations, waypoints):
        """
        Forms a list of anchor points by extracting the location data from destinations and waypoints.
        
        Args:
            destinations (dict): Dictionary containing destination data.
            waypoints (dict): Dictionary containing waypoint data.
            
        Returns:
            list: A list of anchor points (locations) extracted from destinations and waypoints.
        """
        anchor_points = []

        # Iterate through destinations and extract the location
        for dest_id, dest_info in destinations.items():
            location = dest_info.get('location')
            if location:
                anchor_points.append(location)

        # Iterate through waypoints and extract the location
        for waypoint_id, waypoint_info in waypoints.items():
            location = [waypoint_info.get('x'), waypoint_info.get('y')]
            if location:
                anchor_points.append(location)

        return anchor_points

    def _ccw(self, A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def _distance(self, c, d, current_bounderies):
        for boundary in current_bounderies:
            a, b = [boundary[0], boundary[1]], [boundary[2], boundary[3]]
            if self._ccw(a, c, d) != self._ccw(b, c, d) and self._ccw(a, b, c) != self._ccw(a, b, d):
                return 0
        return np.linalg.norm(np.array(c) - np.array(d))

    def _trace_back_path(self, predecessors, locations, current_index, destination_index):
        """
        Trace back the path from the destination index using the predecessor matrix.
        """
        path_list=[]
        
        index = destination_index
        predecessors=predecessors[index]
        
        index=predecessors[current_index]
        while index!=-9999:
            path_list.append(locations[index])
            index=predecessors[index]

        return path_list
    
    def _calculate_trajectory_length(self, points):
        total_length = 0.0
        
        # Iterate over the list of points, calculating distance between consecutive pairs
        for i in range(1, len(points)):
            x1, y1 = points[i - 1]
            x2, y2 = points[i]
            
            # Calculate the Euclidean distance between consecutive points
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            
            # Add to the total length
            total_length += distance
        
        return total_length

    def _initialize_session(self, session_id):
        """Initialize data for a new session."""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'destination_building': None,
                'destination_floor': None,
                'destination_index': None,
                'destination_name': None,
                'dest_from_inter': defaultdict(dict)
            }
            
    def update_destination_graph(self, session_id, navigation_states):
        """Update the destination graph for a specific session."""
        self._initialize_session(session_id)
        session_data = self.sessions[session_id]

        session_data['destination_building'] = navigation_states['Building']
        session_data['destination_floor'] = navigation_states['Floor']
        
        floor_data = self.all_buildings_data.get(session_data['destination_building'], {}).get(session_data['destination_floor'], {})
        destination_M = floor_data.get('access_graph')
        destination_interwaypoints = floor_data.get('interwaypoints')

        destination_anchor_location = self._form_anchor_points(floor_data.get('destinations'), floor_data.get('waypoints'))

        selected_destination_ID = navigation_states['Selected_destination_ID']
        destination_info = floor_data.get('destinations').get(selected_destination_ID)
        
        session_data['destination_index'] = None
        if destination_info:
            for idx, dest_id in enumerate(floor_data.get('destinations').keys()):
                if dest_id == selected_destination_ID:
                    session_data['destination_index'] = idx
                    session_data['destination_name'] = floor_data.get('destinations', {}).get(dest_id, {}).get('name', 'destination')
                    break

        _, Pr = shortest_path(destination_M, directed=True, method='FW', return_predecessors=True)

        for interwaypoint in destination_interwaypoints:
            session_data['dest_from_inter'][interwaypoint['index']] = self._trace_back_path(Pr, destination_anchor_location, interwaypoint['index'], session_data['destination_index'])
    
    def calculate_path(self, manager, session_id, localization_states):
        """Calculate the path for a specific session."""
        self._initialize_session(session_id)
        session_data = self.sessions[session_id]

        current_building = localization_states['building']
        current_floor = localization_states['floor']
        current_pose = localization_states['pose']
        
        floor_data = self.all_buildings_data.get(current_building, {}).get(current_floor, {})
        current_boundaries = floor_data.get('boundaries')
        current_M = floor_data.get('access_graph')
        current_anchor_location = self._form_anchor_points(floor_data.get('destinations'), floor_data.get('waypoints'))

        # Update the access graph for the current pose
        for i, loc in enumerate(current_anchor_location):
            current_M[i, -1] = self._distance(current_pose[:2], loc, current_boundaries)

        _, Pr = shortest_path(current_M, directed=True, method='FW', return_predecessors=True)
        
        trajectory = defaultdict(dict)

        # Check if current building and floor match the destination
        if (current_building, current_floor) == (session_data['destination_building'], session_data['destination_floor']) and session_data['destination_index'] is not None:
            scale = manager.scale_data.get(manager.config['location']['place'], {}).get(current_building, {}).get(current_floor, None)
            following_paths = self._trace_back_path(Pr, current_anchor_location, -1, session_data['destination_index'])
            command_instruction = self.navigator.commander(current_pose, following_paths, scale)
            trajectory[0] = {
                'name': 'destination',
                'building': current_building,
                'floor': current_floor,
                'paths': [[current_pose[0], current_pose[1]]] + following_paths,
                'command': command_instruction,
                'scale': scale
            }
        else:
            possible_paths = self.precalculated_global_paths.get((current_building, current_floor), {}).get((session_data['destination_building'], session_data['destination_floor']), {})

            min_distance = float('inf')
            for path_candidate in possible_paths:
                enter_id = set()
                local_trajectory = defaultdict(dict)
                distance = 0
                for step, step_info in enumerate(path_candidate):
                    inter_name = step_info.get('name')
                    building = step_info.get('building')
                    floor = step_info.get('floor')
                    scale = manager.scale_data.get(manager.config['location']['place'], {}).get(building, {}).get(floor, None)
                    if step == 0:
                        current_paths = [[current_pose[0], current_pose[1]]] + self._trace_back_path(Pr, current_anchor_location, -1, step_info.get('index'))
                        distance += self._calculate_trajectory_length(current_paths)
                    elif step == len(path_candidate) - 1:
                        current_paths = session_data['dest_from_inter'].get(step_info.get('index'))
                        distance += self._calculate_trajectory_length(current_paths)
                    else:
                        inter_id = step_info.get('id')
                        inter_index = step_info.get('index')
                        if inter_id not in enter_id:
                            enter_id.add(inter_index)
                            target_index = step_info.get('index')
                            current_paths = self.precalculated_inter_paths.get((building, floor)).get((step_info.get('index'), target_index))
                            distance += self._calculate_trajectory_length(current_paths)
                        else:
                            continue
                    command_instruction = self.navigator.commander(current_pose, current_paths[1:], scale)
                    local_trajectory[step] = {
                        'name': inter_name if step < len(path_candidate) - 1 else session_data['destination_name'],
                        'building': building,
                        'floor': floor,
                        'paths': current_paths,
                        'command': command_instruction,
                        'scale': manager.scale_data.get(manager.config['location']['place'], {}).get(building, {}).get(floor, None)
                    }
                if distance < min_distance:
                    min_distance = distance
                    trajectory = local_trajectory
        return trajectory