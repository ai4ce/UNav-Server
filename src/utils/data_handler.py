import logging
from PIL import Image, ImageDraw
import numpy as np
from os import listdir
from os.path import join, exists, isdir
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import json
import h5py
import torch
from collections import defaultdict

def load_destination(path):
    with open(path, 'r') as f:
        destinations = json.load(f)
    return destinations

def load_boundaires(path):
    with open(path, 'r') as f:
        data = json.load(f)
        lines = data['lines']
        add_lines = data['add_lines']
        for i in add_lines:
            lines.append(i)
        destinations = data['destination']
        anchor_name,anchor_location=[],[]
        for k, v in destinations.items():
            ll = k.split('-')
            anchor_name.append(v['id'])
            anchor_location.append([int(ll[0]), int(ll[1])])
        for k, v in data['waypoints'].items():
            anchor_name.append(k)
            anchor_location.append(v['location'])
    return anchor_name,anchor_location,lines

class DataHandler:
    def __init__(self, new_root_dir, place, feature):
        self.new_root_dir = new_root_dir
        self.place = place
        self.feature = feature
        self._setup_logging()
        self.all_buildings_data, self.all_interwaypoint_connections = self._load_global_graph()

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def _get_building_floor(self, segment_id):
        # Extract building, floor, and segment number from the segment_id
        parts = segment_id.split('_')
        building = parts[0]  # Extract building name
        floor = parts[1] + '_' + parts[2]  # Extract floor name (e.g., '6_floor')
        segment_number = parts[3] + '_' + parts[4]  # Get the 'Segment_00021' part
        return building, floor, segment_number

    def _load_floor_data(self, floor_dir, floor_name, building):
        """
        Load and format waypoints, destinations, and access graph for a given floor.
        
        Args:
            floor_dir (str): Directory path of the floor.
            floor_name (str): Name of the floor (e.g., '6_floor').
        
        Returns:
            dict: A dictionary containing waypoints, destinations, access graph, and interwaypoints.
        """
        boundaries_file = join(floor_dir, 'boundaries_interwaypoint.json')
        access_graph_file = join(floor_dir, 'access_graph.npy')
        
        floor_data = {
            'waypoints': {},
            'destinations': {},
            'access_graph': None,
            'interwaypoints': []
        }
        
        if exists(boundaries_file):
            with open(boundaries_file, 'r') as json_file:
                data = json.load(json_file)
                floor_data['waypoints'] = data.get('waypoints', {})
                floor_data['boundaries'] = data.get('lines', []) + data.get('add_lines', [])
                # Transform the destinations dictionary
                raw_destinations = data.get('destination', {})
                transformed_destinations = {}
                
                destination_num = 0
                for coordinates, details in raw_destinations.items():
                    x, y = map(int, coordinates.split('-'))  # Extract x and y from the 'x-y' key
                    dest_id = details.get('id')
                    name = details.get('name')
                    transformed_destinations[dest_id] = {
                        'location': [x, y],
                        'name': name
                    }
                    destination_num += 1
                    
                floor_data['destinations'] = transformed_destinations  # Assign the transformed data

                # Extract interwaypoints (waypoints with type 'interwaypoint')
                for ind, (waypoint_id, details) in enumerate(floor_data['waypoints'].items()):
                    if details.get('type') == 'interwaypoint':
                        interwaypoint_data = {
                            'waypoint': waypoint_id,
                            'location': details.get('location'),
                            'id': details.get('index'),
                            'index': destination_num + ind,
                            'building': building, 
                            'floor': floor_name,
                            'name': details.get('name')
                        }
                        floor_data['interwaypoints'].append(interwaypoint_data)
        
        if exists(access_graph_file):
            floor_data['access_graph'] = np.load(access_graph_file)
        
        return floor_data

    def _load_all_floors_in_building(self, building_dir, building):
        """
        Load data from all floors within a building and format it into a dictionary.
        
        Args:
            building_dir (str): Directory where all floor data for a building is stored.
        
        Returns:
            dict: A dictionary containing all floors' data, including waypoints, destinations, access graphs, and interwaypoints.
            dict: A dictionary mapping interwaypoints by their index to track connections across floors.
        """
        all_floors_data = {}
        interwaypoint_connections = defaultdict(list)

        for floor in listdir(building_dir):
            floor_dir = join(building_dir, floor)
            if isdir(floor_dir):
                floor_data = self._load_floor_data(floor_dir, floor, building)
                all_floors_data[floor] = floor_data

                for interwaypoint in floor_data['interwaypoints']:
                    interwaypoint_connections[interwaypoint['id']].append(interwaypoint)
        
        return all_floors_data, interwaypoint_connections

    def _load_all_buildings(self, base_dir):
        """
        Load data from all buildings and their floors.
        
        Args:
            base_dir (str): Base directory where all building data is stored.
        
        Returns:
            dict: A dictionary containing all buildings' data, including floors, waypoints, destinations, and interwaypoints.
            dict: A dictionary mapping interwaypoints by their index to track connections across all buildings.
        """
        all_buildings_data = {}
        all_interwaypoint_connections = defaultdict(list)

        for building in listdir(base_dir):
            building_dir = join(base_dir, building)
            if isdir(building_dir):
                all_floors_data, interwaypoint_connections = self._load_all_floors_in_building(building_dir, building)
                all_buildings_data[building] = all_floors_data
                
                for index, connections in interwaypoint_connections.items():
                    all_interwaypoint_connections[index].extend(connections)

        return all_buildings_data, all_interwaypoint_connections

    def _load_global_graph(self):
        # Load all buildings and floors
        base_directory = join(self.new_root_dir, 'data', self.place)
        all_buildings_data, all_interwaypoint_connections = self._load_all_buildings(base_directory)
        return all_buildings_data, all_interwaypoint_connections

    def load_graph(self, building, floor):
        waypoints = self.all_buildings_data[building][floor].get('waypoints',None)
        destinations = self.all_buildings_data[building][floor].get('destinations',None)
        access_graph = self.all_buildings_data[building][floor].get('access_graph',None)
        return destinations, waypoints, access_graph
    
    def load_map(self, segment_id):
        """
        Load a specific map segment from disk based on the segment ID.
        Example segment ID format: LightHouse_6_floor_Segment_00021
        Map file path example: /mnt/data/UNav-IO/data/New_York_City/LightHouse/6_floor/maps/Segment_00021.h5
        """
        # Extract building, floor, and segment number from the segment_id
        building, floor, segment_number = self._get_building_floor(segment_id)

        # Construct the map directory based on the place, building, and floor from the configuration
        map_directory = join(self.new_root_dir, 'data', self.place, building, floor, f'maps_{self.feature}')

        # Define the segment file path
        segment_file = join(map_directory, f"{segment_number}.h5")


        # Initialize a dictionary to store the loaded map data
        map_data = {
            'T': None,
            'rot_base': None,
            'perspective_frames': {},
        }

        try:
            with h5py.File(segment_file, 'r') as h5_file:
                # Load transformation matrix T
                map_data['T'] = h5_file['T'][:]

                # Calculate 'rot_base' for future usage (if necessary)
                T = np.array(map_data['T'])
                map_data['rot_base'] = np.arctan2(T[1, 0], T[0, 0])

                # Iterate over all the frame groups
                for frame_name in h5_file.keys():
                    if frame_name == 'T':
                        continue  # Skip the transformation matrix

                    # Load the frame data for each frame
                    frame_group = h5_file[frame_name]

                    # Load global descriptor
                    global_descriptor = frame_group['global_descriptor'][:]

                    # Load local features
                    local_features_group = frame_group['local_features']
                    keypoints = local_features_group['keypoints'][:]
                    descriptors = local_features_group['descriptors'][:]
                    image_size = local_features_group['image_size'][:]
                    scores = local_features_group['scores'][:]
                    valid_keypoints_index = local_features_group['valid_keypoints_index'][:]

                    # Load landmarks and frame pose
                    landmarks = frame_group['landmarks'][:]
                    frame_pose = frame_group['frame_pose'][:]

                    # Add the frame data into the map data dictionary
                    map_data['perspective_frames'][f'{building}_{floor}_{frame_name}'] = {
                        'segment_id': segment_id,
                        'global_descriptor': global_descriptor,
                        'local_features': {
                            'keypoints': keypoints,
                            'descriptors': descriptors,
                            'image_size': image_size,
                            'scores': scores,
                            'valid_keypoints_index': valid_keypoints_index
                        },
                        'landmarks': landmarks,
                        'frame_pose': frame_pose
                    }

            return map_data

        except Exception as e:
            print(f"Error loading map segment {segment_file}: {e}")
            return None

class DemoData(DataHandler):
    def __init__(self, new_root_dir):
        super().__init__(new_root_dir)
        self.selected_destination_ID = None

    def load_floorplan_image(self):
        floorplan_url = join(self.new_root_dir, 'data', 'New_York_City', 'LightHouse', '6th_floor', 'floorplan.png')
        floorplan = Image.open(floorplan_url).convert("RGB")
        return floorplan
    
    def plot_floorplan_with_destinations(self, floorplan, destinations, anchor_dict):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(floorplan)
        ax.axis('off')

        for anchor_name, anchor_location in anchor_dict.items():
            if not anchor_name.startswith("w_"):
                ax.plot(anchor_location[0], anchor_location[1], 'ro')

        for idx, dest in enumerate(destinations):
            for name, anchor_id in dest.items():
                location = anchor_dict[anchor_id]
                ax.annotate(f"{idx}: {name}", (location[0], location[1]), color='white', fontsize=8, ha='right')
        
        return fig, ax

    def handle_click_event(self, event, fig, ax, floorplan, destinations, anchor_dict, output):
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            distances = [(np.sqrt((x - loc[0])**2 + (y - loc[1])**2), name) for name, loc in anchor_dict.items() if not name.startswith("w_")]
            selected_name = min(distances, key=lambda t: t[0])[1]
            selected_location = anchor_dict[selected_name]
            
            ax.clear()
            ax.imshow(floorplan)
            ax.axis('off')
            
            for anchor_name, anchor_location in anchor_dict.items():
                if not anchor_name.startswith("w_"):
                    ax.plot(anchor_location[0], anchor_location[1], 'ro')

            ax.plot(selected_location[0], selected_location[1], 'go', markersize=15)
            ax.annotate(f"Selected: {selected_name}", (selected_location[0], selected_location[1]), color='green', fontsize=12, ha='right')

            for idx, dest in enumerate(destinations):
                for name, anchor_id in dest.items():
                    location = anchor_dict[anchor_id]
                    ax.annotate(f"{idx}: {name}", (location[0], location[1]), color='white', fontsize=8, ha='right')

            with output:
                output.clear_output()
                print(f"Selected destination: {selected_name}")
            
            # Save the selected destination ID
            self.selected_destination_ID = selected_name

    def select_destination(self, config):
        floorplan = self.load_floorplan_image()
        destinations, anchor_dict = self.extract_data(config)
        fig, ax = self.plot_floorplan_with_destinations(floorplan, destinations, anchor_dict)

        output = widgets.Output()
        display(output)

        def on_click(event):
            self.handle_click_event(event, fig, ax, floorplan, destinations, anchor_dict, output)

        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()

    def __star_vertices(self,center,r, plot_scale):
        out_vertex = [(r*plot_scale * np.cos(2 * np.pi * k / 5 + np.pi / 2- np.pi / 5) + center[0],
                       r*plot_scale * np.sin(2 * np.pi * k / 5 + np.pi / 2- np.pi / 5) + center[1]) for k in range(5)]
        r = r/2
        in_vertex = [(r*plot_scale * np.cos(2 * np.pi * k / 5 + np.pi / 2 ) + center[0],
                      r*plot_scale * np.sin(2 * np.pi * k / 5 + np.pi / 2 ) + center[1]) for k in range(5)]
        vertices = []
        for i in range(5):
            vertices.append(out_vertex[i])
            vertices.append(in_vertex[i])
        vertices = tuple(vertices)
        return vertices

    def show_localization(self, pose):
        floorplan_url = join(self.new_root_dir, 'data', 'New_York_City', 'LightHouse', '6th_floor', 'floorplan.png')
        floorplan = Image.open(floorplan_url).convert("RGB")

        x1 = pose[0] - 80 * np.sin(pose[2] / 180 * np.pi)
        y1 = pose[1] - 80 * np.cos(pose[2] / 180 * np.pi)

        draw_floorplan = ImageDraw.Draw(floorplan)
        draw_floorplan.ellipse((pose[0] - 40, pose[1] - 40, pose[0] + 40, pose[1] + 40), fill=(50, 0, 106))
        draw_floorplan.line([(pose[0], pose[1]), (x1, y1)], fill=(50, 0, 106), width=20)

        return floorplan
    
    def plot_trajectory(self, paths):
        floorplan_url = join(self.new_root_dir, 'data', 'New_York_City', 'LightHouse', '6th_floor', 'floorplan.png')
        floorplan = Image.open(floorplan_url).convert("RGB")
        draw_floorplan = ImageDraw.Draw(floorplan)
        width, _ = floorplan.size
        plot_scale = width / 3400

        # Plot the trajectory path
        for i in range(1, len(paths)):
            x0, y0 = paths[i - 1][:2]
            x1, y1 = paths[i][:2]
            vertices = self.__star_vertices([x0, y0], 15, plot_scale)
            draw_floorplan.polygon(vertices, fill='yellow', outline='red')
            draw_floorplan.line([(x0, y0), (x1, y1)], fill=(0, 255, 0), width=int(5 * plot_scale))

        # Plot the start point as a large circle
        start_x, start_y = paths[0][:2]
        draw_floorplan.ellipse((start_x - 30 * plot_scale, start_y - 30 * plot_scale, 
                                start_x + 30 * plot_scale, start_y + 30 * plot_scale), 
                            fill=(50, 0, 106), outline='black')

        # Plot the end star in red and larger
        end_x, end_y = paths[-1][:2]
        end_vertices = self.__star_vertices([end_x, end_y], 30, plot_scale)
        draw_floorplan.polygon(end_vertices, fill='red', outline='red')

        return floorplan