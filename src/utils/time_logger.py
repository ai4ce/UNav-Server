import csv
import time
import json
from datetime import datetime

class TimeLogger:
    def __init__(self, filename="experiment_times.csv", temp_file="temp_localization.json"):
        self.filename = filename
        self.temp_file = temp_file
        # Initialize the CSV file with headers if it doesn't exist
        with open(self.filename, mode="a", newline='') as file:
            writer = csv.writer(file)
            file.seek(0, 2)  # Move to the end of the file
            if file.tell() == 0:  # Check if the file is empty
                writer.writerow([
                    "Timestamp", "Session ID", "Building", "Floor", "Localization Time (s)", 
                    "Localization Success", "Navigation Time (s)", "Navigation Success"
                ])

    def log_localization_time(self, session_id, building, floor, start_time, pose_update_info):
        # Set 'N/A' if building or floor is None
        building = building if building is not None else "N/A"
        floor = floor if floor is not None else "N/A"
        
        # Calculate elapsed time for localization
        localization_duration = round(time.time() - start_time, 2)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Determine success or failure based on pose_update_info
        localization_success = "Success" if pose_update_info.get('pose') is not None else "Failed"

        # Save to temporary file for later use by navigation logging
        with open(self.temp_file, "w") as temp:
            json.dump({
                "timestamp": timestamp,
                "session_id": session_id,
                "building": building,
                "floor": floor,
                "localization_duration": localization_duration,
                "localization_success": localization_success
            }, temp)
        
        # Log the initial data to the CSV
        with open(self.filename, mode="a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, session_id, building, floor, localization_duration, localization_success, "", ""])

        return timestamp

    def log_navigation_time(self, navigation_start_time, trajectory):
        # Load data from the temporary file
        with open(self.temp_file, "r") as temp:
            data = json.load(temp)

        # Calculate navigation duration
        navigation_duration = round(time.time() - navigation_start_time, 2)

        # Determine navigation success or failure
        navigation_success = "Success" if trajectory else "Failed"

        # Update CSV with navigation duration and success status
        rows = []
        with open(self.filename, mode="r", newline='') as file:
            reader = csv.reader(file)
            rows = list(reader)
            for row in rows:
                if row[0] == data["timestamp"] and row[6] == "":
                    row[6] = navigation_duration  # Update the navigation time
                    row[7] = navigation_success  # Update the navigation success
                    break
        with open(self.filename, mode="w", newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)
