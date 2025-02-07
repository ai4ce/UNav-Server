import modal
from pathlib import Path

volume = modal.Volume.from_name("NewVisiondata")

file_path = Path("MapConnnection_Graph.pkl")


with volume.batch_upload() as batch:
    batch.put_file(str(file_path), "data/nyc/MapConnnection_Graph.pkl")


# import modal
# import os

# volume = modal.Volume.from_name("NewVisiondata")
# app = modal.App()

# @app.function(volumes={"/root/UNav-IO": volume})
# def delete_file_from_volume(file_path: str):
#     """Deletes a file from a Modal volume.

#     Args:
#         file_path: The path to the file within the Modal volume (e.g., /root/UNav-IO/data/myfile.txt).
#     """
#     try:
#         os.remove(file_path)
#         print(f"Deleted {file_path} from volume.")
#     except FileNotFoundError:
#         print(f"File not found: {file_path}")
#     except Exception as e:
#         print(f"Error deleting {file_path}: {e}")


# @app.local_entrypoint()
# def main():
#     file_to_delete = "/root/UNav-IO/data/nyc/MapConnnection_Graph.pkl"  # Replace with the actual path

#     delete_file_from_volume.remote(file_to_delete)
