import modal
from pathlib import Path
import os

volume = modal.Volume.from_name("NewVisiondata")


file_mappings = [
    ("boundaries.json", "data/New_York_City/LightHouse/6_floor/boundaries.json"),
    ("intersection.json", "data/New_York_City/LightHouse/6_floor/intersection.json"),
    ("destination.json", "data/New_York_City/LightHouse/6_floor/destination.json"),
    ("access_graph.npy", "data/New_York_City/LightHouse/6_floor/access_graph.npy"),

]


def upload_files():
    with volume.batch_upload() as batch:
        for local_path, remote_path in file_mappings:
            local_file_path = Path(local_path)
            if local_file_path.exists():
                print(f"Uploading {local_path} to {remote_path}...")
                batch.put_file(str(local_file_path), remote_path)
            else:
                print(f"Warning: Local file {local_path} not found, skipping.")

    print("Upload complete.")


if __name__ == "__main__":
    upload_files()
