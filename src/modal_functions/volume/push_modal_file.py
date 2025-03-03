import modal
from pathlib import Path

volume = modal.Volume.from_name("NewVisiondata")

file_path = Path("MapConnnection_Graph.pkl")


with volume.batch_upload() as batch:
    batch.put_file(str(file_path), "data/New_York_City/MapConnnection_Graph.pkl")
