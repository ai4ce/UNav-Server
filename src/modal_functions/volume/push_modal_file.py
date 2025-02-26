import modal
from pathlib import Path

volume = modal.Volume.from_name("NewVisiondata")

file_path = Path("path.h5")


with volume.batch_upload() as batch:
    batch.put_file(
        str(file_path), "data/New_York_City/LightHouse/6_floor/path.h5"
    )
