import modal
from pathlib import Path

volume = modal.Volume.from_name("Visiondata")

file_path = Path("CricaVPR.pth")


with volume.batch_upload() as batch:
    batch.put_file(str(file_path), "parameters/CricaVPR/ckpts/CricaVPR.pth")
