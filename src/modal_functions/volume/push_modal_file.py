import modal
from pathlib import Path

volume = modal.Volume.from_name("NewVisiondata")

file_path = Path("dino_salad.ckpt")


with volume.batch_upload() as batch:
    batch.put_file(str(file_path), "parameters/DinoV2Salad/ckpts/dino_salad.ckpt")
