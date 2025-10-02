import modal
import gdown
import os
import shutil 
import logging
import yaml
import zipfile
from dotenv import load_dotenv
import boto3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Reference the requirements-modal.txt for installing dependencies
image = modal.Image.debian_slim().pip_install_from_requirements("modal_functions/volumesetup_requirements.txt")

volume = modal.Volume.from_name("Visiondata", create_if_missing=True)

app = modal.App("DataSetup", image=image, mounts=[modal.Mount.from_local_file(".env")])
logging.info('created s3_client')
load_dotenv()
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

bucket_name = 'vis4ion'

files = {
    # "demo_query.png": "17MzPE9TyKiNsi6G59rqLCMMd40cIK3bU",
    # "destination.json": "1sIzFujoumSsVlZqlwwO20l96ZziORP-w",
    # "hloc.yaml": "15JYLqU9Y56keMrg9ZfxwfbkbL6_haYpx",
    # "MapConnection_Graph.pkl": "199xZSc9jSajiCqzDW_AzhuqOp_YS41fZ",
}

@app.function(volumes={"/files": volume})   ## to create necesasry directories 
def create_directories():  
    os.makedirs(os.path.join("/files", "data", "New_York_City", "LightHouse"), exist_ok=True)
    logging.info('created path : /files/data/New_York_City/LightHouse')
    os.makedirs(os.path.join("/files", "configs"), exist_ok=True)
    logging.info('created path : /files/Configs')

@app.function(volumes={"/files": volume}, timeout=86400)
def checkAndDownload_file_from_remoteStorage():   ## download the data to the respective locations
    for filename, file_id in files.items():
        logging.info(f"Processing {filename}")
        if filename == "destination.json":
            download_path = os.path.join('/files', 'data', 'destination.json')
        elif filename == "hloc.yaml":
            download_path = os.path.join("/files", "configs", "hloc.yaml")
        elif filename == "MapConnection_Graph.pkl":
            download_path = os.path.join("/files", "data", "New_York_City", "MapConnection_Graph.pkl")
        else:
            download_path = os.path.join("/files", filename)

        if not os.path.exists(download_path):
            gdown.download(f'https://drive.google.com/uc?id={file_id}', download_path, quiet=False)
            logging.info(f"Downloaded {download_path}")
        else:
            logging.info(f"{download_path} already exists. Skipping download.")
        
    with open(os.path.join("/files", "configs", "hloc.yaml"), 'r') as file:
        config = yaml.safe_load(file)
    config['IO_root'] = "/root/UNav-IO"

    with open(os.path.join("/files", "configs", "hloc.yaml"), 'w') as file:
        yaml.safe_dump(config, file)

    #downloading from s3 to modal volumes 
    modal_directory = "/files/data"
     # List objects in the S3 bucket
    logging.info(f"Listing objects in the S3 bucket: {bucket_name}")
    response = s3_client.list_objects_v2(Bucket=bucket_name)

    if 'Contents' not in response:
        logging.info(f"No objects found in S3 bucket {bucket_name}.")
        return
    
    # Download each object from S3
    for obj in response['Contents']:
        s3_key = obj['Key']
        file_path = os.path.join(modal_directory, s3_key)

        # Check if the file already exists in the modal volume
        if not os.path.exists(file_path):
            # Create directories for nested keys if necessary
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Log the downloading process
            logging.info(f"Downloading {s3_key} to {file_path}")

            # Download the file from S3 to the modal volume
            s3_client.download_file(bucket_name, s3_key, file_path)
        else:
            logging.info(f"{file_path} already exists, skipping download.")
    
    logging.info(f"All files from S3 bucket {bucket_name} have been downloaded to {modal_directory}")


    logging.info("All files downloaded successfully from google drive and s3 bucket.")

@app.function(volumes={"/files": volume})
def rearrange_files_and_folders():
    items_to_rearrange = [
    # ("/files/data/6_floor", "/files/data/New_York_City/6_floor"),  # File
    # ("/files/data/global_features.h5", "/files/data/New_York_City/global_features.h5"),  # File
    # ("/files/data/NYISE_VC", "/files/data/New_York_City/NYISE_VC"),    # Folder
    # ("/files/data/save.pkl", "/files/data/New_York_City/save.pkl"),
    # ("/files/data/MapConnnection_Graph.pkl", "/files/data/New_York_City/MapConnnection_Graph.pkl"),
]
    for source_path, destination_path in items_to_rearrange:
        try:
            if os.path.isfile(source_path):
        
                logging.info(f"Moving file from {source_path} to {destination_path}")
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                shutil.move(source_path, destination_path)
                logging.info(f"Successfully moved file from {source_path} to {destination_path}")
                
            elif os.path.isdir(source_path):
                
                logging.info(f"Moving folder from {source_path} to {destination_path}")
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                shutil.move(source_path, destination_path)
                logging.info(f"Successfully moved folder from {source_path} to {destination_path}")
                
            else:
                logging.warning(f"Source path not found: {source_path}")
        except Exception as e:
            logging.error(f"Failed to move {source_path} to {destination_path}. Error: {e}")

    logging.info("All files and folders have been rearranged successfully.")


if __name__ == "__main__":
    with app.run(detach=True):
        create_directories.remote()
        checkAndDownload_file_from_remoteStorage.remote()
        rearrange_files_and_folders.remote()
