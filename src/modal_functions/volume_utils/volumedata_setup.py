import os
import shutil 
import logging
import yaml
import zipfile
from dotenv import load_dotenv
import boto3
import gdown

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataManager:
    def __init__(self, s3_bucket_name, env_file=".env"):
        # Load environment variables
        load_dotenv(env_file)

        # AWS credentials setup
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )

        self.bucket_name = s3_bucket_name
        self.remotefilesIds = {
            "demo_query.png": "17MzPE9TyKiNsi6G59rqLCMMd40cIK3bU",
            "destination.json": "1sIzFujoumSsVlZqlwwO20l96ZziORP-w",
            "hloc.yaml": "15JYLqU9Y56keMrg9ZfxwfbkbL6_haYpx",
            "MapConnection_Graph.pkl": "199xZSc9jSajiCqzDW_AzhuqOp_YS41fZ",
        }

    def create_directories(self, base_path="/files"):
        """Create necessary directories."""
        os.makedirs(os.path.join(base_path, "data", "New_York_City", "LightHouse"), exist_ok=True)
        logging.info('Created path: /files/data/New_York_City/LightHouse')

        os.makedirs(os.path.join(base_path, "configs"), exist_ok=True)
        logging.info('Created path: /files/configs')

    def download_files_from_google_drive(self, base_path="/files"):
        """Download files from Google Drive."""
        for filename, file_id in self.remotefilesIds.items():
            logging.info(f"Processing {filename}")
            download_path = self.get_download_path(filename, base_path)

            if not os.path.exists(download_path):
                gdown.download(f'https://drive.google.com/uc?id={file_id}', download_path, quiet=False)
                logging.info(f"Downloaded {download_path}")
            else:
                logging.info(f"{download_path} already exists. Skipping download.")

    def get_download_path(self, filename, base_path):
        """Generate the download path based on the file type."""
        if filename == "destination.json":
            return os.path.join(base_path, 'data', 'destination.json')
        elif filename == "hloc.yaml":
            return os.path.join(base_path, "configs", "hloc.yaml")
        elif filename == "MapConnection_Graph.pkl":
            return os.path.join(base_path, "data", "New_York_City", "MapConnection_Graph.pkl")
        else:
            return os.path.join(base_path, filename)

    def modify_hloc_yaml(self, base_path="/files"):
        """Modify hloc.yaml configuration."""
        with open(os.path.join(base_path, "configs", "hloc.yaml"), 'r') as file:
            config = yaml.safe_load(file)
        config['IO_root'] = "/root/UNav-IO"

        with open(os.path.join(base_path, "configs", "hloc.yaml"), 'w') as file:
            yaml.safe_dump(config, file)

    def download_files_from_s3(self, modal_directory="/files/data"):
        """Download all files from the specified S3 bucket."""
        logging.info(f"Listing objects in the S3 bucket: {self.bucket_name}")
        response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)

        if 'Contents' not in response:
            logging.info(f"No objects found in S3 bucket {self.bucket_name}.")
            return
        
        # Download each object from S3
        for obj in response['Contents']:
            s3_key = obj['Key']
            file_path = os.path.join(modal_directory, s3_key)

            # Create directories for nested keys if necessary
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            logging.info(f"Downloading {s3_key} to {file_path}")
            
            # Download the file
            self.s3_client.download_file(self.bucket_name, s3_key, file_path)

        logging.info(f"All files from S3 bucket {self.bucket_name} have been downloaded to {modal_directory}")

    def run(self, base_path="/files"):
        """Main function to create directories, download files from Google Drive and S3."""
        self.create_directories(base_path)
        self.download_files_from_google_drive(base_path)
        self.modify_hloc_yaml(base_path)
        self.download_files_from_s3(os.path.join(base_path, "data"))
        logging.info("All files downloaded successfully from Google Drive and S3 bucket.")
