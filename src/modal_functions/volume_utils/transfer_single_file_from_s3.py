import modal
import os
import logging
from dotenv import load_dotenv
import boto3
import argparse

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Reference the requirements for installing dependencies
image = modal.Image.debian_slim().pip_install(["boto3", "python-dotenv"])

# Create or reference existing volume
volume = modal.Volume.from_name("NewVisiondata")

app = modal.App("S3Transfer", image=image)


@app.function(volumes={"/files": volume})
def create_destination_directory(destination_path):
    """Create the destination directory structure if it doesn't exist"""
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    logging.info(f"Created directory structure for: {destination_path}")


@app.function(volumes={"/files": volume}, timeout=3600)
def transfer_file_from_s3(bucket_name, s3_key, destination_path=None):
    """
    Transfer a specific file from S3 bucket to Modal volume

    Args:
        bucket_name: Name of the S3 bucket
        s3_key: Path to the file in S3 bucket
        destination_path: Path where to save the file in Modal volume (default: /files/data/{s3_key})
    """
    load_dotenv()

    # Initialize S3 client
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.getenv(
            "AWS_SECRET_ACCESS_KEY", ""
        ),
    )

    # Set destination path if not specified
    if destination_path is None:
        destination_path = os.path.join("/files/data", s3_key)

    # Create directory structure if needed
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    # Check if file already exists
    if os.path.exists(destination_path):
        logging.info(f"File already exists at {destination_path}, skipping download.")
        return destination_path

    try:
        # Log the download process
        logging.info(
            f"Downloading {s3_key} from bucket {bucket_name} to {destination_path}"
        )

        # Download the file from S3 to Modal volume
        s3_client.download_file(bucket_name, s3_key, destination_path)

        logging.info(f"Successfully downloaded {s3_key} to {destination_path}")
        return destination_path

    except Exception as e:
        logging.error(f"Failed to download {s3_key}: {str(e)}")
        raise


@app.function()
def main(bucket_name, s3_key, destination_path=None):
    """Main function to orchestrate the transfer process"""
    if destination_path:
        create_destination_directory.remote(destination_path)
    result_path = transfer_file_from_s3.remote(bucket_name, s3_key, destination_path)
    return result_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transfer a file from S3 to Modal volume"
    )
    parser.add_argument("--bucket", type=str, required=True, help="S3 bucket name")
    parser.add_argument(
        "--file-path", type=str, required=True, help="File path in S3 bucket"
    )
    parser.add_argument(
        "--destination", type=str, help="Destination path in Modal volume"
    )

    args = parser.parse_args()

    with app.run(detach=True):
        result = main.remote(args.bucket, args.file_path, args.destination)
        print(f"Transferred file to: {result}")
