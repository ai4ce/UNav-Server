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

app = modal.App("S3toModalTransfer", image=image)


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
        aws_access_key_id="",
        aws_secret_access_key="",
    )

    # Set destination path if not specified
    if destination_path is None:
        destination_path = os.path.join("/files/data", s3_key)

    # Ensure destination path starts with /files
    if not destination_path.startswith("/files"):
        destination_path = f"/files/{destination_path}"

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


@app.function(volumes={"/files": volume}, timeout=3600)
def list_s3_objects(bucket_name, prefix):
    """List all objects in S3 bucket with the given prefix"""
    load_dotenv()

    s3_client = boto3.client(
        "s3",
        aws_access_key_id="",
        aws_secret_access_key='',
    )

    try:
        # Handle both with and without trailing slash
        if not prefix.endswith("/"):
            prefix = prefix + "/"

        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

        if "Contents" not in response:
            logging.warning(
                f"No objects found with prefix '{prefix}' in bucket '{bucket_name}'"
            )
            return []

        # Return list of keys
        objects = [obj["Key"] for obj in response["Contents"]]

        # Handle pagination if there are more than 1000 objects
        while response["IsTruncated"]:
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix,
                ContinuationToken=response["NextContinuationToken"],
            )
            objects.extend([obj["Key"] for obj in response["Contents"]])

        logging.info(f"Found {len(objects)} objects with prefix '{prefix}'")
        return objects

    except Exception as e:
        logging.error(f"Failed to list objects with prefix '{prefix}': {str(e)}")
        raise


@app.function(volumes={"/files": volume})
def transfer_folder_from_s3(bucket_name, folder_prefix, destination_folder=None):
    """Transfer all files in a folder from S3 to Modal volume"""

    # Set destination folder if not specified
    if destination_folder is None:
        destination_folder = os.path.join("/files/data", folder_prefix)

    # Ensure destination folder starts with /files and ends with /
    if not destination_folder.startswith("/files"):
        destination_folder = f"/files/{destination_folder}"
    if not destination_folder.endswith("/"):
        destination_folder = f"{destination_folder}/"

    # List all objects with the given prefix
    objects = list_s3_objects.remote(bucket_name, folder_prefix)

    if not objects:
        return None

    # Download each object
    transferred_files = []
    for obj_key in objects:
        # Skip directory markers (objects that end with /)
        if obj_key.endswith("/"):
            continue

        # Calculate relative path from prefix
        if folder_prefix.endswith("/"):
            relative_path = obj_key[len(folder_prefix) :]
        else:
            relative_path = obj_key[len(folder_prefix) + 1 :]

        # Set destination path
        dest_path = os.path.join(destination_folder, relative_path)

        # Transfer file
        result = transfer_file_from_s3.remote(bucket_name, obj_key, dest_path)
        transferred_files.append(result)

    return transferred_files


@app.function()
def main(bucket_name, s3_key, destination_path=None):
    """Main function to orchestrate the transfer process"""

    # Check if this is a folder transfer (ends with / or doesn't exist as a file)
    is_folder = s3_key.endswith("/")

    if not is_folder:
        # Try to check if it's a file
        try:
            load_dotenv()
            s3_client = boto3.client(
                "s3",
                aws_access_key_id="",
                aws_secret_access_key="",
            )
            s3_client.head_object(Bucket=bucket_name, Key=s3_key)
            # If we reach here, it's a file
        except:
            # If not found, it might be a folder
            is_folder = True

    if is_folder:
        logging.info(f"Processing '{s3_key}' as a folder")
        results = transfer_folder_from_s3.remote(bucket_name, s3_key, destination_path)
        return results
    else:
        logging.info(f"Processing '{s3_key}' as a file")
        if destination_path:
            create_destination_directory.remote(destination_path)
        result_path = transfer_file_from_s3.remote(
            bucket_name, s3_key, destination_path
        )
        return result_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transfer a file or folder from S3 to Modal volume"
    )
    parser.add_argument("--bucket", type=str, required=True, help="S3 bucket name")
    parser.add_argument(
        "--file-path", type=str, required=True, help="File or folder path in S3 bucket"
    )
    parser.add_argument(
        "--destination", type=str, help="Destination path in Modal volume"
    )

    args = parser.parse_args()

    with app.run(detach=True):
        result = main.remote(args.bucket, args.file_path, args.destination)
        if isinstance(result, list):
            print(f"Transferred {len(result)} files to Modal volume")
        else:
            print(f"Transferred file to: {result}")
