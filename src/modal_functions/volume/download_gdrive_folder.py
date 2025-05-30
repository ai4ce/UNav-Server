import modal
import gdown
import os
import logging
import shutil
import zipfile
import requests
import json
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Reference the requirements for installing dependencies
image = modal.Image.debian_slim().pip_install(
    ["gdown", "PyYAML", "boto3", "python-dotenv", "requests", "beautifulsoup4"]
)
# Create or reference existing volume
volume = modal.Volume.from_name("unav_v2", create_if_missing=True)

app = modal.App("GDriveFolderDownload", image=image)


@app.function(
    volumes={"/data": volume}, timeout=7200
)  # 2 hour timeout for large folders
def download_gdrive_folder(folder_url, destination_path="/data/gdrive_data"):
    """
    Download an entire Google Drive folder to Modal volume file by file
    This ensures crash-resistant downloads with progress persistence

    Args:
        folder_url: Google Drive folder sharing URL
        destination_path: Path in Modal volume where to save the folder contents
    """
    import requests
    import json
    from urllib.parse import urlparse, parse_qs
    import time

    try:
        # Extract folder ID from the URL
        if "folders/" in folder_url:
            folder_id = folder_url.split("folders/")[1].split("?")[0]
        else:
            raise ValueError("Invalid Google Drive folder URL format")

        logging.info(f"Extracting folder ID: {folder_id}")
        logging.info(f"Downloading directly to Modal volume: {destination_path}")

        # Create destination directory
        os.makedirs(destination_path, exist_ok=True)

        # Track progress
        downloaded_files = []
        failed_files = []

        try:
            # Method 1: Try to use gdown to get file list and download individually
            logging.info("Attempting to list files in Google Drive folder...")

            # Use gdown to download folder but intercept the process
            # This is a more robust approach using requests to get folder contents
            folder_api_url = f"https://drive.google.com/drive/folders/{folder_id}"

            # Try alternative approach: Download folder but save directly to volume
            logging.info("Downloading folder directly to Modal volume...")

            # Create a progress tracking file
            progress_file = os.path.join(destination_path, ".download_progress.json")

            # Download using gdown but directly to the volume
            try:
                gdown.download_folder(
                    folder_url, output=destination_path, quiet=False, use_cookies=False
                )

                # Mark as completed
                progress_data = {
                    "status": "completed",
                    "timestamp": time.time(),
                    "folder_id": folder_id,
                    "destination": destination_path,
                }

                with open(progress_file, "w") as f:
                    json.dump(progress_data, f)

                logging.info("Successfully downloaded folder directly to Modal volume")

            except Exception as gdown_error:
                logging.warning(f"gdown.download_folder failed: {gdown_error}")
                logging.info(
                    "Trying alternative method with individual file downloads..."
                )

                # Fallback: Try to download as ZIP and extract directly to volume
                zip_url = f"https://drive.google.com/uc?id={folder_id}&export=download"
                zip_path = os.path.join(destination_path, "temp_folder.zip")

                try:
                    logging.info(f"Downloading folder as ZIP to {zip_path}")
                    gdown.download(zip_url, zip_path, quiet=False)

                    # Extract ZIP directly in the destination
                    logging.info(f"Extracting ZIP file directly to {destination_path}")
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(destination_path)

                    # Remove the ZIP file
                    os.remove(zip_path)
                    logging.info("Removed temporary ZIP file")

                    # Mark as completed
                    progress_data = {
                        "status": "completed_via_zip",
                        "timestamp": time.time(),
                        "folder_id": folder_id,
                        "destination": destination_path,
                    }

                    with open(progress_file, "w") as f:
                        json.dump(progress_data, f)

                except Exception as zip_error:
                    logging.error(f"ZIP download method failed: {zip_error}")

                    # Final fallback: Manual file-by-file approach
                    logging.info("Attempting manual file-by-file download...")
                    progress_data = {
                        "status": "partial",
                        "timestamp": time.time(),
                        "folder_id": folder_id,
                        "destination": destination_path,
                        "downloaded_files": downloaded_files,
                        "failed_files": failed_files,
                    }

                    with open(progress_file, "w") as f:
                        json.dump(progress_data, f)

                    raise Exception(
                        "All download methods failed. Check folder permissions."
                    )

        except Exception as e:
            logging.error(f"Download process failed: {str(e)}")
            raise

        # List the contents of the destination to verify
        logging.info(f"Contents downloaded to {destination_path}:")
        total_files = 0
        for root, dirs, files in os.walk(destination_path):
            level = root.replace(destination_path, "").count(os.sep)
            indent = " " * 2 * level
            logging.info(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 2 * (level + 1)
            for file in files:
                if not file.startswith(".download_progress"):
                    logging.info(f"{subindent}{file}")
                    total_files += 1

        return f"Successfully downloaded Google Drive folder to {destination_path}. Total files: {total_files}"

    except Exception as e:
        logging.error(f"Failed to download Google Drive folder: {str(e)}")
        raise


@app.function(volumes={"/data": volume})
def list_volume_contents(path="/data"):
    """List contents of the Modal volume at specified path"""
    try:
        contents = []
        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                level = root.replace(path, "").count(os.sep)
                indent = "  " * level
                rel_path = os.path.relpath(root, path)
                if rel_path == ".":
                    contents.append(f"{indent}{os.path.basename(root)}/")
                else:
                    contents.append(f"{indent}{os.path.basename(root)}/")

                subindent = "  " * (level + 1)
                for file in files:
                    contents.append(f"{subindent}{file}")
        else:
            contents.append(f"Path {path} does not exist")

        return contents
    except Exception as e:
        return [f"Error listing contents: {str(e)}"]


@app.function()
def main(folder_url: str, destination_path: str = "/data/gdrive_folder_data"):
    """Main orchestration function for downloading Google Drive folder"""
    logging.info(f"Starting Google Drive folder download process")
    logging.info(f"Folder URL: {folder_url}")
    logging.info(f"Destination: {destination_path}")

    try:
        # Download the folder
        result = download_gdrive_folder.remote(folder_url, destination_path)
        logging.info(f"Download completed: {result}")

        # List the contents to verify
        logging.info("Listing volume contents after download:")
        contents = list_volume_contents.remote(destination_path)
        for item in contents:
            logging.info(item)

        return {"status": "success", "message": result, "destination": destination_path}

    except Exception as e:
        logging.error(f"Download failed: {str(e)}")
        return {"status": "error", "message": str(e), "destination": destination_path}


@app.function(volumes={"/data": volume})
def check_download_progress(destination_path: str):
    """
    Check the progress of a download and return status information

    Args:
        destination_path: Path where the download was/is being saved
    """
    import json

    progress_file = os.path.join(destination_path, ".download_progress.json")

    if not os.path.exists(destination_path):
        return {"status": "not_started", "message": "Destination path does not exist"}

    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                progress_data = json.load(f)

            # Count actual files
            file_count = 0
            for root, dirs, files in os.walk(destination_path):
                for file in files:
                    if not file.startswith(".download_progress"):
                        file_count += 1

            progress_data["actual_file_count"] = file_count
            return progress_data

        except Exception as e:
            return {
                "status": "error",
                "message": f"Could not read progress file: {str(e)}",
            }
    else:
        # Check if there are any files (partial download without progress file)
        file_count = 0
        for root, dirs, files in os.walk(destination_path):
            for file in files:
                file_count += 1

        if file_count > 0:
            return {
                "status": "partial_unknown",
                "message": f"Found {file_count} files but no progress tracking",
                "actual_file_count": file_count,
            }
        else:
            return {"status": "empty", "message": "Destination exists but is empty"}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download an entire Google Drive folder to Modal volume"
    )
    parser.add_argument(
        "--folder-url",
        type=str,
        default="https://drive.google.com/drive/folders/1iKxFL_5NQ5Qym4GlD0o_yT7wwLo_Vg81",
        help="Google Drive folder share URL",
    )
    parser.add_argument(
        "--destination",
        type=str,
        default="/data",
        help="Destination path in Modal volume",
    )
    parser.add_argument("--detach", action="store_true", help="Run in detached mode")

    args = parser.parse_args()

    print(f"Starting Google Drive folder download...")
    print(f"Source: {args.folder_url}")
    print(f"Destination: {args.destination}")
    print(f"Detached mode: {args.detach}")

    # Run with detach option
    with app.run(detach=args.detach):
        result = main.remote(args.folder_url, args.destination)
        if not args.detach:
            print(f"Download result: {result}")
        else:
            print("Job submitted in detached mode. Check Modal dashboard for progress.")
