import modal
import gdown
import os
import logging
import shutil
import zipfile
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Reference the requirements for installing dependencies
image = modal.Image.debian_slim().pip_install([
    "gdown",
    "PyYAML", 
    "boto3",
    "python-dotenv"
])
# Create or reference existing volume
volume = modal.Volume.from_name("unav_v2", create_if_missing=True)

app = modal.App("GDriveFolderDownload", image=image)


@app.function(
    volumes={"/data": volume}, timeout=7200
)  # 2 hour timeout for large folders
def download_gdrive_folder(folder_url, destination_path="/data/gdrive_data"):
    """
    Download an entire Google Drive folder to Modal volume

    Args:
        folder_url: Google Drive folder sharing URL
        destination_path: Path in Modal volume where to save the folder contents
    """
    try:
        # Extract folder ID from the URL
        # URL format: https://drive.google.com/drive/folders/FOLDER_ID
        if "folders/" in folder_url:
            folder_id = folder_url.split("folders/")[1].split("?")[0]
        else:
            raise ValueError("Invalid Google Drive folder URL format")

        logging.info(f"Extracting folder ID: {folder_id}")

        # Create destination directory
        os.makedirs(destination_path, exist_ok=True)

        # Create a temporary directory for download
        temp_dir = "/tmp/gdrive_download"
        os.makedirs(temp_dir, exist_ok=True)

        logging.info(f"Starting download of Google Drive folder to {temp_dir}")

        # Download the entire folder using gdown
        # This will download as a zip file and extract it
        try:
            # Download folder as zip
            zip_path = os.path.join(temp_dir, "folder_download.zip")
            gdown.download_folder(
                f"https://drive.google.com/drive/folders/{folder_id}",
                output=temp_dir,
                quiet=False,
                use_cookies=False,
            )

            # Find the downloaded content and move to destination
            for item in os.listdir(temp_dir):
                item_path = os.path.join(temp_dir, item)
                dest_item_path = os.path.join(destination_path, item)

                if os.path.isdir(item_path):
                    # Move directory
                    if os.path.exists(dest_item_path):
                        shutil.rmtree(dest_item_path)
                    shutil.move(item_path, dest_item_path)
                    logging.info(f"Moved directory: {item} to {dest_item_path}")
                else:
                    # Move file
                    if os.path.exists(dest_item_path):
                        os.remove(dest_item_path)
                    shutil.move(item_path, dest_item_path)
                    logging.info(f"Moved file: {item} to {dest_item_path}")

        except Exception as e:
            logging.error(f"gdown.download_folder failed: {e}")
            logging.info("Trying alternative method with zip download...")

            # Alternative method: download as zip file directly
            zip_url = f"https://drive.google.com/uc?id={folder_id}&export=download"
            zip_path = os.path.join(temp_dir, "folder.zip")

            try:
                gdown.download(zip_url, zip_path, quiet=False)

                # Extract zip file
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(destination_path)
                    logging.info(f"Extracted zip file to {destination_path}")

            except Exception as zip_error:
                logging.error(f"Zip download method also failed: {zip_error}")
                raise

        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        # List the contents of the destination to verify
        logging.info(f"Contents downloaded to {destination_path}:")
        for root, dirs, files in os.walk(destination_path):
            level = root.replace(destination_path, "").count(os.sep)
            indent = " " * 2 * level
            logging.info(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 2 * (level + 1)
            for file in files:
                logging.info(f"{subindent}{file}")

        return f"Successfully downloaded Google Drive folder to {destination_path}"

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
        default="/data/cloned_folder",
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
