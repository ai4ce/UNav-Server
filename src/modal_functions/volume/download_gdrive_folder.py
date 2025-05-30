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
    ["gdown", "PyYAML", "boto3", "python-dotenv", "requests", "beautifulsoup4", "lxml"]
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
                # Try basic folder download first
                logging.info("Attempting basic folder download...")
                gdown.download_folder(
                    folder_url,
                    output=destination_path,
                    quiet=False,
                    use_cookies=False,
                    remaining_ok=True,  # Continue even if some files fail
                )

                # Count successful downloads
                file_count = sum(
                    1
                    for _, _, files in os.walk(destination_path)
                    for f in files
                    if not f.startswith(".download_progress")
                )

                if file_count > 0:
                    logging.info(f"Successfully downloaded {file_count} files")
                    progress_data = {
                        "status": "completed",
                        "timestamp": time.time(),
                        "folder_id": folder_id,
                        "destination": destination_path,
                        "files_downloaded": file_count,
                    }

                    with open(progress_file, "w") as f:
                        json.dump(progress_data, f)

                    logging.info("Basic folder download completed successfully")
                else:
                    raise Exception("No files were downloaded")

            except Exception as gdown_error:
                logging.warning(f"Basic folder download failed: {gdown_error}")
                logging.info("Trying individual file download approach...")

                # Fallback: Try to get individual files with permission handling
                success_count, fail_count = download_folder_files_individually(
                    folder_url, destination_path, progress_file
                )

                if success_count > 0:
                    logging.info(
                        f"Individual download completed: {success_count} files succeeded, {fail_count} files failed"
                    )

                    # Check progress file for detailed breakdown
                    if os.path.exists(progress_file):
                        try:
                            with open(progress_file, "r") as f:
                                progress_data = json.load(f)

                            perm_denied = progress_data.get(
                                "files_permission_denied", 0
                            )
                            other_fails = fail_count - perm_denied

                            message = f"Downloaded {success_count} files to {destination_path}."
                            if perm_denied > 0:
                                message += f" {perm_denied} files were skipped due to permission restrictions."
                            if other_fails > 0:
                                message += (
                                    f" {other_fails} files failed for other reasons."
                                )

                            return message

                        except Exception:
                            pass

                    return f"Downloaded {success_count} files to {destination_path}. {fail_count} files failed due to permissions or other issues."
                else:
                    # Try ZIP download as final fallback
                    logging.info("Trying ZIP download as final fallback...")
                    zip_url = (
                        f"https://drive.google.com/uc?id={folder_id}&export=download"
                    )
                    zip_path = os.path.join(destination_path, "temp_folder.zip")

                    try:
                        logging.info(f"Downloading folder as ZIP to {zip_path}")
                        gdown.download(zip_url, zip_path, quiet=False)

                        # Extract ZIP directly in the destination
                        logging.info(
                            f"Extracting ZIP file directly to {destination_path}"
                        )
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

                        # Final fallback: Try with cookies and different approaches
                        logging.info("Attempting alternative download methods...")

                    # Try with fuzzy matching for folder name
                    try:
                        logging.info("Trying gdown with fuzzy matching...")
                        gdown.download_folder(
                            folder_url,
                            output=destination_path,
                            quiet=False,
                            use_cookies=True,  # Enable cookies
                            remaining_ok=True,  # Allow partial downloads
                        )

                        # Check if anything was downloaded
                        file_count = sum(
                            1
                            for _, _, files in os.walk(destination_path)
                            for _ in files
                            if not _.startswith(".")
                        )

                        if file_count > 0:
                            logging.info(
                                f"Partial download successful! Downloaded {file_count} files"
                            )
                            progress_data = {
                                "status": "completed_partial",
                                "timestamp": time.time(),
                                "folder_id": folder_id,
                                "destination": destination_path,
                                "files_downloaded": file_count,
                            }

                            with open(progress_file, "w") as f:
                                json.dump(progress_data, f)
                        else:
                            raise Exception("No files downloaded")

                    except Exception as final_error:
                        logging.error(f"All download methods failed: {final_error}")

                        # Provide detailed error message and solution
                        error_message = f"""
Download failed for Google Drive folder: {folder_id}

POSSIBLE SOLUTIONS:
1. SHARING PERMISSIONS: Make sure the folder is shared as 'Anyone with the link can view'
   - Go to: https://drive.google.com/drive/folders/{folder_id}
   - Right-click → Share → Change to 'Anyone with the link'
   
2. FOLDER TOO LARGE: If folder is very large, try downloading in smaller batches
   
3. RATE LIMITING: Google may be rate-limiting downloads. Try again later.

4. MANUAL DOWNLOAD: You can manually download from:
   https://drive.google.com/drive/folders/{folder_id}

Error details: {str(final_error)}
"""

                        progress_data = {
                            "status": "failed",
                            "timestamp": time.time(),
                            "folder_id": folder_id,
                            "destination": destination_path,
                            "error": error_message,
                            "downloaded_files": downloaded_files,
                            "failed_files": failed_files,
                        }

                        with open(progress_file, "w") as f:
                            json.dump(progress_data, f)

                        raise Exception(error_message)

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
        # First, check if the folder is accessible
        logging.info("Checking folder permissions...")
        permission_check = check_gdrive_folder_permissions.remote(folder_url)

        if permission_check["status"] == "permission_denied":
            error_msg = f"""
PERMISSION ERROR: {permission_check['message']}

TO FIX THIS ISSUE:
1. Go to: {permission_check['folder_url']}
2. Right-click on the folder → Share
3. Click 'Change to anyone with the link'
4. Set permission to 'Viewer' or 'Editor'
5. Copy the new sharing link and try again

Current folder ID: {permission_check['folder_id']}
"""
            logging.error(error_msg)
            return {
                "status": "permission_error",
                "message": error_msg,
                "destination": destination_path,
            }

        elif permission_check["status"] == "error":
            logging.warning(f"Permission check failed: {permission_check['message']}")
            logging.info("Proceeding with download attempt anyway...")

        else:
            logging.info(f"Permission check: {permission_check['message']}")

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


@app.function()
def check_gdrive_folder_permissions(folder_url: str):
    """
    Check if a Google Drive folder is publicly accessible before attempting download

    Args:
        folder_url: Google Drive folder URL to check
    """
    import requests

    try:
        # Extract folder ID
        if "folders/" in folder_url:
            folder_id = folder_url.split("folders/")[1].split("?")[0]
        else:
            return {
                "status": "error",
                "message": "Invalid Google Drive folder URL format",
            }

        # Try to access the folder metadata
        test_url = f"https://drive.google.com/uc?id={folder_id}&export=download"

        response = requests.head(test_url, allow_redirects=True)

        if response.status_code == 200:
            return {
                "status": "accessible",
                "message": "Folder appears to be publicly accessible",
                "folder_id": folder_id,
            }
        elif response.status_code == 403:
            return {
                "status": "permission_denied",
                "message": f"Folder is not publicly accessible. Please change sharing settings to 'Anyone with the link can view'",
                "folder_id": folder_id,
                "folder_url": f"https://drive.google.com/drive/folders/{folder_id}",
            }
        else:
            return {
                "status": "unknown",
                "message": f"Received status code {response.status_code}. Folder may not exist or have access issues.",
                "folder_id": folder_id,
            }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error checking folder permissions: {str(e)}",
        }


def download_folder_files_individually(folder_url, destination_path, progress_file):
    """
    Download files from Google Drive folder one by one, skipping files with permission errors

    Returns:
        tuple: (success_count, fail_count)
    """
    import re
    import requests
    from bs4 import BeautifulSoup

    # Extract folder ID
    folder_id = folder_url.split("folders/")[1].split("?")[0]

    success_count = 0
    fail_count = 0
    downloaded_files = []
    failed_files = []
    permission_denied_files = []

    try:
        # Try to get the folder's file list from the HTML page
        logging.info("Attempting to parse folder contents from web page...")

        # Get the folder page HTML
        folder_page_url = f"https://drive.google.com/drive/folders/{folder_id}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(folder_page_url, headers=headers)

        if response.status_code != 200:
            logging.warning(
                f"Could not access folder page. Status: {response.status_code}"
            )
            return success_count, fail_count

        # Parse HTML to find file IDs and names
        soup = BeautifulSoup(response.text, "lxml")

        # Look for file data in the page (Google Drive embeds this in JavaScript)
        # This is a simplified approach - Google Drive's actual structure is complex
        script_tags = soup.find_all("script")
        file_patterns = []

        for script in script_tags:
            if script.string:
                # Look for patterns that might contain file IDs
                matches = re.findall(r'"([a-zA-Z0-9_-]{25,})"', script.string)
                for match in matches:
                    if len(match) > 20:  # Google Drive file IDs are typically long
                        file_patterns.append(match)

        # Remove duplicates and filter likely file IDs
        potential_file_ids = list(set(file_patterns))

        logging.info(f"Found {len(potential_file_ids)} potential file IDs to try")

        # Try to download each potential file
        for i, file_id in enumerate(potential_file_ids[:100]):  # Increased limit to 100
            try:
                logging.info(
                    f"Attempting to download file {i+1}/{min(100, len(potential_file_ids))}: {file_id}"
                )

                # First, check if file is accessible with a HEAD request
                file_url = f"https://drive.google.com/uc?id={file_id}"

                try:
                    head_response = requests.head(
                        file_url, allow_redirects=True, timeout=10
                    )

                    # Check for permission denied or other access issues
                    if head_response.status_code == 403:
                        logging.warning(f"✗ Permission denied for file: {file_id}")
                        permission_denied_files.append(file_id)
                        fail_count += 1
                        continue
                    elif head_response.status_code == 404:
                        logging.warning(f"✗ File not found: {file_id}")
                        failed_files.append(file_id)
                        fail_count += 1
                        continue
                    elif head_response.status_code != 200:
                        logging.warning(
                            f"✗ HTTP {head_response.status_code} for file: {file_id}"
                        )
                        failed_files.append(file_id)
                        fail_count += 1
                        continue

                except requests.RequestException as req_error:
                    logging.warning(
                        f"✗ Request error for file {file_id}: {str(req_error)}"
                    )
                    failed_files.append(file_id)
                    fail_count += 1
                    continue

                # Try to download the file
                output_path = os.path.join(destination_path, f"file_{file_id}")

                # Try download with gdown
                try:
                    # Use gdown with error handling
                    result = gdown.download(file_url, output_path, quiet=True)

                    # Check if file was actually downloaded and has content
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        success_count += 1
                        downloaded_files.append(file_id)
                        logging.info(
                            f"✓ Successfully downloaded file: {file_id} ({os.path.getsize(output_path)} bytes)"
                        )
                    else:
                        # Remove empty file
                        if os.path.exists(output_path):
                            os.remove(output_path)
                        fail_count += 1
                        failed_files.append(file_id)
                        logging.warning(
                            f"✗ Downloaded file is empty or invalid: {file_id}"
                        )

                except Exception as download_error:
                    error_str = str(download_error).lower()

                    # Check for specific permission errors
                    if (
                        "permission" in error_str
                        or "forbidden" in error_str
                        or "403" in error_str
                    ):
                        permission_denied_files.append(file_id)
                        logging.warning(
                            f"✗ Permission denied for file {file_id}: {str(download_error)}"
                        )
                    else:
                        failed_files.append(file_id)
                        logging.warning(
                            f"✗ Failed to download {file_id}: {str(download_error)}"
                        )

                    fail_count += 1

                    # Clean up any partially downloaded file
                    if os.path.exists(output_path):
                        os.remove(output_path)

                    continue

            except Exception as e:
                logging.warning(f"Error processing file ID {file_id}: {str(e)}")
                failed_files.append(file_id)
                fail_count += 1
                continue

        # Save progress with detailed breakdown
        progress_data = {
            "status": "completed_individual",
            "timestamp": time.time(),
            "folder_id": folder_id,
            "destination": destination_path,
            "files_downloaded": success_count,
            "files_failed": fail_count,
            "files_permission_denied": len(permission_denied_files),
            "downloaded_files": downloaded_files,
            "failed_files": failed_files,
            "permission_denied_files": permission_denied_files,
            "summary": f"Successfully downloaded {success_count} files. {len(permission_denied_files)} files had permission issues. {len(failed_files)} files failed for other reasons.",
        }

        with open(progress_file, "w") as f:
            json.dump(progress_data, f)

        logging.info(
            f"Individual download summary: {success_count} successful, {fail_count} failed ({len(permission_denied_files)} permission denied, {len(failed_files)} other errors)"
        )

        if permission_denied_files:
            logging.info(
                f"Files with permission issues: {permission_denied_files[:10]}..."
            )  # Show first 10

    except Exception as e:
        logging.error(f"Error in individual file download: {str(e)}")

    return success_count, fail_count


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
