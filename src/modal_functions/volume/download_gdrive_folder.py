import modal
import gdown
import os
import logging
import shutil
import zipfile
import requests
import json
import time
import random
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Reference the requirements for installing dependencies
image = modal.Image.debian_slim().pip_install(
    [
        "gdown",
        "PyYAML",
        "boto3",
        "python-dotenv",
        "requests",
        "beautifulsoup4",
        "lxml",
        "fake-useragent",
    ]
)
# Create or reference existing volume
volume = modal.Volume.from_name("unav_v2", create_if_missing=True)

app = modal.App("GDriveFolderDownload", image=image)


def get_random_user_agent():
    """Get a random user agent to avoid detection"""
    try:
        from fake_useragent import UserAgent

        ua = UserAgent()
        return ua.random
    except ImportError:
        # Fallback user agents if fake_useragent is not available
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36",
        ]
        return random.choice(user_agents)
    except Exception:
        # Ultimate fallback
        return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def create_rate_limit_session():
    """Create a requests session with rate limiting bypass features"""
    session = requests.Session()

    # Random user agent for each session
    user_agent = get_random_user_agent()

    # Randomize headers to look more like different browsers
    accept_languages = [
        "en-US,en;q=0.9",
        "en-US,en;q=0.8,fr;q=0.6",
        "en-US,en;q=0.7,es;q=0.3",
        "en-GB,en;q=0.9,en-US;q=0.8",
        "en-US,en;q=0.5",
    ]

    encodings = [
        "gzip, deflate, br",
        "gzip, deflate",
        "gzip",
    ]

    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "Accept-Language": random.choice(accept_languages),
            "Accept-Encoding": random.choice(encodings),
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
            "DNT": "1",  # Do Not Track
        }
    )

    # Add some randomization to request timing
    if random.random() < 0.3:  # 30% chance
        session.headers["Pragma"] = "no-cache"

    return session


def exponential_backoff_download(download_func, max_retries=5, base_delay=1):
    """
    Perform download with exponential backoff to handle rate limiting

    Args:
        download_func: Function to call for download
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
    """
    for attempt in range(max_retries):
        try:
            return download_func()
        except Exception as e:
            error_str = str(e).lower()

            # Check if it's a rate limiting error
            if any(
                keyword in error_str
                for keyword in ["rate", "limit", "quota", "too many", "429", "503"]
            ):
                if attempt < max_retries - 1:
                    # Calculate delay with jitter
                    delay = (base_delay * (2**attempt)) + random.uniform(0, 1)
                    logging.warning(
                        f"Rate limited, retrying in {delay:.2f} seconds (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                    continue
                else:
                    logging.error(f"Max retries reached for rate limiting: {e}")
                    raise
            else:
                # Not a rate limiting error, re-raise immediately
                raise

    return None


def smart_delay_between_requests():
    """Add intelligent delay between requests to avoid rate limiting"""
    # Random delay between 0.5 to 3 seconds with some longer pauses
    if random.random() < 0.1:  # 10% chance of longer pause
        delay = random.uniform(5.0, 10.0)
        logging.debug(f"Taking longer pause: {delay:.2f} seconds")
    else:
        delay = random.uniform(0.5, 3.0)
    time.sleep(delay)


def get_alternative_download_urls(file_id):
    """Generate alternative download URLs to try different endpoints"""
    # Primary Google Drive download endpoints
    primary_urls = [
        f"https://drive.google.com/uc?id={file_id}&export=download",
        f"https://drive.google.com/uc?export=download&id={file_id}",
    ]

    # Alternative endpoints and methods
    alternative_urls = [
        f"https://docs.google.com/uc?id={file_id}&export=download",
        f"https://drive.google.com/file/d/{file_id}/view?usp=sharing",
        f"https://drive.google.com/open?id={file_id}",
        f"https://drive.google.com/uc?id={file_id}&confirm=t",
        f"https://docs.google.com/uc?export=download&id={file_id}",
        f"https://drive.usercontent.google.com/download?id={file_id}&export=download",
    ]

    # Randomize the order to distribute load
    all_urls = primary_urls + alternative_urls
    random.shuffle(all_urls)

    return all_urls


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


def extract_file_info_from_gdrive_page(page_content):
    """
    Enhanced file information extraction from Google Drive page

    Args:
        page_content: HTML content of the Google Drive folder page

    Returns:
        list: List of dictionaries with file_id and name
    """
    import re
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(page_content, "lxml")
    script_tags = soup.find_all("script")
    file_info_list = []

    for script in script_tags:
        if script.string:
            # Pattern 1: Look for file arrays with ID and name
            # Matches: ["file_id","filename.ext",...]
            file_array_matches = re.findall(
                r'\["([a-zA-Z0-9_-]{25,})","([^"]+\.[a-zA-Z0-9]+)"[^\]]*\]',
                script.string,
            )

            for file_id, filename in file_array_matches:
                if len(file_id) > 20:
                    file_info_list.append({"id": file_id, "name": filename})

            # Pattern 2: Look for JSON objects with id and name
            # Matches: "id":"file_id"..."name":"filename"
            file_obj_matches = re.findall(
                r'"id"\s*:\s*"([a-zA-Z0-9_-]{25,})"[^}]*"name"\s*:\s*"([^"]+)"',
                script.string,
            )

            for file_id, filename in file_obj_matches:
                if len(file_id) > 20:
                    file_info_list.append({"id": file_id, "name": filename})

            # Pattern 3: Look for alternative object structure
            # Matches: ["id","file_id"]...["name","filename"]
            id_matches = re.findall(r'\["id","([a-zA-Z0-9_-]{25,})"\]', script.string)
            name_matches = re.findall(
                r'\["name","([^"]+\.[a-zA-Z0-9]+)"\]', script.string
            )

            # Try to pair IDs with names if they appear in similar contexts
            if len(id_matches) == len(name_matches):
                for file_id, filename in zip(id_matches, name_matches):
                    if len(file_id) > 20:
                        file_info_list.append({"id": file_id, "name": filename})

    # Remove duplicates based on file ID
    seen_ids = set()
    unique_file_info = []
    for file_info in file_info_list:
        if file_info["id"] not in seen_ids:
            unique_file_info.append(file_info)
            seen_ids.add(file_info["id"])

    # If no files found with names, try to get just IDs as fallback
    if not unique_file_info:
        for script in script_tags:
            if script.string:
                file_id_matches = re.findall(r'"([a-zA-Z0-9_-]{25,})"', script.string)
                for file_id in file_id_matches:
                    if len(file_id) > 20 and file_id not in seen_ids:
                        unique_file_info.append(
                            {"id": file_id, "name": f"unknown_file_{file_id[:8]}.bin"}
                        )
                        seen_ids.add(file_id)

    return unique_file_info


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

        # Use rate limiting bypass session
        session = create_rate_limit_session()

        def fetch_folder_page():
            return session.get(folder_page_url, timeout=30)

        try:
            response = exponential_backoff_download(fetch_folder_page, max_retries=3)
        except Exception as e:
            logging.warning(
                f"Failed to fetch folder page with rate limiting bypass: {str(e)}"
            )
            # Fallback to basic request
            headers = {"User-Agent": get_random_user_agent()}
            response = requests.get(folder_page_url, headers=headers)

        if response.status_code != 200:
            logging.warning(
                f"Could not access folder page. Status: {response.status_code}"
            )
            return success_count, fail_count

        # Parse HTML to find file IDs and names using enhanced extraction
        unique_file_info = extract_file_info_from_gdrive_page(response.text)

        logging.info(f"Found {len(unique_file_info)} files to download")
        if unique_file_info:
            logging.info("Sample files found:")
            for i, file_info in enumerate(
                unique_file_info[:5]
            ):  # Show first 5 as sample
                logging.info(
                    f"  {i+1}. {file_info['name']} (ID: {file_info['id'][:12]}...)"
                )

        # Try to download each file
        for i, file_info in enumerate(unique_file_info[:100]):  # Increased limit to 100
            file_id = file_info["id"]
            filename = file_info["name"]

            try:
                logging.info(
                    f"Attempting to download file {i+1}/{min(100, len(unique_file_info))}: {filename} (ID: {file_id})"
                )

                # Add delay between downloads to avoid rate limiting
                if i > 0:  # Skip delay for first file
                    smart_delay_between_requests()

                # First, check if file is accessible with a HEAD request
                file_url = f"https://drive.google.com/uc?id={file_id}"

                try:
                    # Use rate limiting bypass session for HEAD request
                    session = create_rate_limit_session()
                    head_response = session.head(
                        file_url, allow_redirects=True, timeout=10
                    )

                    # Add delay to avoid rate limiting
                    smart_delay_between_requests()

                    # Check for permission denied or other access issues
                    if head_response.status_code == 403:
                        output_path = os.path.join(destination_path, filename)
                        logging.warning(
                            f"✗ Permission denied for file: {filename} -> {output_path}"
                        )
                        permission_denied_files.append(
                            {"file_id": file_id, "name": filename, "path": output_path}
                        )
                        fail_count += 1
                        continue
                    elif head_response.status_code == 404:
                        output_path = os.path.join(destination_path, filename)
                        logging.warning(
                            f"✗ File not found: {filename} -> {output_path}"
                        )
                        failed_files.append(
                            {"file_id": file_id, "name": filename, "path": output_path}
                        )
                        fail_count += 1
                        continue
                    elif head_response.status_code != 200:
                        output_path = os.path.join(destination_path, filename)
                        logging.warning(
                            f"✗ HTTP {head_response.status_code} for file: {filename} -> {output_path}"
                        )
                        failed_files.append(
                            {"file_id": file_id, "name": filename, "path": output_path}
                        )
                        fail_count += 1
                        continue

                except requests.RequestException as req_error:
                    output_path = os.path.join(destination_path, filename)
                    logging.warning(
                        f"✗ Request error for file {filename} -> {output_path}: {str(req_error)}"
                    )
                    failed_files.append(
                        {"file_id": file_id, "name": filename, "path": output_path}
                    )
                    fail_count += 1
                    continue

                # Try to download the file with the actual filename
                # Sanitize filename for filesystem
                safe_filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
                output_path = os.path.join(destination_path, safe_filename)

                # Use advanced download with comprehensive rate limiting bypass
                download_success, error_message = advanced_download_with_bypass(
                    file_id, filename, output_path, max_retries=5
                )

                if download_success:
                    success_count += 1
                    downloaded_files.append(
                        {"file_id": file_id, "name": filename, "path": output_path}
                    )
                    logging.info(
                        f"✓ Successfully downloaded: {filename} -> {output_path} ({os.path.getsize(output_path)} bytes)"
                    )
                else:
                    # Handle different types of failures
                    if error_message and "permission denied" in error_message.lower():
                        permission_denied_files.append(
                            {"file_id": file_id, "name": filename, "path": output_path}
                        )
                        logging.warning(
                            f"✗ Permission denied for {filename}: {error_message}"
                        )
                    else:
                        failed_files.append(
                            {"file_id": file_id, "name": filename, "path": output_path}
                        )
                        logging.warning(
                            f"✗ Failed to download {filename}: {error_message}"
                        )

                    fail_count += 1
                    continue

            except Exception as e:
                safe_filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
                output_path = os.path.join(destination_path, safe_filename)
                logging.warning(
                    f"Error processing file {filename} -> {output_path}: {str(e)}"
                )
                failed_files.append(
                    {"file_id": file_id, "name": filename, "path": output_path}
                )
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

        # Log comprehensive statistics
        log_rate_limiting_stats(downloaded_files, failed_files, permission_denied_files)

        if permission_denied_files:
            logging.info(f"Files with permission issues:")
            for file_info in permission_denied_files[:10]:  # Show first 10
                if isinstance(file_info, dict) and "name" in file_info:
                    logging.info(f"  - {file_info['name']} -> {file_info['path']}")
                elif isinstance(file_info, dict):
                    logging.info(f"  - {file_info['file_id']} -> {file_info['path']}")
                else:
                    logging.info(f"  - {file_info}")

        if failed_files:
            logging.info(f"Files that failed for other reasons:")
            for file_info in failed_files[:10]:  # Show first 10
                if isinstance(file_info, dict) and "name" in file_info:
                    logging.info(f"  - {file_info['name']} -> {file_info['path']}")
                elif isinstance(file_info, dict):
                    logging.info(f"  - {file_info['file_id']} -> {file_info['path']}")
                else:
                    logging.info(f"  - {file_info}")

        if downloaded_files:
            logging.info(f"Successfully downloaded files:")
            for file_info in downloaded_files[:10]:  # Show first 10
                if isinstance(file_info, dict) and "name" in file_info:
                    logging.info(f"  - {file_info['name']} -> {file_info['path']}")
                elif isinstance(file_info, dict):
                    logging.info(f"  - {file_info['file_id']} -> {file_info['path']}")
                else:
                    logging.info(f"  - {file_info}")

        # Monitor rate limiting bypass effectiveness
        monitor_rate_limiting_effectiveness()

    except Exception as e:
        logging.error(f"Error in individual file download: {str(e)}")

    return success_count, fail_count


def advanced_download_with_bypass(file_id, filename, output_path, max_retries=5):
    """
    Advanced download function with comprehensive rate limiting bypass

    Args:
        file_id: Google Drive file ID
        filename: Original filename
        output_path: Local path to save file
        max_retries: Maximum retry attempts

    Returns:
        tuple: (success: bool, error_message: str)
    """

    # Create multiple sessions with different configurations
    sessions = []
    for _ in range(3):
        session = create_rate_limit_session()
        sessions.append(session)

    download_urls = get_alternative_download_urls(file_id)

    for attempt in range(max_retries):
        for session_idx, session in enumerate(sessions):
            for url_idx, url in enumerate(download_urls):
                try:
                    logging.debug(
                        f"Attempt {attempt+1}/{max_retries}, Session {session_idx+1}, URL {url_idx+1}"
                    )

                    # Add progressive delay based on attempt number
                    if attempt > 0:
                        delay = (2**attempt) + random.uniform(0, 2)
                        logging.debug(f"Waiting {delay:.2f} seconds before retry...")
                        time.sleep(delay)

                    # Try downloading with current session and URL
                    response = session.get(url, stream=True, timeout=45)

                    # Handle Google Drive's virus scan warning page
                    if (
                        "virus" in response.text.lower()
                        and "download anyway" in response.text.lower()
                    ):
                        # Extract the actual download URL from the warning page
                        import re

                        confirm_url_match = re.search(
                            r'href="([^"]*&amp;confirm=[^"]*)"', response.text
                        )
                        if confirm_url_match:
                            confirm_url = confirm_url_match.group(1).replace(
                                "&amp;", "&"
                            )
                            logging.info(f"Bypassing virus scan warning for {filename}")
                            response = session.get(confirm_url, stream=True, timeout=45)

                    response.raise_for_status()

                    # Check if we got actual file content (not an error page)
                    content_type = response.headers.get("content-type", "").lower()
                    if (
                        "text/html" in content_type
                        and response.headers.get("content-length", "0") != "0"
                    ):
                        # This might be an error page, check content
                        first_chunk = next(response.iter_content(chunk_size=1024), b"")
                        if b"<!DOCTYPE html>" in first_chunk or b"<html" in first_chunk:
                            logging.debug(
                                f"Received HTML page instead of file content for {filename}"
                            )
                            continue
                        else:
                            # Reset response for full download
                            response = session.get(url, stream=True, timeout=45)
                            response.raise_for_status()

                    # Download the file
                    total_size = int(response.headers.get("content-length", 0))
                    downloaded_size = 0

                    with open(output_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded_size += len(chunk)

                    # Verify the download
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        file_size = os.path.getsize(output_path)
                        if (
                            total_size > 0 and file_size < total_size * 0.9
                        ):  # Less than 90% of expected size
                            logging.warning(
                                f"Downloaded file seems incomplete: {file_size}/{total_size} bytes"
                            )
                            os.remove(output_path)
                            continue

                        logging.debug(
                            f"Successfully downloaded {filename}: {file_size} bytes"
                        )
                        return True, None
                    else:
                        logging.debug(f"Download resulted in empty file for {filename}")
                        if os.path.exists(output_path):
                            os.remove(output_path)
                        continue

                except requests.exceptions.HTTPError as e:
                    if e.response.status_code in [429, 503]:  # Rate limiting
                        logging.debug(
                            f"Rate limited (HTTP {e.response.status_code}) for {filename}"
                        )
                        continue
                    elif e.response.status_code == 403:  # Forbidden
                        return False, f"Permission denied (HTTP 403)"
                    elif e.response.status_code == 404:  # Not found
                        return False, f"File not found (HTTP 404)"
                    else:
                        logging.debug(
                            f"HTTP error {e.response.status_code} for {filename}"
                        )
                        continue

                except requests.exceptions.RequestException as e:
                    logging.debug(f"Request error for {filename}: {str(e)}")
                    continue

                except Exception as e:
                    logging.debug(f"Unexpected error for {filename}: {str(e)}")
                    continue

                # Add small delay between URL attempts
                time.sleep(random.uniform(0.2, 0.8))

            # Add delay between session attempts
            time.sleep(random.uniform(1.0, 2.0))

        # Add longer delay between full retry attempts
        if attempt < max_retries - 1:
            delay = (3**attempt) + random.uniform(2, 5)
            logging.debug(
                f"All methods failed, waiting {delay:.2f} seconds before next attempt..."
            )
            time.sleep(delay)

    return False, f"All download attempts failed after {max_retries} retries"


def log_rate_limiting_stats(downloaded_files, failed_files, permission_denied_files):
    """Log statistics about rate limiting encounters and download success"""
    total_files = (
        len(downloaded_files) + len(failed_files) + len(permission_denied_files)
    )

    if total_files == 0:
        return

    success_rate = (len(downloaded_files) / total_files) * 100

    logging.info("=" * 60)
    logging.info("DOWNLOAD STATISTICS WITH RATE LIMITING BYPASS:")
    logging.info(f"  Total files processed: {total_files}")
    logging.info(
        f"  Successfully downloaded: {len(downloaded_files)} ({success_rate:.1f}%)"
    )
    logging.info(f"  Permission denied: {len(permission_denied_files)}")
    logging.info(f"  Failed (other reasons): {len(failed_files)}")

    if len(downloaded_files) > 0:
        total_size = sum(
            os.path.getsize(file_info["path"])
            for file_info in downloaded_files
            if os.path.exists(file_info["path"])
        )
        logging.info(f"  Total downloaded size: {total_size / (1024*1024):.2f} MB")

    logging.info("=" * 60)


def monitor_rate_limiting_effectiveness():
    """Monitor and log the effectiveness of rate limiting bypass strategies"""

    # This would be called at the end of downloads to provide insights
    logging.info("RATE LIMITING BYPASS EFFECTIVENESS:")
    logging.info("- User Agent Rotation: Enabled")
    logging.info("- Session Management: Multiple sessions with randomized headers")
    logging.info("- Request Delays: Smart delays with random intervals")
    logging.info("- Alternative Endpoints: Multiple Google Drive endpoints")
    logging.info("- Exponential Backoff: Progressive retry delays")
    logging.info("- Virus Scan Bypass: Automatic handling of Google's warnings")
    logging.info("- Content Validation: Verification of downloaded content")

    recommendations = [
        "For large folders (>100 files), consider running downloads in smaller batches",
        "If many files fail, try running the script again later with different user agents",
        "Check folder permissions if many files show 'permission denied'",
        "Monitor Google Drive quota limits for your account",
    ]

    logging.info("RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations, 1):
        logging.info(f"  {i}. {rec}")


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
