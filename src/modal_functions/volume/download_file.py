import modal
import os

volume = modal.Volume.from_name("NewVisiondata")

app = modal.App()  # Changed from Stub to App


@app.function(volumes={"/root/UNav-IO": volume})  # Changed stub to app
def download_file_from_volume(file_path: str, local_path: str):
    """
    Downloads a file from a Modal volume to a local path.
    Args:
        file_path: The path to the file within the Modal volume (e.g., /root/UNav-IO/data/myfile.txt).
        local_path: The local path where the file should be downloaded (e.g., /Users/youruser/Downloads/myfile.txt).
    """
    import shutil

    # Check if the file exists in the volume
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found in volume: {file_path}")

    # Ensure the local directory exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Copy the file from the volume to the local path
    shutil.copy2(file_path, local_path)
    print(f"Downloaded {file_path} from volume to {local_path}")


@app.local_entrypoint()  # Changed stub to app
def main():
    volume_file_path = (
        "/root/UNav-IO/data/MapConnnection_Graph.pkl"
    )
    local_download_path = ""  # Change this to your desired local path

    download_file_from_volume.remote(volume_file_path, local_download_path)
    print("Download complete.")
