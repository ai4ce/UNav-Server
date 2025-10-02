import modal
import os
import shutil

volume = modal.Volume.from_name("NewVisiondata")
app = modal.App()


@app.function(volumes={"/root/UNav-IO": volume})
def copy_files_in_modal(source_path: str, destination_path: str) -> bool:
    """
    Copy files within Modal volume storage.

    Args:
        source_path (str): Source path of the file relative to volume mount point
        destination_path (str): Destination path relative to volume mount point

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Construct full paths within the volume
        source = os.path.join("/root/UNav-IO", source_path)
        destination = os.path.join("/root/UNav-IO", destination_path)

        # Create destination directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        # Copy the file
        shutil.copy2(source, destination)
        print(f"Successfully copied file from {source} to {destination}")
        return True

    except Exception as e:
        print(f"Error copying file: {str(e)}")
        return False


@app.local_entrypoint() 
def main():
    success = copy_files_in_modal.remote(
        "data/nyc/lighthouse/6th_floor/global_features_DinoV2Salad.h5",
        "data/nyc/global_features_DinoV2Salad.h5",
    )
    print(f"Operation {'succeeded' if success else 'failed'}")
