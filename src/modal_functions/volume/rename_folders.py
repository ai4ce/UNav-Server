import modal
import os

volume = modal.Volume.from_name("unav_multifloor")

app = modal.App()


@app.function(volumes={"/root/UNav-IO": volume})
def rename_specific_folder(old_folder_name: str, new_folder_name: str):
    base_path = "/root/UNav-IO"

    try:
        # Debug: List what's actually in the volume
        print(
            f"Base path contents: {os.listdir(base_path) if os.path.exists(base_path) else 'Base path does not exist'}"
        )

        # Check if data folder exists
        data_path = os.path.join(base_path, "data")
        print(f"Data path exists: {os.path.exists(data_path)}")
        if os.path.exists(data_path):
            print(f"Data folder contents: {os.listdir(data_path)}")

            # Check New_York_City folder
            nyc_path = os.path.join(data_path, "New_York_City")
            print(f"NYC path exists: {os.path.exists(nyc_path)}")
            if os.path.exists(nyc_path):
                print(f"NYC folder contents: {os.listdir(nyc_path)}")

                # Check LightHouse folder
                lighthouse_path = os.path.join(nyc_path, "LightHouse")
                print(f"LightHouse path exists: {os.path.exists(lighthouse_path)}")
                if os.path.exists(lighthouse_path):
                    print(f"LightHouse folder contents: {os.listdir(lighthouse_path)}")

        old_path = os.path.join(base_path, old_folder_name)
        new_path = os.path.join(base_path, new_folder_name)

        # Check if old folder exists
        if not os.path.exists(old_path):
            return {
                "success": False,
                "error": f"Folder '{old_folder_name}' does not exist",
            }

        # Check if new folder name already exists
        if os.path.exists(new_path):
            return {
                "success": False,
                "error": f"Folder '{new_folder_name}' already exists",
            }

        # Create parent directories for the new path if they don't exist
        new_parent_dir = os.path.dirname(new_path)
        if not os.path.exists(new_parent_dir):
            os.makedirs(new_parent_dir, exist_ok=True)
            print(f"Created parent directory: {new_parent_dir}")

        # Rename the directory
        os.rename(old_path, new_path)
        print(f"Renamed: {old_folder_name} → {new_folder_name}")

        return {
            "success": True,
            "message": f"Successfully renamed folder from '{old_folder_name}' to '{new_folder_name}'",
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# To run the function locally for testing
@app.local_entrypoint()
def main():
    # Example usage: rename "old_folder" to "new_folder"
    result = rename_specific_folder.remote(
        "data/New_York_City/LightHouse/17_floor", "data/New_York_City/Langone/17_floor"
    )
    print(result)
