import modal
import os

volume = modal.Volume.from_name("NewVisiondata")

app = modal.App()


@app.function(volumes={"/root/UNav-IO": volume})
def rename_specific_folder(old_folder_name: str, new_folder_name: str):
    base_path = "/root/UNav-IO"

    try:
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
        "data/New_York_City/MapConnnection_Graph.pkl", "data/New_York_City/MapConnnection_Graph_3_floor.pkl"
    )
    # result = rename_specific_folder.remote(
    #     "data/New_York_City/LightHouse/3_floor", "data/nyc/LightHouse/3_floor"
    # )
    print(result)
