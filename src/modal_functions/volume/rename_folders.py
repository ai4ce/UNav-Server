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


@app.function(volumes={"/root/UNav-IO": volume})
def bulk_rename_files(rename_mapping: dict):
    """
    Bulk rename files based on a mapping dictionary.

    Args:
        rename_mapping: Dict with old paths as keys and new paths as values
                       (paths should be relative to the volume root)

    Returns:
        Dict with results of the operation
    """
    base_path = "/root/UNav-IO"
    results = {"successful": [], "failed": []}

    for old_path_rel, new_path_rel in rename_mapping.items():
        try:
            old_path = os.path.join(base_path, old_path_rel)
            new_path = os.path.join(base_path, new_path_rel)

            # Check if old file exists
            if not os.path.exists(old_path):
                results["failed"].append(
                    {
                        "old_path": old_path_rel,
                        "new_path": new_path_rel,
                        "error": f"File/folder '{old_path_rel}' does not exist",
                    }
                )
                continue

            # Check if new file name already exists
            if os.path.exists(new_path):
                results["failed"].append(
                    {
                        "old_path": old_path_rel,
                        "new_path": new_path_rel,
                        "error": f"File/folder '{new_path_rel}' already exists",
                    }
                )
                continue

            # Create parent directory if it doesn't exist
            parent_dir = os.path.dirname(new_path)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)

            # Rename the file/directory
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path_rel} → {new_path_rel}")

            results["successful"].append(
                {"old_path": old_path_rel, "new_path": new_path_rel}
            )

        except Exception as e:
            results["failed"].append(
                {"old_path": old_path_rel, "new_path": new_path_rel, "error": str(e)}
            )

    return results


# To run the function locally for testing
@app.local_entrypoint()
def main():
    # Example 1: Rename a single folder
    # result = rename_specific_folder.remote(
    #     "data/New_York_City/LightHouse/6_floor/0topo-map.json",
    #     "data/New_York_City/LightHouse/6_floor/old_0topo-map.json"
    # )

    # Example 2: Bulk rename multiple files
    rename_mapping = {
        "data/New_York_City/LightHouse/6_floor/boundaries.json": "data/New_York_City/LightHouse/6_floor/old_two_boundaries.json",
        "data/New_York_City/LightHouse/6_floor/intersection.json": "data/New_York_City/LightHouse/6_floor/old_intersection.json",
        "data/New_York_City/LightHouse/6_floor/access_graph.npy": "data/New_York_City/LightHouse/6_floor/old_access_graph.npy",
        "data/New_York_City/LightHouse/6_floor/destination.json": "data/New_York_City/LightHouse/6_floor/old_destination.json",
        "data/New_York_City/LightHouse/6_floor/path.h5": "data/New_York_City/LightHouse/6_floor/old_two_path.h5",
    }

    result = bulk_rename_files.remote(rename_mapping)
    print(result)
