import modal
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

volume = modal.Volume.from_name("NewVisiondata", create_if_missing=True)


def process_upload(remote_path, folder_path):
    with volume.batch_upload() as batch: 
        try:
            batch.put_directory(folder_path, remote_path)
            logging.info(f"Successfully uploaded '{folder_path}' to '{remote_path}' in the Modal volume.")
        except Exception as e:
            logging.error(f"Failed to upload '{folder_path}' to '{remote_path}': {e}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 5:
        print("Usage: python script_name.py <place> <building> <floor> <local folder path>")
        sys.exit(1)

    arg1, arg2, arg3, folder_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    remote_path = os.path.join("data",arg1, arg2, arg3)

    process_upload(remote_path, folder_path)
