import modal
import os
import logging
import traceback

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

volume = modal.Volume.from_name("NewVisiondata", create_if_missing=True)


def process_upload(remote_path, folder_path):
    with volume.batch_upload() as batch:
        try:
            if not os.path.exists(folder_path):
                logging.error(f"Error: The path '{folder_path}' does not exist")
                return False

            if os.path.isfile(folder_path):
                logging.info(f"Uploading file '{folder_path}' to '{remote_path}'")
                file_name = os.path.basename(folder_path)
                destination = os.path.join(remote_path, file_name)
                batch.put_file(folder_path, destination)
            else:
                logging.info(f"Uploading directory '{folder_path}' to '{remote_path}'")
                batch.put_directory(folder_path, remote_path)

            logging.info(
                f"Successfully uploaded '{folder_path}' to '{remote_path}' in the Modal volume."
            )
            return True
        except Exception as e:
            logging.error(
                f"Failed to upload '{folder_path}' to '{remote_path}': {str(e)}"
            )
            logging.error(traceback.format_exc())
            return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 5:
        print("Usage: python script_name.py <place> <building> <floor> <local path>")
        print("Note: <local path> can be a file or directory")
        sys.exit(1)

    arg1, arg2, arg3, local_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    remote_path = os.path.join("data", arg1, arg2, arg3)

    success = process_upload(remote_path, local_path)
    if not success:
        sys.exit(1)
