import modal

app = modal.App()

original_volume = modal.Volume.from_name("NewVisiondata")
backup_volume = modal.Volume.from_name("backupNewVistionData", create_if_missing=True)


@app.function(volumes={"/original": original_volume, "/backup": backup_volume})
def copy_files():
    import shutil
    import os

    # Copy all files from original to backup
    for root, dirs, files in os.walk("/original"):
        print(files)
        for file in files:
            src_path = os.path.join(root, file)
            rel_path = os.path.relpath(src_path, "/original")
            dest_path = os.path.join("/backup", rel_path)
            dest_dir = os.path.dirname(dest_path)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(src_path, dest_path)

    backup_volume.commit()  # Ensure changes are persisted


@app.local_entrypoint()
def main():
    copy_files.remote()
    print("Files copied from original to backup volume.")
