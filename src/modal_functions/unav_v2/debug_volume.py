from modal import method, gpu
from modal_config import app, unav_image, volume


@app.function(
    image=unav_image,
    volumes={"/root/UNav-IO": volume},
)
def debug_volume_contents():
    import os

    def explore_directory(path, max_depth=3, current_depth=0):
        """Recursively explore directory structure"""
        if current_depth > max_depth:
            return

        try:
            items = os.listdir(path)
            for item in sorted(items):
                item_path = os.path.join(path, item)
                indent = "  " * current_depth

                if os.path.isdir(item_path):
                    print(f"{indent}📂 {item}/")
                    # Don't go too deep into common system directories
                    if (
                        item not in [".git", "__pycache__", ".cache"]
                        and current_depth < max_depth
                    ):
                        explore_directory(item_path, max_depth, current_depth + 1)
                else:
                    file_size = os.path.getsize(item_path)
                    print(f"{indent}📄 {item} ({file_size} bytes)")
        except PermissionError:
            print(f"{indent}[Permission denied]")
        except Exception as e:
            print(f"{indent}[Error: {e}]")

    print("🔍 Complete volume structure at /root/UNav-IO:")
    explore_directory("/root/UNav-IO", max_depth=4)

    # Also check if data is organized differently
    print("\n🔍 Looking for any map-related files:")
    for root, dirs, files in os.walk("/root/UNav-IO"):
        for file in files:
            if any(
                keyword in file.lower()
                for keyword in ["boundaries", "colmap", "camera", "image", "bin"]
            ):
                print(f"  📄 {os.path.join(root, file)}")

    return {"status": "debug_complete"}


@app.local_entrypoint()
def main():
    result = debug_volume_contents.remote()
    print(result)
