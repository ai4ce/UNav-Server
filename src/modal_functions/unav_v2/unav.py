from modal import method, gpu, build, enter

from modal_config import app, unav_image, volume


@app.cls(
    image=unav_image,
    volumes={"/root/UNav-IO": volume},
    gpu=gpu.Any(),
    enable_memory_snapshot=True,
    concurrency_limit=20,
    allow_concurrent_inputs=20,
    memory=16152,
    container_idle_timeout=60,
)
class UnavServer:
    @method()
    def start_server(self):
        import json

        """
        Initializes and starts the serverless instance.
    
        This function helps in reducing the server response time for actual requests by pre-warming the server. 
        By starting the server in advance, it ensures that the server is ready to handle incoming requests immediately, 
        thus avoiding the latency associated with a cold start.
        """
        print("UNAV Container started...")

        response = {"status": "success", "message": "Server started."}
        return json.dumps(response)

    @method()
    def get_destinations(
        self,
        floor="6_floor",
        place="NewYorkCity",
        building="LightHouse",
    ):
        try:
            import os
            import sys

            # Add current directory to Python path for proper imports
            current_dir = os.getcwd()
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
                print(f"Added {current_dir} to Python path")

            # Print current directory and its contents
            print(f"Current directory: {current_dir}")

            # Print Python path
            print("Python path (sys.path):")
            for path in sys.path:
                print(f"  {path}")

            print("Contents of current directory:")
            for item in os.listdir(current_dir):
                print(f"  {item}")

            # Check if 'core' directory exists
            if os.path.exists("core"):
                print("Contents of 'core' directory:")
                for item in os.listdir("core"):
                    print(f"  core/{item}")

                # Check if 'core/tasks' directory exists
                if os.path.exists("core/tasks"):
                    print("Contents of 'core/tasks' directory:")
                    for item in os.listdir("core/tasks"):
                        print(f"  core/tasks/{item}")
            else:
                print("'core' directory does not exist in current directory")

            # Check what's in core/unav_state.py (the actual problematic file)
            if os.path.exists("core/unav_state.py"):
                with open("core/unav_state.py", "r") as f:
                    content = f.read()
                    print("Contents of core/unav_state.py:")
                    print("=" * 50)
                    print(content[:500])  # First 500 chars to see the imports
                    print("=" * 50)

            # Check what's in config.py
            if os.path.exists("config.py"):
                with open("config.py", "r") as f:
                    content = f.read()
                    print("Contents of config.py:")
                    print("=" * 50)
                    print(content[:500])  # First 500 chars
                    print("=" * 50)

            # Try to fix the import by temporarily modifying sys.modules
            print("Attempting to fix the import issue...")

            try:
                # Create a mock unav module with config
                import types
                import config as root_config

                # Create unav module
                unav_module = types.ModuleType("unav")
                unav_module.config = root_config

                # Add to sys.modules
                sys.modules["unav"] = unav_module
                sys.modules["unav.config"] = root_config

                print("✓ Created mock unav module with config")

                # Now try importing
                import core.tasks.unav

                print("✓ Successfully imported core.tasks.unav after fixing imports")

                # Get the function
                get_destinations = getattr(core.tasks.unav, "get_destinations", None)
                if get_destinations:
                    print("✓ Successfully found get_destinations function")
                else:
                    print("✗ get_destinations function not found")
                    return {"error": "get_destinations function not found"}

            except Exception as e:
                print(f"Mock import method failed: {e}")
                import traceback

                traceback.print_exc()
                return {"error": f"Failed to fix imports: {e}"}


            # Actually call the function now
            print("Retrieving destinations...")
            print(f"📍 Place: {place}")
            print(f"🏢 Building: {building}")
            print(f"🏠 Floor: {floor}")

            result = get_destinations(
                inputs={"floor": floor, "place": place, "building": building}
            )
            print(result)
            return result
        except ImportError as e:
            print(f"Import error: {e}")
            return {"error": f"Failed to import required modules: {e}"}
        except Exception as e:
            print(f"Error getting destinations: {e}")
            return {"error": f"Failed to get destinations: {e}"}
