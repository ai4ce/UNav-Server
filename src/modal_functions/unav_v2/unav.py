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
    container_idle_timeout=300,
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

            # Print current directory and its contents
            current_dir = os.getcwd()
            print(f"Current directory: {current_dir}")
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

            from core.tasks.unav import get_destinations

            # """
            # Retrieves the list of available destinations from the UNAV model.
            # """
            # print("Retrieving destinations...")
            # print(f"📍 Place: {place}")
            # print(f"🏢 Building: {building}")
            # print(f"🏠 Floor: {floor}")

            # result = get_destinations(
            #     inputs={"floor": floor, "place": place, "building": building}
            # )
            # print(result)
            # return result
        except ImportError as e:
            print(f"Import error: {e}")
            return {"error": f"Failed to import required modules: {e}"}
        except Exception as e:
            print(f"Error getting destinations: {e}")
            return {"error": f"Failed to get destinations: {e}"}
