from modal import method, gpu, build, enter

from modal_config import app, unav_image, volume


@app.cls(
    image=unav_image,
    volumes={"/root/UNav-IO": volume},
    gpu=gpu.Any(),
    enable_memory_snapshot=True,
    concurrency_limit=20,
    allow_concurrent_inputs=20,
    memory=49152,
    container_idle_timeout=900,
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
        from core.tasks.unav import get_destinations

        """
        Retrieves the list of available destinations from the UNAV model.
        """
        print("Retrieving destinations...")
        print(f"📍 Place: {place}")
        print(f"🏢 Building: {building}")
        print(f"🏠 Floor: {floor}")

        print(
            get_destinations(
                inputs={"floor": floor, "place": place, "building": building}
            )
        )
