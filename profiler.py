import modal
import os
import time
import logging

# Define image with Middleware APM installed
image = (
    modal.Image.debian_slim()
    .apt_install("git")  # Required for middleware package
    .pip_install(
        "middleware-io", "middleware-io[profiling]"  # For continuous profiling
    )
    .run_commands("middleware-bootstrap -a install")
    .run_commands("export MW_TRACKER=True")
    .run_commands("export MW_APM_COLLECT_PROFILING=True")
    .run_commands("export MW_SERVICE_NAME='MyPythonApp'")
)

app = modal.App("hello-world-app", image=image)


@app.function(secrets=[modal.Secret.from_name("middleware")])
def hello():
    logging.info("Function started")

    # Get credentials from environment
    api_key = os.environ.get("MW_API_KEY")
    target = os.environ.get("MW_TARGET")
    logging.info("Credentials retrieved")

    # Import the mw_tracker from middleware to your app
    from middleware import mw_tracker, MWOptions, record_exception, DETECT_AWS_EC2

    logging.info("Middleware imports completed")

    mw_tracker(
        MWOptions(
            access_token=api_key,
            target=target,
            service_name="hello-world-app",
            otel_propagators="b3,tracecontext",
            detectors=[DETECT_AWS_EC2],
            console_exporter=True,  # add to console log telemetry data
            log_level="DEBUG",
        )
    )
    logging.info("Middleware tracker initialized")

    # Simulate some work
    time.sleep(0.5)
    logging.info("Processing step 1 completed")

    time.sleep(0.3)
    logging.info("Processing step 2 completed")

    logging.info("Hello world execution completed")

    return {"status": "success", "message": "hello world"}


@app.local_entrypoint()
def main():
    result = hello.remote()
    print(f"Result: {result}")
