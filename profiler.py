import modal
import time
import logging

# This image configuration is the key to fixing the crash loop.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("middleware-io", "middleware-io[profiling]")
    .run_commands(
        "middleware-bootstrap -a install",
        # 1. Create a wrapper script named 'run.sh'.
        #    This script will execute any command it receives ("$@")
        #    inside 'middleware-run'.
        "echo '#!/bin/sh' > /root/run.sh",
        "echo 'exec middleware-run \"$@\"' >> /root/run.sh",
        "chmod +x /root/run.sh",
    )
    # 2. Set our wrapper script as the entrypoint. Modal's own startup
    #    command will now be passed to our script as "$@".
    .dockerfile_commands('ENTRYPOINT ["/root/run.sh"]')
)

app = modal.App(
    "hello-world-app",
    image=image
)

@app.function(
    # This part remains correct from our last version
    secrets=[
        modal.Secret.from_name("middleware"),
        modal.Secret.from_dict({
            "MW_TRACKER": "True",
            "MW_APM_COLLECT_PROFILING": "True",
            "MW_SERVICE_NAME": "MyModalApp-Fixed",
        })
    ]
)
def hello():
    # Your function code doesn't need to change
    logging.info("Function started. Middleware is now correctly wrapping the Modal worker.")
    time.sleep(1)
    logging.info("Processing complete.")
    return {"status": "success", "message": "hello world from a non-crashing container"}

@app.local_entrypoint()
def main():
    print("Running local entrypoint to test the remote function...")
    result = hello.remote()
    print(f"Result from remote function: {result}")
