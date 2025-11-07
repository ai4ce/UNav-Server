import modal
import time
import logging

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("middleware-io", "middleware-io[profiling]")
    .run_commands(
        "middleware-bootstrap -a install",
        # This creates the CORRECT wrapper script inside the container
        "echo '#!/bin/sh' > /root/run.sh",
        "echo 'exec middleware-run \"$@\"' >> /root/run.sh",
        "chmod +x /root/run.sh",
    )
    .dockerfile_commands('ENTRYPOINT ["/root/run.sh"]')
)

app = modal.App(
    "hello-world-app",
    image=image
)

@app.function(
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
    logging.info("Function started. Middleware is now correctly wrapping the Modal worker.")
    time.sleep(1)
    logging.info("Processing complete.")
    return {"status": "success", "message": "hello world from a non-crashing container"}

@app.local_entrypoint()
def main():
    result = hello.remote()
    print(f"Result from remote function: {result}")

