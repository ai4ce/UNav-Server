import modal


# Define image with Middleware APM installed
image = (
    modal.Image.debian_slim()
    .apt_install("git")  # Required for middleware package
    .pip_install(
        "middleware-io", "middleware-io[profiling]"  # For continuous profiling
    )
    .run_commands("middleware-bootstrap -a install")
)

app = modal.App("hello-world-app", image=image)


@app.function(secrets=[modal.Secret.from_name("middleware")])
def hello():
    # Initialize Middleware tracking
    from middleware import mw_tracker, MWOptions

    mw_tracker(
        MWOptions(
            service_name="hello-world-app",
            console_exporter=True,  # For debugging, remove in production
            log_level="INFO",
        )
    )

    print("hello world")


@app.local_entrypoint()
def main():
    hello.remote()
