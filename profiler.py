import modal
import time
import logging

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("middleware-io", "middleware-io[profiling]")
    .env(
        {
            "MW_SERVICE_NAME": "MyPythonApp-Final",
            "MW_APM_COLLECT_PROFILING": "true",
            "MW_TRACKER": "true",
            "MW_CONSOLE_EXPORTER": "false",
            # Crucial for debugging logs
            "OTEL_SERVICE_NAME": "modal-unav-server",
        }
    )
    .run_commands(
        "middleware-bootstrap -a install",
        "echo '#!/bin/sh' > /root/run.sh",
        "echo 'exec middleware-run \"$@\"' >> /root/run.sh",
        "chmod +x /root/run.sh",
    )
    .dockerfile_commands('ENTRYPOINT ["/root/run.sh"]')
)

app = modal.App("hello-world-app", image=image)


@app.function(secrets=[modal.Secret.from_name("middleware")])
def hello():
    from middleware import mw_tracker, MWOptions
    from opentelemetry import trace
    import os

    api_key = os.environ.get("MW_API_KEY")
    target = os.environ.get("MW_TARGET")

    if not api_key or not target:
        raise ValueError("MW_API_KEY and MW_TARGET environment variables must be set")

    mw_tracker(
        MWOptions(
            access_token=api_key,
            target=target,
            service_name="MyPythonApp-Final",
            console_exporter=False,
            log_level="INFO",
            collect_profiling=True,
            collect_traces=True,
            collect_metrics=True,
        )
    )

    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("hello_function") as span:

        def child_function_1():
            with tracer.start_as_current_span("child_function_1") as child_span:
                child_span.set_attribute("operation", "task_1")
                time.sleep(0.5)  # 0.5 seconds

        def child_function_2():
            with tracer.start_as_current_span("child_function_2") as child_span:
                child_span.set_attribute("operation", "task_2")
                time.sleep(1.0)  # 1 second

        def child_function_3():
            with tracer.start_as_current_span("child_function_3") as child_span:
                child_span.set_attribute("operation", "task_3")
                time.sleep(1.5)  # 1.5 seconds

        logging.info(
            "Function started. Middleware is now correctly wrapping the Modal worker."
        )

        # Call child functions
        child_function_1()
        child_function_2()
        child_function_3()

        logging.info("Processing complete. Check logs for telemetry output.")
        return {
            "status": "success",
            "message": "hello world from a non-crashing container",
        }


@app.local_entrypoint()
def main():
    # This is not used for deployment, only for local testing.
    pass
