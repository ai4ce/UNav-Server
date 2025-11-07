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
    # Initialize Middleware tracking with proper serverless configuration
    from middleware import mw_tracker, MWOptions
    from opentelemetry.trace import get_tracer
    from opentelemetry.metrics import get_meter_provider

    # Get credentials from environment
    api_key = os.environ.get("MW_API_KEY")
    target = os.environ.get("MW_TARGET")

    # Verify configuration
    print("=" * 60)
    print("MIDDLEWARE CONFIGURATION VERIFICATION")
    print("=" * 60)
    print(f"MW_API_KEY: {api_key}")
    print(f"MW_TARGET: {target}")
    print(f"API Key Length: {len(api_key) if api_key else 0}")
    print(f"Target is HTTPS: {target.startswith('https://') if target else False}")
    print("=" * 60)

    # Import the mw_tracker from middleware to your app
    from middleware import mw_tracker, MWOptions, record_exception, DETECT_AWS_EC2

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

    # Create custom tracer for tracking
    tracer = get_tracer("hello_world_tracer")

    # Create custom span
    with tracer.start_as_current_span("hello_function_execution") as span:
        span.set_attribute("function.name", "hello")
        span.set_attribute("execution.environment", "modal")

        # Create custom metrics
        meter = get_meter_provider().get_meter("hello_world_meter")
        execution_counter = meter.create_counter(
            "hello_executions", description="Number of hello function executions"
        )

        # Simulate some work
        with tracer.start_as_current_span("processing_step_1"):
            time.sleep(0.5)

        with tracer.start_as_current_span("processing_step_2"):
            time.sleep(0.3)

        # Log with structured data
        logging.info(
            "Hello world execution completed",
            extra={"step": "final", "status": "success"},
        )

        # Increment counter
        execution_counter.add(1, {"status": "success"})

    # Force flush telemetry data before function exits
    from opentelemetry import trace, metrics

    print("Flushing telemetry...")

    # Force flush trace provider
    trace_provider = trace.get_tracer_provider()
    if hasattr(trace_provider, "force_flush"):
        result = trace_provider.force_flush(timeout_millis=10000)
        print(f"Trace flush result: {result}")

    # Force flush metric provider
    metric_provider = metrics.get_meter_provider()
    if hasattr(metric_provider, "force_flush"):
        result = metric_provider.force_flush(timeout_millis=10000)
        print(f"Metric flush result: {result}")

    print("Telemetry flushed")

    return {"status": "success", "message": "hello world"}


@app.local_entrypoint()
def main():
    result = hello.remote()
    print(f"Result: {result}")
