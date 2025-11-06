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

    print(f"Initializing Middleware with target: {target}")

    # Initialize tracker with serverless configuration
    mw_tracker(
        MWOptions(
            access_token=api_key,  # Required for serverless
            target=target,  # Required for serverless
            service_name="hello-world-app",
            console_exporter=True,  # For debugging
            log_level="DEBUG",
            collect_traces=True,
            collect_metrics=True,
            collect_logs=True,
            collect_profiling=True,
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

        # Add some actual work to trace
        print("Starting hello world execution...")

        # Simulate some work
        with tracer.start_as_current_span("processing_step_1"):
            time.sleep(0.5)
            print("Step 1 completed")

        with tracer.start_as_current_span("processing_step_2"):
            time.sleep(0.3)
            print("Step 2 completed")

        # Log with structured data
        logging.info(
            "Hello world execution completed",
            extra={"step": "final", "status": "success"},
        )

        # Increment counter
        execution_counter.add(1, {"status": "success"})

        print("hello world")

    # Force flush telemetry data before function exits
    print("Flushing telemetry data to Middleware...")
    from opentelemetry import trace, metrics

    # Flush trace provider
    trace_provider = trace.get_tracer_provider()
    if hasattr(trace_provider, "force_flush"):
        trace_provider.force_flush(timeout_millis=5000)
        print("Traces flushed")

    # Flush metric provider
    metric_provider = metrics.get_meter_provider()
    if hasattr(metric_provider, "force_flush"):
        metric_provider.force_flush(timeout_millis=5000)
        print("Metrics flushed")

    # Small delay to ensure all data is sent
    time.sleep(2)
    print("Telemetry flush complete")

    return {"status": "success", "message": "hello world"}


@app.local_entrypoint()
def main():
    result = hello.remote()
    print(f"Result: {result}")
