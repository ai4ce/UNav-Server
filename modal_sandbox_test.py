import modal
import time

app = modal.App("unav-sandbox-test")

app_image = (
    modal.Image.from_dockerfile(
        "Dockerfile",
        context_dir=".",
        ignore=[".venv", "__pycache__", ".git", ".modal-cache"],
        add_python="3.10",
    )
    .run_commands("ln -sf /opt/conda/envs/unav/bin/python3 /usr/bin/python")
    .entrypoint([])
)


@app.function(image=app_image, timeout=3600)
def run_test():
    print("Creating sandbox...")
    sb = modal.Sandbox.create(
        "sleep",
        "infinity",
        image=app_image,
        timeout=3600,
    )

    print("Starting server inside sandbox...")
    sb.exec(
        "bash",
        "-c",
        "source /opt/conda/etc/profile.d/conda.sh && "
        "conda activate unav && "
        "uvicorn main:app --host 0.0.0.0 --port 5001 &",
    )

    print("Waiting 15s for server to boot...")
    time.sleep(15)

    print("Creating public tunnel...")
    tunnel = sb.tunnel(5001)
    print(f"\n✅ Server is live at: {tunnel.url}")
    print(f"   Try: {tunnel.url}/docs")
    print(f"   Try: {tunnel.url}/predict\n")

    try:
        input("Press Enter to stop the sandbox...\n")
    except (EOFError, KeyboardInterrupt):
        pass

    print("Stopping sandbox...")
    sb.terminate()


if __name__ == "__main__":
    with modal.enable_output():
        with app.run():
            run_test.remote()
