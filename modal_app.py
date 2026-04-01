import modal

app = modal.App("unav-server-anbang-copy")
image = modal.Image.from_dockerfile("Dockerfile", add_python="3.10").entrypoint([])


@app.function(image=image, gpu="A10")
@modal.web_server(port=5001)
def web():
    import subprocess

    subprocess.Popen(
        [
            "bash",
            "-c",
            "source /opt/conda/etc/profile.d/conda.sh && conda activate unav && uvicorn main:app --host 0.0.0.0 --port 5001 --log-level info",
        ]
    )
