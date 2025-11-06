import modal

app = modal.App("hello-world-app")

@app.function()
def hello():
    print("hello world")

@app.local_entrypoint()
def main():
    hello.remote()