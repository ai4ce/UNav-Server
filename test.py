import modal

def main():
    try:
        # 1. Look up the deployed function by its app name and function name.
        #    The app name is "hello-world-app" from your profiler.py.
        #    The function name is "hello".
        f = modal.Function.lookup("hello-world-app", "hello")
        print("Successfully found the deployed function 'hello'.")

        # 2. Trigger the function using .remote()
        print("Calling the function on Modal's servers...")
        result = f.remote()

        # 3. Print the result
        print(f"Result from deployed function: {result}")

    except modal.exception.NotFoundError:
        print("Error: Could not find the deployed app 'hello-world-app'.")
        print("Please make sure you have successfully run 'modal deploy profiler.py'.")

if __name__ == "__main__":
    main()
