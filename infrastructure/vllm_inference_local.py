import os
import subprocess
import sys

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
MODEL_REVISION = "c54f2e6e80b2d7b7de06f51cec4959f6b3e03418"
VLLM_PORT = 8000
HOST = "0.0.0.0"


def setup_environment():
    """Set up environment variables for optimal performance."""
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def start_vllm_server():
    print(f"Starting vLLM embedding server for {MODEL_NAME}")
    print(f"Server will be available at: http://{HOST}:{VLLM_PORT}")
    print("API endpoint: /v1/embeddings")
    print("\nPress Ctrl+C to stop the server\n")

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        MODEL_NAME,
        "--task",
        "embed",
        "--host",
        HOST,
        "--port",
        str(VLLM_PORT),
        "--enforce-eager",
        "--tensor-parallel-size",
        "1",
        "--uvicorn-log-level",
        "info",
    ]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    start_vllm_server()
