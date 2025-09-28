import subprocess

import modal

vllm_image = (
    # Modal documentation for vLLM uses devel rather than runtime image of CUDA.
    # Compilation of dependencies listed below requires CUDA toolkit.
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install(
        "vllm==0.10.1.1",
        "huggingface_hub[hf_transfer]==0.34.4",
        # From vLLM
        #
        # There are no pre-built vllm wheels containing Flash Infer,
        #   so you must install it in your environment first.
        "flashinfer-python==0.2.8",
        "torch==2.7.1",
    )
    # From HuggingFace:
    #
    # Set to True for faster uploads and downloads from the Hub using hf_transfer.
    # By default, huggingface_hub uses the Python-based httpx.get
    #   and httpx.post functions.
    # Although these are reliable and versatile,
    #   they may not be the most efficient choice for machines with high bandwidth.
    # hf_transfer is a Rust-based package developed to maximize the bandwidth
    #   used by dividing large files into smaller parts
    #   and transferring them simultaneously using multiple threads.
    # This approach can potentially double the transfer speed.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
# The `main` branch commit hash as of 2025-09-28
MODEL_REVISION = "c54f2e6e80b2d7b7de06f51cec4959f6b3e03418"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("qwen3-embedding-inference")

NUM_GPU = 1
MINUTES = 60  # seconds
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"A100-40GB:{NUM_GPU}",
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        MODEL_NAME,
        "--task",
        "embed",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--enforce-eager",
        "--tensor-parallel-size",
        str(NUM_GPU),
    ]

    subprocess.Popen(" ".join(cmd), shell=True)
