import modal
import os
import subprocess
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = modal.App("quantize")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install([
        "git",
        "build-essential",
        "cmake",
        "python3-dev",
    ])
    .pip_install([
        "huggingface_hub>=0.20.0",
        "boto3>=1.34.0",
        "transformers>=4.45.0",
        "torch>=2.1.0",
        "safetensors",
        "mistral_common"
    ])
    .run_commands([
        # Clone and build llama.cpp using CMake
        "cd /tmp && git clone https://github.com/ggerganov/llama.cpp.git",
        "cd /tmp/llama.cpp && mkdir build && cd build",
        "cd /tmp/llama.cpp/build && cmake .. -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF",
        "cd /tmp/llama.cpp/build && make -j$(nproc)",
        # Install Python bindings
        "cd /tmp/llama.cpp && pip install -e .",
        # Make binaries available globally
        "ls -l /tmp/llama.cpp", # Diagnostic: list files after build
        "cp /tmp/llama.cpp/convert_hf_to_gguf.py /usr/local/bin/",
        "cp /tmp/llama.cpp/build/bin/llama-quantize /usr/local/bin/",
        "chmod +x /usr/local/bin/convert_hf_to_gguf.py",
        "chmod +x /usr/local/bin/llama-quantize",
    ])
)

def run_command(cmd: list, cwd: str, description: str) -> subprocess.CompletedProcess:
    """Run a subprocess command with proper error handling and logging."""
    logger.info(f"Running {description}...")
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd
        )
        if result.stdout:
            logger.info(f"{description} output: {result.stdout}")
        if result.stderr:
            logger.warning(f"{description} warnings: {result.stderr}")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"{description} failed: {e}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        raise

def validate_gguf_file(file_path: str) -> None:
    """Validate that the GGUF file exists and has valid format."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Quantized model not found at {file_path}")

    file_size = os.path.getsize(file_path)

    logger.info(f"Quantized model size: {file_size / (1024*1024):.1f} MB")

    # Validate GGUF file format
    logger.info("Validating GGUF file format...")
    try:
        with open(file_path, 'rb') as f:
            header = f.read(8)
            if not header.startswith(b'GGUF'):
                raise ValueError(f"File is not a valid GGUF format (header: {header})")
        logger.info("GGUF file format validation passed")
    except Exception as e:
        logger.error(f"GGUF validation failed: {e}")
        raise

def upload_to_r2(file_path: str, remote_filename: str) -> None:
    """Upload file to Cloudflare R2 storage."""
    logger.info("Uploading to R2...")
    import boto3

    s3 = boto3.client(
        's3',
        endpoint_url=f"https://{os.environ['CLOUDFLARE_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
        region_name='auto'
    )

    bucket_name = os.environ.get('R2_BUCKET_NAME', 'masala-embed-models')

    try:
        s3.upload_file(file_path, bucket_name, remote_filename)
        logger.info("Upload complete!")
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise



def generate_output_filename(model_name: str, quantization_type: str) -> str:
    """Generate output filename from model name and quantization type."""
    model_slug = model_name.replace("/", "-").lower()
    return f"{model_slug}-{quantization_type.lower()}.gguf"

def cleanup_files(*file_paths: str) -> None:
    """Clean up temporary files."""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up {file_path}: {e}")

@app.function(
    image=image,
    cpu=8,
    memory=16384,  # 16GB memory for quantization
    timeout=3600,  # 1 hour timeout
    secrets=[modal.Secret.from_name("cloudflare-r2")]
)
def quantize_model(
    model_name: str,
    quantization_type: str,
    output_filename: Optional[str] = None,
    precision: str = "f16"
) -> dict:
    """
    Quantize a HuggingFace model to GGUF format.

    Args:
        model_name: HuggingFace model identifier (e.g., "Qwen/Qwen2.5-0.5B")
        quantization_type: Type of quantization (Q4_0, Q8_0, etc.)
        output_filename: Custom output filename (auto-generated if None)
        precision: Precision type for GGUF conversion (f16, bf16, f32, etc.)

    Returns:
        Dict containing quantization results and metadata
    """
    from huggingface_hub import snapshot_download

    # Generate output filename if not provided
    if not output_filename:
        output_filename = generate_output_filename(model_name, quantization_type)

    # Setup work directory
    model_slug = model_name.replace("/", "_")
    work_dir = f"/tmp/{model_slug}_quantize"
    os.makedirs(work_dir, exist_ok=True)

    logger.info(f"Starting quantization for model: {model_name}")
    logger.info(f"Quantization type: {quantization_type}")
    logger.info(f"Output filename: {output_filename}")

    # Download model
    logger.info(f"Downloading model: {model_name}")
    model_path = snapshot_download(
        repo_id=model_name,
        local_dir=f"{work_dir}/original",
        ignore_patterns=["*.bin"]  # Skip .bin files, use safetensors
    )

    # Convert to GGUF format with specified precision
    gguf_path = f"{work_dir}/model-{precision}.gguf"
    convert_cmd = [
        "python3", "/usr/local/bin/convert_hf_to_gguf.py",
        model_path,
        "--outtype", precision,
        "--outfile", gguf_path
    ]

    run_command(convert_cmd, work_dir, "HF to GGUF conversion")

    # Quantize model
    quantized_path = f"{work_dir}/{output_filename}"
    quantize_cmd = [
        "/usr/local/bin/llama-quantize",
        gguf_path,
        quantized_path,
        quantization_type
    ]

    run_command(quantize_cmd, work_dir, f"{quantization_type} quantization")

    # Validate the quantized file
    validate_gguf_file(quantized_path)
    file_size = os.path.getsize(quantized_path)

    # Upload to R2
    upload_to_r2(quantized_path, output_filename)

    # Clean up large temporary files
    cleanup_files(gguf_path, quantized_path)

    return {
        "status": "success",
        "file": output_filename,
        "size_mb": round(file_size / (1024*1024), 1),
        "format": f"gguf_{quantization_type.lower()}",
        "model": model_name,
        "quantization_type": quantization_type
    }

@app.local_entrypoint()
def main(
    model_name: str,
    quantization_type: str,
    output_filename: Optional[str] = None,
    precision: str = "f16"
):
    """
    Main entrypoint for quantization process.

    Args:
        model_name: HuggingFace model to quantize
        quantization_type: Quantization method (Q4_0, Q8_0, etc.)
        output_filename: Custom output filename (optional)
        precision: Precision type for GGUF conversion (f16, bf16, f32, etc.)
    """
    logger.info("Starting GGUF quantization process...")
    result = quantize_model.remote(
        model_name=model_name,
        quantization_type=quantization_type,
        output_filename=output_filename,
        precision=precision
    )
    logger.info(f"Quantization result: {result}")
    return result
