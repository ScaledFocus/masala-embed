import modal
import os
from typing import Optional, List, Dict, Any

app = modal.App("serve-qwen")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "llama-cpp-python>=0.2.90",
        "boto3",
        "fastapi[standard]",
    ])
)

@app.cls(
    image=image,
    cpu=4,
    memory=8192,  # 8GB memory for GGUF model loading
    keep_warm=1,
    secrets=[modal.Secret.from_name("cloudflare-r2")],
    timeout=300  # 5 minute timeout
)
class QwenModel:
    def __init__(self):
        self.model: Optional[Any] = None
        self.model_path = "/tmp/qwen-q4_0.gguf"

    @modal.enter()
    def load_model(self):
        import boto3
        from llama_cpp import Llama

        try:
            print("Downloading GGUF model from R2...")

            # Download model from R2
            s3 = boto3.client(
                's3',
                endpoint_url=f"https://{os.environ['CLOUDFLARE_ACCOUNT_ID']}.r2.cloudflarestorage.com",
                aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
                aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
                region_name='auto'
            )

            bucket_name = os.environ.get('R2_BUCKET_NAME', 'masala-embed-models')

            # Check if file exists in bucket
            try:
                s3.head_object(Bucket=bucket_name, Key="qwen-q4_0.gguf")
                print("GGUF file found in R2, downloading...")
            except Exception as e:
                print(f"GGUF file not found in R2: {e}")
                raise FileNotFoundError("qwen-q4_0.gguf not found in R2. Run quantization first.")

            s3.download_file(bucket_name, "qwen-q4_0.gguf", self.model_path)

            # Verify download
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Failed to download model to {self.model_path}")

            file_size = os.path.getsize(self.model_path)
            print(f"Downloaded GGUF model: {file_size / (1024*1024):.1f} MB")

            if file_size < 1024 * 1024:  # Less than 1MB
                raise ValueError(f"Downloaded model seems corrupted (size: {file_size} bytes)")

            print("Loading GGUF model with llama.cpp...")

            # Validate GGUF file before loading
            try:
                with open(self.model_path, 'rb') as f:
                    header = f.read(8)
                    if not header.startswith(b'GGUF'):
                        raise ValueError(f"File is not a valid GGUF format (header: {header})")
                print("GGUF file format validated")
            except Exception as e:
                print(f"❌ GGUF validation failed: {e}")
                raise

            # Load quantized GGUF model with conservative settings
            try:
                self.model = Llama(
                    model_path=self.model_path,
                    embedding=True,
                    n_ctx=512,
                    n_batch=128,  # Smaller batch size for stability
                    n_threads=4,  # Use available CPUs
                    verbose=True,  # Enable verbose for debugging
                    use_mlock=False,  # Don't lock memory
                    use_mmap=True   # Use memory mapping for efficiency
                )
            except Exception as e:
                print(f"❌ Llama model loading failed: {e}")
                print(f"Model path: {self.model_path}")
                print(f"File exists: {os.path.exists(self.model_path)}")
                print(f"File size: {os.path.getsize(self.model_path) if os.path.exists(self.model_path) else 'N/A'}")
                raise

            print("GGUF model loaded successfully!")

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.model = None
            raise

    @modal.method()
    def embed(self, text: Optional[str] = None, texts: Optional[List[str]] = None) -> Dict[str, Any]:
        if self.model is None:
            return {"error": "Model not loaded"}

        try:
            if texts:
                # Batch processing with error handling
                embeddings = []
                for i, t in enumerate(texts):
                    if not t or not t.strip():
                        embeddings.append([0.0] * 1536)  # Default embedding size
                        continue

                    try:
                        result = self.model.create_embedding(t[:512])  # Truncate long texts
                        embeddings.append(result['data'][0]['embedding'])
                    except Exception as e:
                        print(f"Error embedding text {i}: {e}")
                        embeddings.append([0.0] * 1536)

                return {
                    "embeddings": embeddings,
                    "model": "qwen",
                    "count": len(embeddings)
                }
            elif text:
                # Single text with validation
                if not text or not text.strip():
                    return {"error": "Empty text provided"}

                result = self.model.create_embedding(text[:512])  # Truncate long texts
                return {
                    "embedding": result['data'][0]['embedding'],
                    "model": "qwen",
                    "text_length": len(text)
                }
            else:
                return {"error": "Need text or texts"}

        except Exception as e:
            print(f"Embedding error: {e}")
            return {"error": f"Embedding failed: {str(e)}"}

qwen_model = QwenModel()

@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def embed(data: Dict[str, Any]) -> Dict[str, Any]:
    return qwen_model.embed.remote(
        text=data.get("text"),
        texts=data.get("texts")
    )

@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def health() -> Dict[str, str]:
    return {"status": "healthy", "model": "qwen"}
