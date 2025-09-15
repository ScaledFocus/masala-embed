import modal
import os
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration from environment
MODEL_NAME = os.environ.get("MODEL_NAME", "default")
MODEL_TYPE = os.environ.get("MODEL_TYPE", "text_embedding")
MODEL_SOURCE = os.environ.get("MODEL_SOURCE", "huggingface")
CPU_COUNT = int(os.environ.get("CPU_COUNT", "4"))
MEMORY_MB = int(os.environ.get("MEMORY_MB", "8192"))
GPU_TYPE = os.environ.get("GPU_TYPE")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "512"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1536"))

# R2 specific config
R2_MODEL_FILE = os.environ.get("R2_MODEL_FILE")
R2_FRAMEWORK = os.environ.get("R2_FRAMEWORK", "llama_cpp")

# HuggingFace specific config
HF_MODEL_ID = os.environ.get("HF_MODEL_ID")
HF_FRAMEWORK = os.environ.get("HF_FRAMEWORK", "transformers")

app = modal.App(f"serve-{MODEL_NAME}")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        [
            "llama-cpp-python>=0.2.90",
            "boto3",
            "torch",
            "transformers",
            "sentence-transformers",
            "pillow",
            "requests",
            "fastapi",  # required for web endpoints
        ]
    )
)

# Build decorator kwargs to handle optional parameters
cls_kwargs = {
    "image": image,
    "cpu": CPU_COUNT,
    "memory": MEMORY_MB,
    "min_containers": 1,
    "timeout": 300
}

# Add secrets for R2 models only
if MODEL_SOURCE == "r2":
    cls_kwargs["secrets"] = [modal.Secret.from_name("cloudflare-r2")]

# Add GPU if specified
if GPU_TYPE:
    cls_kwargs["gpu"] = GPU_TYPE

@app.cls(**cls_kwargs)
class UnifiedModel:
    model: Optional[Any] = None
    processor: Optional[Any] = None
    model_path: Optional[str] = None
    model_config: Dict[str, Any] = {}

    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration from environment."""
        return {
            "model_name": MODEL_NAME,
            "model_type": MODEL_TYPE,
            "model_source": MODEL_SOURCE,
            "r2_file": R2_MODEL_FILE,
            "r2_framework": R2_FRAMEWORK,
            "hf_model_id": HF_MODEL_ID,
            "hf_framework": HF_FRAMEWORK,
            "max_length": MAX_LENGTH,
            "embedding_dim": EMBEDDING_DIM
        }

    @modal.enter()
    def load_model(self):
        """Load model based on configuration."""
        self.model_config = self._load_config()
        logger.info(f"Loading model: {MODEL_NAME}")
        logger.info(f"Model type: {MODEL_TYPE}")
        logger.info(f"Model source: {MODEL_SOURCE}")

        try:
            if self.model_config["model_source"] == "r2":
                self._load_r2_model()
            else:
                self._load_hf_model()
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _load_r2_model(self):
        """Load model from R2 storage."""
        if self.model_config["r2_framework"] == "llama_cpp":
            self._load_llama_cpp_from_r2()
        else:
            raise ValueError(f"Unsupported R2 framework: {self.model_config['r2_framework']}")

    def _load_llama_cpp_from_r2(self):
        """Load GGUF model from R2 using llama-cpp-python."""
        import boto3
        from llama_cpp import Llama

        logger.info("Downloading GGUF model from R2...")
        self.model_path = f"/tmp/{self.model_config['r2_file']}"

        # Download model from R2
        s3 = boto3.client(
            's3',
            endpoint_url=f"https://{os.environ['CLOUDFLARE_ACCOUNT_ID']}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
            region_name='auto'
        )

        bucket_name = os.environ.get('R2_BUCKET_NAME', 'masala-embed-models')

        try:
            s3.head_object(Bucket=bucket_name, Key=self.model_config['r2_file'])
            logger.info("GGUF file found in R2, downloading...")
        except Exception as e:
            logger.error(f"GGUF file not found in R2: {e}")
            raise FileNotFoundError(f"{self.model_config['r2_file']} not found in R2")

        s3.download_file(bucket_name, self.model_config['r2_file'], self.model_path)

        # Verify download
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Failed to download model to {self.model_path}")

        file_size = os.path.getsize(self.model_path)
        logger.info(f"Downloaded GGUF model: {file_size / (1024*1024):.1f} MB")

        if file_size < 1024 * 1024:  # Less than 1MB
            raise ValueError(f"Downloaded model seems corrupted (size: {file_size} bytes)")

        # Validate GGUF format
        with open(self.model_path, 'rb') as f:
            header = f.read(8)
            if not header.startswith(b'GGUF'):
                raise ValueError(f"File is not a valid GGUF format (header: {header})")

        logger.info("Loading GGUF model with llama.cpp...")
        self.model = Llama(
            model_path=self.model_path,
            embedding=True,
            n_ctx=self.model_config['max_length'],
            n_batch=128,
            n_threads=CPU_COUNT,
            verbose=False,
            use_mlock=False,
            use_mmap=True
        )

    def _load_hf_model(self):
        """Load model from HuggingFace."""
        if self.model_config["hf_framework"] == "transformers":
            self._load_transformers_model()
        elif self.model_config["hf_framework"] == "sentence_transformers":
            self._load_sentence_transformers_model()
        else:
            raise ValueError(f"Unsupported HF framework: {self.model_config['hf_framework']}")

    def _load_transformers_model(self):
        """Load model using transformers library."""
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        model_id = self.model_config["hf_model_id"]
        logger.info(f"Loading transformers model: {model_id}")

        try:
            self.model = AutoModel.from_pretrained(model_id)
            if self.model is not None and hasattr(self.model, 'eval'):
                self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Try to load processor first, fall back to tokenizer
        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
            logger.info("Loaded AutoProcessor")
        except Exception:
            try:
                self.processor = AutoTokenizer.from_pretrained(model_id)
                logger.info("Loaded AutoTokenizer")
            except Exception as e:
                logger.warning(f"Could not load processor or tokenizer: {e}")

    def _load_sentence_transformers_model(self):
        """Load model using sentence-transformers library."""
        from sentence_transformers import SentenceTransformer

        model_id = self.model_config["hf_model_id"]
        logger.info(f"Loading sentence-transformers model: {model_id}")

        self.model = SentenceTransformer(model_id)

    @modal.method()
    def embed(self, text: Optional[str] = None, image_url: Optional[str] = None) -> Dict[str, Any]:
        """Generate embeddings based on model type."""
        if self.model is None:
            return {"error": "Model not loaded"}

        try:
            if self.model_config["model_type"] == "text_embedding":
                if not text:
                    return {"error": "Text required for text embedding model"}
                return self._embed_text(text)

            elif self.model_config["model_type"] == "image_embedding":
                if not image_url:
                    return {"error": "Image URL required for image embedding model"}
                return self._embed_image(image_url)

            elif self.model_config["model_type"] == "multimodal":
                return self._embed_multimodal(text, image_url)

            else:
                return {"error": f"Unsupported model type: {self.model_config['model_type']}"}

        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return {"error": f"Embedding failed: {str(e)}"}

    def _embed_text(self, text: str) -> Dict[str, Any]:
        """Generate text embeddings."""
        if not text or not text.strip():
            return {"error": "Empty text provided"}

        text = text[:self.model_config['max_length']]  # Truncate if too long

        if self.model_config["model_source"] == "r2":
            # llama.cpp text embedding
            from llama_cpp import Llama
            if isinstance(self.model, Llama):
                result = self.model.create_embedding(text)
                return {
                    "embedding": result['data'][0]['embedding'],
                    "model": self.model_config["model_name"]
                }

        elif self.model_config["hf_framework"] == "sentence_transformers":
            # sentence-transformers
            from sentence_transformers import SentenceTransformer
            if isinstance(self.model, SentenceTransformer):
                embedding = self.model.encode(text).tolist()
                return {
                    "embedding": embedding,
                    "model": self.model_config["model_name"]
                }

        elif self.model_config["hf_framework"] == "transformers":
            # transformers with tokenizer
            import torch

            if self.processor is not None and (hasattr(self.processor, 'tokenize') or hasattr(self.processor, '__call__')):
                inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)

                with torch.no_grad():
                    if self.model is not None:
                        if hasattr(self.model, 'get_text_features'):
                            embeddings = self.model.get_text_features(**inputs)
                        else:
                            outputs = self.model(**inputs)
                            # Try different output attributes
                            if hasattr(outputs, 'last_hidden_state'):
                                embeddings = outputs.last_hidden_state.mean(dim=1)
                            elif hasattr(outputs, 'pooler_output'):
                                embeddings = outputs.pooler_output
                            else:
                                embeddings = outputs[0].mean(dim=1)

                        return {
                            "embedding": embeddings[0].tolist(),
                            "model": self.model_config["model_name"]
                        }

        return {"error": "Text embedding not supported for this model configuration"}

    def _embed_image(self, image_url: str) -> Dict[str, Any]:
        """Generate image embeddings."""
        if self.model_config["hf_framework"] != "transformers":
            return {"error": "Image embedding only supported with transformers framework"}

        import torch
        import requests
        from PIL import Image

        try:
            # Load image
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            image = Image.open(response.content)

            # Process image
            if self.processor is not None:
                inputs = self.processor(images=image, return_tensors="pt")
            else:
                return {"error": "Processor not loaded"}

                with torch.no_grad():
                    if self.model is not None:
                        if hasattr(self.model, 'get_image_features'):
                            embeddings = self.model.get_image_features(**inputs)
                        else:
                            outputs = self.model(**inputs)
                            embeddings = outputs.last_hidden_state.mean(dim=1) if hasattr(outputs, 'last_hidden_state') else outputs[0].mean(dim=1)

                        return {
                            "embedding": embeddings[0].tolist(),
                            "model": self.model_config["model_name"]
                        }

        except Exception as e:
            return {"error": f"Image processing failed: {str(e)}"}

        return {"error": "Image embedding failed"}

    def _embed_multimodal(self, text: Optional[str], image_url: Optional[str]) -> Dict[str, Any]:
        """Generate multimodal embeddings."""
        if self.model_config["hf_framework"] != "transformers":
            return {"error": "Multimodal embedding only supported with transformers framework"}

        if not text and not image_url:
            return {"error": "Either text or image_url required"}

        if self.processor is None or self.model is None:
            return {"error": "Model or processor not loaded"}

        try:
            inputs = self._prepare_multimodal_inputs(text, image_url)
            return self._process_multimodal_embeddings(inputs, text, image_url)
        except Exception as e:
            return {"error": f"Multimodal processing failed: {str(e)}"}

    def _prepare_multimodal_inputs(self, text: Optional[str], image_url: Optional[str]) -> Dict[str, Any]:
        """Prepare inputs for multimodal processing."""
        import requests
        from PIL import Image

        if text and image_url:
            # Combined processing
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            image = Image.open(response.content)
            if self.processor is not None:
                return self.processor(text=[text], images=image, return_tensors="pt", padding=True, truncation=True)

        inputs = {}
        if text and self.processor is not None:
            text_inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            inputs.update(text_inputs)

        if image_url and self.processor is not None:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            image = Image.open(response.content)
            image_inputs = self.processor(images=image, return_tensors="pt")
            inputs.update(image_inputs)

        return inputs

    def _process_multimodal_embeddings(self, inputs: Dict[str, Any], text: Optional[str], image_url: Optional[str]) -> Dict[str, Any]:
        """Process embeddings from multimodal inputs."""
        import torch

        with torch.no_grad():
            if (text and image_url and self.model is not None and
                hasattr(self.model, 'get_text_features') and hasattr(self.model, 'get_image_features')):
                text_embeds = self.model.get_text_features(
                    input_ids=inputs.get('input_ids'),
                    attention_mask=inputs.get('attention_mask')
                )
                image_embeds = self.model.get_image_features(pixel_values=inputs.get('pixel_values'))
                return {
                    "text_embedding": text_embeds[0].tolist(),
                    "image_embedding": image_embeds[0].tolist(),
                    "model": self.model_config["model_name"]
                }
            else:
                # Single modality
                if self.model is not None and hasattr(self.model, 'get_text_features') and text:
                    embeddings = self.model.get_text_features(**inputs)
                elif self.model is not None and hasattr(self.model, 'get_image_features') and image_url:
                    embeddings = self.model.get_image_features(**inputs)
                elif self.model is not None:
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1) if hasattr(outputs, 'last_hidden_state') else outputs[0].mean(dim=1)
                else:
                    raise ValueError("Model not loaded")

                return {
                    "embedding": embeddings[0].tolist(),
                    "model": self.model_config["model_name"]
                }

# Initialize model instance
unified_model = UnifiedModel()

@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def embed(data: Dict[str, Any]) -> Dict[str, Any]:
    """FastAPI endpoint for embeddings."""
    return unified_model.embed.remote(
        text=data.get("text"),
        image_url=data.get("image_url")
    )

@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "model": MODEL_NAME}
