import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import faiss
import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse
from PIL import Image
from pydantic import BaseModel, HttpUrl
from vllm import LLM

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

MODEL = os.getenv("MODEL", "google/siglip2-base-patch16-224")
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.8"))
TOP_K = int(os.getenv("TOP_K", "5"))


class DishRequest(BaseModel):
    text: str | None = None
    image: HttpUrl | None = None


class DishResponse(BaseModel):
    dishes: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    BASE_DIR = Path(__file__).resolve().parent
    TEXT_INDEX_PATH = BASE_DIR / "setup" / "dish_index_text.faiss"
    IMAGE_INDEX_PATH = BASE_DIR / "setup" / "dish_index_image.faiss"
    CSV_PATH = BASE_DIR / "setup" / "dish_index.csv"

    # Load both FAISS indices (IndexFlatIP - no HNSW tuning needed)
    app.state.text_index = faiss.read_index(str(TEXT_INDEX_PATH))
    app.state.image_index = faiss.read_index(str(IMAGE_INDEX_PATH))

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        app.state.dishes = [row.strip() for row in f]

    # Reusable HTTP client with connection pooling
    app.state.http = httpx.Client(
        http2=True,
        timeout=httpx.Timeout(connect=2.0, read=3.0),
        limits=httpx.Limits(max_connections=200, max_keepalive_connections=100),
    )

    # vLLM model
    app.state.model = LLM(
        model=MODEL,
        task="embed",
        tokenizer_mode="auto",
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )
    yield
    app.state.http.close()


app = FastAPI(default_response_class=ORJSONResponse, lifespan=lifespan)


def _embed_text(model: LLM, text: str) -> np.ndarray:
    """Embed text and normalize with FAISS."""
    out = model.embed([text])
    arr = np.asarray(out[0].outputs.embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(arr)
    return arr


def _embed_image(http: httpx.Client, model: LLM, url: str) -> np.ndarray:
    """Embed image from URL and normalize with FAISS."""
    try:
        r = http.get(url, follow_redirects=True)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        out = model.embed([img])
        arr = np.asarray(out[0].outputs.embedding, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(arr)
        return arr
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {e}")


@app.post("/v1/dish")
def get_dish(req: DishRequest) -> DishResponse:
    """Search for dishes using text and/or image, return top 5."""
    if not req.text and not req.image:
        raise HTTPException(
            status_code=400, detail="Must provide at least text or image"
        )

    try:
        dishes = app.state.dishes

        # Text-only: search text index (already sorted by FAISS)
        if req.text and not req.image:
            text_vec = _embed_text(app.state.model, req.text)
            D, I = app.state.text_index.search(text_vec, k=TOP_K)  # noqa: E741
            return DishResponse(
                dishes=[dishes[int(idx)] for idx in I[0] if 0 <= idx < len(dishes)][:TOP_K]
            )

        # Image-only: search image index (already sorted by FAISS)
        if req.image and not req.text:
            image_vec = _embed_image(app.state.http, app.state.model, str(req.image))
            D, I = app.state.image_index.search(image_vec, k=TOP_K)  # noqa: E741
            return DishResponse(
                dishes=[dishes[int(idx)] for idx in I[0] if 0 <= idx < len(dishes)][:TOP_K]
            )

        # Both modalities: search both indices and combine scores
        combined_scores: dict[str, float] = {}

        text_vec = _embed_text(app.state.model, req.text)
        D, I = app.state.text_index.search(text_vec, k=TOP_K)  # noqa: E741
        for score, idx in zip(D[0], I[0]):
            if 0 <= idx < len(dishes):
                combined_scores[dishes[int(idx)]] = combined_scores.get(
                    dishes[int(idx)], 0.0
                ) + float(score)

        image_vec = _embed_image(app.state.http, app.state.model, str(req.image))
        D, I = app.state.image_index.search(image_vec, k=TOP_K)  # noqa: E741
        for score, idx in zip(D[0], I[0]):
            if 0 <= idx < len(dishes):
                combined_scores[dishes[int(idx)]] = combined_scores.get(
                    dishes[int(idx)], 0.0
                ) + float(score)

        if not combined_scores:
            raise HTTPException(status_code=500, detail="No valid search results")

        # Sort combined scores and return top 5
        top_dishes = [
            dish
            for dish, _ in sorted(
                combined_scores.items(), key=lambda x: x[1], reverse=True
            )[:TOP_K]
        ]
        return DishResponse(dishes=top_dishes)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error processing request", exc_info=e)
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        loop="uvloop",
        http="httptools",
    )
