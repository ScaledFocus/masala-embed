import io
import logging
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MODEL = "openai/clip-vit-base-patch32"
MODEL = "google/siglip2-base-patch16-224"


class DishRequest(BaseModel):
    text: str | None = None
    image: HttpUrl | None = None


class DishResponse(BaseModel):
    dish: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    BASE_DIR = Path(__file__).resolve().parent
    INDEX_PATH = BASE_DIR / "setup" / "dish_index.faiss"
    CSV_PATH = BASE_DIR / "setup" / "dish_index.csv"
    app.state.faiss_index = faiss.read_index(str(INDEX_PATH))
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        app.state.dishes = [row.strip() for row in f]
    app.state.model = LLM(
        model=MODEL,
        task="embed",
        tokenizer_mode="auto",
        gpu_memory_utilization=0.5,
    )
    yield


app = FastAPI(default_response_class=ORJSONResponse, lifespan=lifespan)


def _embed_text(model: LLM, text: str) -> np.ndarray:
    out = model.embed([text])
    embed = np.asarray(out[0].outputs.embedding, dtype="float32")
    embed = embed / np.linalg.norm(embed)
    return embed


def _embed_image(model: LLM, url: str) -> np.ndarray:
    try:
        r = httpx.get(url, follow_redirects=True, timeout=30)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        out = model.embed([img])
        embed = np.asarray(out[0].outputs.embedding, dtype="float32")
        embed = embed / np.linalg.norm(embed)
        return embed
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {e}")


@app.post("/v1/dish")
def get_dish(req: DishRequest) -> DishResponse:
    # Validate input
    if not req.text and not req.image:
        raise HTTPException(
            status_code=400, detail="Must provide at least text or image"
        )

    try:
        parts = []
        if req.text:
            parts.append(_embed_text(app.state.model, req.text))
        if req.image:
            parts.append(_embed_image(app.state.model, str(req.image)))

        if len(parts) == 1:
            query = parts[0]
        else:
            query = (parts[0] + parts[1]) / 2.0
            query = query / np.linalg.norm(query)

        query = query[None, :]

        D, I = app.state.faiss_index.search(query, k=5)  # noqa: E741

        # Validate index
        idx = int(I[0][0])
        if idx < 0 or idx >= len(app.state.dishes):
            raise HTTPException(status_code=500, detail="Invalid search result index")

        dish = app.state.dishes[idx]

        logger.info(f"Query: text={req.text}, image={req.image}")
        logger.info(f"Top result: {dish} (score={D[0][0]:.4f})")

        return DishResponse(dish=dish)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
