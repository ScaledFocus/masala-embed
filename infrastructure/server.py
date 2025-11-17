import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import faiss
import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from PIL import Image
from pydantic import BaseModel, HttpUrl
from vllm import LLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL = "openai/clip-vit-base-patch32"
# MODEL = "google/siglip-base-patch16-224"


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
    return np.asarray(out[0].outputs.embedding, dtype="float32")


def _embed_image(model: LLM, url: str) -> np.ndarray:
    r = httpx.get(url, follow_redirects=True)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    out = model.embed([img])
    return np.asarray(out[0].outputs.embedding, dtype="float32")


@app.post("/v1/dish")
def get_dish(req: DishRequest) -> DishResponse:
    parts = []
    if req.text:
        parts.append(_embed_text(app.state.model, req.text))
    if req.image:
        parts.append(_embed_image(app.state.model, str(req.image)))

    query = parts[0] if len(parts) == 1 else (parts[0] + parts[1]) / 2.0
    query = query[None, :]
    faiss.normalize_L2(query)

    D, I = app.state.faiss_index.search(query, k=5)  # noqa: E741
    logger.info(f"Query: text={req.text}, image={req.image}")
    logger.info(f"Top 5 indices: {I[0]}")
    logger.info(f"Top 5 scores: {D[0]}")
    for i in range(min(5, len(I[0]))):
        idx = int(I[0][i])
        score = D[0][i]
        logger.info(
            f"  Rank {i + 1}: idx={idx}, score={score:.4f}, dish={app.state.dishes[idx]}"
        )

    dish = app.state.dishes[int(I[0][0])]
    return DishResponse(dish=dish)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
