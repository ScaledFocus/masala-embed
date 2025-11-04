from contextlib import asynccontextmanager
from pathlib import Path

import faiss
import numpy as np
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, HttpUrl
from vllm import LLM

MODEL = "google/siglip-base-patch16-224"
# MODEL = "Qwen/Qwen3-Embedding-0.6B"


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
        app.state.dishes = [row for row in f]
    app.state.model = LLM(model=MODEL, task="embed")
    yield


app = FastAPI(default_response_class=ORJSONResponse, lifespan=lifespan)


@app.post("/v1/dish")
async def get_dish(request: DishRequest) -> DishResponse:
    output = app.state.model.embed(request.text)
    embeds = output[0].outputs.embedding
    query = np.array(embeds, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(query)
    D, I = app.state.faiss_index.search(query, k=1)
    idx = int(I[0][0])
    dish = app.state.dishes[idx]
    return DishResponse(dish=dish)
