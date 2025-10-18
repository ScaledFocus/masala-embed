import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
# This has one column with no header containing dish names extracted from the dataset.
CSV_FILE = "setup/dish_name.csv"
EMB_FILE = "setup/dish_embeddings.npy"
DISH_FILE = "setup/dish_index.csv"

dishes = pd.read_csv(CSV_FILE, header=None, names=["dish"])
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).cuda()
model.eval()

with torch.no_grad():
    inputs = tokenizer(
        dishes["dish"].tolist(), padding=True, truncation=True, return_tensors="pt"
    ).to("cuda")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    embeddings_np = embeddings.cpu().numpy().astype("float32")

np.save(EMB_FILE, embeddings_np)
dishes.to_csv(DISH_FILE, index=False)
