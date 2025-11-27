import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, get_linear_schedule_with_warmup

# Configuration from environment variables
MODEL_NAME = os.getenv("MODEL_NAME", "google/siglip2-base-patch16-224")
TRAIN_CSV = os.getenv("TRAIN_CSV", "train.csv")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "finetuned_siglip")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "5e-6"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "10"))
WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", "0.1"))
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


class ImageTextDataset(Dataset):
    def __init__(self, csv_path, processor):
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        print(f"Loaded {len(self.df)} samples from {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Handle flexible modalities
        has_text = pd.notna(row["text"]) and str(row["text"]).strip()
        has_image = pd.notna(row["image"]) and str(row["image"]).strip()

        if not has_text and not has_image:
            raise ValueError(f"Row {idx} has neither text nor image")

        if has_text and has_image:
            # Both modalities
            image = Image.open(row["image"]).convert("RGB")
            text = str(row["text"]).strip()
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt",
                truncation=True,
            )
        elif has_text:
            # Text only
            text = str(row["text"]).strip()
            inputs = self.processor(
                text=text,
                return_tensors="pt",
                truncation=True,
            )
        else:
            # Image only
            image = Image.open(row["image"]).convert("RGB")
            inputs = self.processor(
                images=image,
                return_tensors="pt",
            )

        return {k: v.squeeze(0) for k, v in inputs.items()}


def collate_fn(batch, processor):
    # Separate items by modality
    texts = []
    images = []
    has_text_list = []
    has_image_list = []

    for item in batch:
        has_text = "input_ids" in item
        has_image = "pixel_values" in item
        has_text_list.append(has_text)
        has_image_list.append(has_image)

        if has_text:
            texts.append(
                processor.tokenizer.decode(item["input_ids"], skip_special_tokens=True)
            )
        if has_image:
            images.append(item["pixel_values"])

    # Process based on what modalities are present
    if texts and images and all(has_text_list) and all(has_image_list):
        # All items have both modalities
        inputs = processor(
            text=texts,
            images=torch.stack(images),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
    elif texts and not images:
        # All items have text only
        inputs = processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
    elif images and not texts:
        # All items have images only
        inputs = processor(
            images=torch.stack(images),
            return_tensors="pt",
        )
    else:
        raise ValueError("Batch contains mixed modalities.")

    return inputs


def sigmoid_loss(image_embeds, text_embeds):
    """SigLIP loss function using sigmoid instead of softmax."""
    # Normalize embeddings
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    # Compute similarity matrix
    logits = torch.matmul(image_embeds, text_embeds.t())

    batch_size = logits.shape[0]
    labels = torch.eye(batch_size, device=logits.device)

    # Sigmoid loss
    loss = (
        -torch.sum(
            labels * torch.nn.functional.logsigmoid(logits)
            + (1 - labels) * torch.nn.functional.logsigmoid(-logits)
        )
        / batch_size**2
    )

    return loss


def train():
    print("Configuration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Train CSV: {TRAIN_CSV}")
    print(f"  Output Dir: {OUTPUT_DIR}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Device: {DEVICE}")
    print("-" * 60)

    device = torch.device(DEVICE)

    # Load model and processor
    processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)

    # Create dataset
    train_dataset = ImageTextDataset(TRAIN_CSV, processor)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with images
        collate_fn=lambda batch: collate_fn(batch, processor),
    )

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    print(f"Training steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")
    print("-" * 60)

    # Training loop
    model.train()

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            loss = sigmoid_loss(image_embeds, text_embeds)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f}")

    # Save final model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
