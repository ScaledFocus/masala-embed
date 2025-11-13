import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, get_linear_schedule_with_warmup


class ImageTextDataset(Dataset):
    def __init__(self, csv_path, processor):
        self.df = pd.read_csv(csv_path)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image"]).convert("RGB")
        text = row["query"]

        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        return {k: v.squeeze(0) for k, v in inputs.items()}


def sigmoid_loss(image_embeds, text_embeds):
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    logits = torch.matmul(image_embeds, text_embeds.t())

    batch_size = logits.shape[0]
    labels = torch.eye(batch_size, device=logits.device)

    loss = (
        -torch.sum(
            labels * torch.nn.functional.logsigmoid(logits)
            + (1 - labels) * torch.nn.functional.logsigmoid(-logits)
        )
        / batch_size**2
    )

    return loss


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = "google/siglip-base-patch16-224"
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)

    dataset = ImageTextDataset("train.csv", processor)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    num_epochs = 10
    num_training_steps = num_epochs * len(dataloader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            loss = sigmoid_loss(image_embeds, text_embeds)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

    output_dir = "finetuned_siglip"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    train()
