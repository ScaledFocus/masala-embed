import os
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from tqdm import tqdm

# Configuration from environment variables
INPUT_CSV = os.getenv("INPUT_CSV", "dataset.csv")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "dataset.csv")
IMAGE_DIR = os.getenv("IMAGE_DIR", "images")


def download_images(input_csv, output_csv, image_dir):
    df = pd.read_csv(input_csv)

    os.makedirs(image_dir, exist_ok=True)

    unique_urls = df["image"].unique()
    url_to_local = {}

    print(f"Downloading {len(unique_urls)} unique images to {image_dir}/...")

    for idx, url in enumerate(tqdm(unique_urls)):
        ext = Path(urlparse(url).path).suffix or ".jpg"
        local_filename = f"image_{idx:08d}{ext}"
        local_path = os.path.join(image_dir, local_filename)

        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        url_to_local[url] = local_path

    df["image"] = df["image"].map(url_to_local)
    df.to_csv(output_csv, index=False)

    print(f"Downloaded {len(unique_urls)} images")
    print(f"Updated CSV saved to {output_csv}")


if __name__ == "__main__":
    download_images(INPUT_CSV, OUTPUT_CSV, IMAGE_DIR)
