import os
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from tqdm import tqdm


def download_images(input_csv, output_csv, image_dir="images"):
    df = pd.read_csv(input_csv)

    os.makedirs(image_dir, exist_ok=True)

    unique_urls = df["image"].unique()
    url_to_local = {}

    print(f"Downloading {len(unique_urls)} unique images...")

    for idx, url in enumerate(tqdm(unique_urls)):
        ext = Path(urlparse(url).path).suffix or ".jpg"
        local_filename = f"image_{idx:08d}{ext}"
        local_path = os.path.join(image_dir, local_filename)

        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        url_to_local[url] = local_path

    df["image"] = df["image"].map(url_to_local)

    df.to_csv(output_csv, index=False)

    print(f"Downloaded {len(unique_urls)} images to {image_dir}/")
    print(f"Updated CSV saved to {output_csv}")


if __name__ == "__main__":
    download_images("train.csv", "train.csv", image_dir="images")
