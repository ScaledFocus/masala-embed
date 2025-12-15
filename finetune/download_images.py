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


def normalize_extension(ext):
    """Normalize file extension to .png, .jpg, or .jpeg"""
    ext = ext.lower().strip()
    if ext in ['.png']:
        return '.png'
    elif ext in ['.jpg', '.jpeg']:
        return '.jpg'
    else:
        # Default to .jpg for unknown extensions
        return '.jpg'


def download_images(input_csv, output_csv, image_dir):
    df = pd.read_csv(input_csv)

    os.makedirs(image_dir, exist_ok=True)

    # Filter out empty/NaN image values
    df["image"] = df["image"].fillna("")
    df_with_images = df[df["image"] != ""]
    
    unique_urls = [url for url in df_with_images["image"].unique() if url and str(url).strip() != ""]
    url_to_local = {}
    failed_downloads = []

    print(f"Downloading {len(unique_urls)} unique images to {image_dir}/...")

    for idx, url in enumerate(tqdm(unique_urls)):
        if not url or str(url).strip() == "":
            continue
        
        try:
            # Get and normalize extension
            ext = Path(urlparse(str(url)).path).suffix
            ext = normalize_extension(ext) if ext else ".jpg"
            local_filename = f"image_{idx:08d}{ext}"
            local_path = os.path.join(image_dir, local_filename)

            # Download with timeout handling
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            url_to_local[url] = local_path
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, 
                requests.exceptions.RequestException, Exception) as e:
            # Continue to next image if download fails
            failed_downloads.append(url)
            tqdm.write(f"Failed to download {url}: {type(e).__name__}")
            continue

    # Map URLs to local paths, keeping empty values as empty
    def map_image(x):
        if pd.isna(x) or x == "" or str(x).strip() == "":
            return ""
        return url_to_local.get(x, x)
    
    df["image"] = df["image"].apply(map_image)
    df.to_csv(output_csv, index=False)

    print(f"\nDownloaded {len(url_to_local)} images successfully")
    if failed_downloads:
        print(f"Failed to download {len(failed_downloads)} images")
    print(f"Updated CSV saved to {output_csv}")


if __name__ == "__main__":
    download_images(INPUT_CSV, OUTPUT_CSV, IMAGE_DIR)
