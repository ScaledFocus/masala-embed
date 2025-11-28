from pathlib import Path

import modal

APP_NAME = "masala-embed-upload-embeddings"
VOLUME_NAME = "masala-embed-setup"

# Local directory containing artifacts to upload
LOCAL_SETUP_DIR = (Path(__file__).parent / "setup").resolve()

CSV_FILENAME = "dish_index.csv"

REMOTE_DIR = "/setup"

app = modal.App(APP_NAME)


@app.local_entrypoint()
def main():
    # Prepare volume
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    # Resolve local paths
    src_csv = LOCAL_SETUP_DIR / CSV_FILENAME

    # Required CSV must exist
    assert src_csv.exists(), f"Missing required CSV: {src_csv}"

    # Build upload list
    uploads: list[tuple[Path, str]] = [
        (src_csv, f"{REMOTE_DIR}/{CSV_FILENAME}"),
    ]

    # Perform batch upload
    with vol.batch_upload() as batch:
        for local_path, remote_path in uploads:
            batch.put_file(str(local_path), remote_path)

    print(f"Uploaded to volume '{VOLUME_NAME}':")
    for _, remote_path in uploads:
        print(f"- {remote_path}")
