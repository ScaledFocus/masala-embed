from pathlib import Path

import modal

APP_NAME = "masala-embed-create-faiss-upload"
VOLUME_NAME = "masala-embed-setup"

LOCAL_SETUP_DIR = (Path(__file__).parent / "setup").resolve()

INDEX_FILENAME = "dish_index.faiss"

REMOTE_DIR = "/setup"
REMOTE_IDX = f"{REMOTE_DIR}/{INDEX_FILENAME}"

app = modal.App(APP_NAME)


@app.local_entrypoint()
def main():
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    src_idx = LOCAL_SETUP_DIR / INDEX_FILENAME

    assert src_idx.exists(), f"Missing {src_idx}"

    with vol.batch_upload() as batch:
        batch.put_file(str(src_idx), REMOTE_IDX)

    print(f"Uploaded to volume '{VOLUME_NAME}':")
    print(f"- {REMOTE_IDX}")
