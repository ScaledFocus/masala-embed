import os
import pandas as pd

# Environment variables (default to main dataset)
INPUT_CSV = os.getenv("INPUT_CSV", "dataset.csv")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", None)  # If None, overwrite input


def remove_web_urls(input_csv: str, output_csv: str | None = None) -> None:
    """
    Remove web URLs in the image column (keep only local paths or empty).
    Any value starting with http/https is replaced with an empty string.
    """
    df = pd.read_csv(input_csv)

    def clean_image(val: str) -> str:
        if isinstance(val, str) and val.lower().startswith(("http://", "https://")):
            return ""
        return val if isinstance(val, str) else ""

    df["image"] = df["image"].apply(clean_image)

    out_path = output_csv or input_csv
    df.to_csv(out_path, index=False)
    print(f"Saved cleaned CSV to {out_path}")


if __name__ == "__main__":
    remove_web_urls(INPUT_CSV, OUTPUT_CSV)

