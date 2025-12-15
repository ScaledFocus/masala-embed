import os
import pandas as pd

# Configuration
DATASET_CSV = os.getenv("DATASET_CSV", "dataset.csv")
MM_FOOD_CSV = os.getenv("MM_FOOD_CSV", "MM-Food-100K.csv")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "dataset.csv")


def prepare_dataset(dataset_csv, mm_food_csv, output_csv):
    """
    Merge dataset.csv (with text queries) and MM-Food-100K.csv (with images)
    to create a new dataset.csv with format: text,image,dish_name
    """
    print(f"Reading {dataset_csv}...")
    df_text = pd.read_csv(dataset_csv)
    print(f"  Found {len(df_text)} rows")
    
    print(f"Reading {mm_food_csv}...")
    df_images = pd.read_csv(mm_food_csv)
    print(f"  Found {len(df_images)} rows")
    
    # Rename columns for consistency
    df_text = df_text.rename(columns={
        "Dish Name": "dish_name",
        "Generated Query": "text"
    })
    
    # Create a mapping from dish_name to image_urls
    # Group by dish_name and collect all image URLs
    dish_to_images = df_images.groupby("dish_name")["image_url"].apply(list).to_dict()
    
    print(f"\nFound {len(dish_to_images)} unique dishes with images")
    
    # Prepare the output dataframe
    output_rows = []
    
    # Process each text query row
    for idx, row in df_text.iterrows():
        dish_name = row["dish_name"]
        text = row["text"]
        
        # Try to find matching images for this dish
        images = dish_to_images.get(dish_name, [])
        
        if images:
            # If we have images, create one row per image (or just use first image)
            # For now, we'll create one row per text query, using the first available image
            # This creates multimodal examples
            image_url = images[0]  # Use first image for this dish
            output_rows.append({
                "text": text,
                "image": image_url,
                "dish_name": dish_name
            })
        else:
            # No matching image found - create text-only row
            output_rows.append({
                "text": text,
                "image": "",  # Empty string for text-only
                "dish_name": dish_name
            })
    
    # Also add image-only rows for dishes that don't have text queries
    # (optional - but might be useful for training)
    dishes_with_text = set(df_text["dish_name"].unique())
    for dish_name, images in dish_to_images.items():
        if dish_name not in dishes_with_text:
            # Add image-only rows for dishes without text
            for image_url in images[:1]:  # Just add one image per dish
                output_rows.append({
                    "text": "",  # Empty string for image-only
                    "image": image_url,
                    "dish_name": dish_name
                })
    
    # Create output dataframe
    df_output = pd.DataFrame(output_rows)
    
    # Reorder columns to match expected format: text,image,dish_name
    df_output = df_output[["text", "image", "dish_name"]]
    
    print(f"\nCreated {len(df_output)} rows in output dataset")
    print(f"  Rows with both text and image: {len(df_output[(df_output['text'] != '') & (df_output['image'] != '')])}")
    print(f"  Rows with text only: {len(df_output[(df_output['text'] != '') & (df_output['image'] == '')])}")
    print(f"  Rows with image only: {len(df_output[(df_output['text'] == '') & (df_output['image'] != '')])}")
    
    # Save to CSV
    df_output.to_csv(output_csv, index=False)
    print(f"\nSaved dataset to {output_csv}")
    
    return df_output


if __name__ == "__main__":
    prepare_dataset(DATASET_CSV, MM_FOOD_CSV, OUTPUT_CSV)

