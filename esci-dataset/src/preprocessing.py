import ast
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import pandas as pd
import semhash
from dotenv import load_dotenv
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def combine_text_columns(row: pd.Series, columns: list[str]) -> str:
    """Combine specified columns into a structured text field."""
    if not columns:
        return ""
    
    # First column should be dish_name
    dish_name = ""
    if columns[0] in row:
        dish_name_value = row[columns[0]]
        dish_name = str(dish_name_value).strip() if pd.notna(dish_name_value) else ""
    
    # Process remaining columns (typically ingredients)
    ingredients_parts = []
    for col in columns[1:]:
        if col not in row:
            logger.warning(f"Column '{col}' not found in row")
            continue

        value = row[col]

        # Handle list-like strings (e.g., ingredients stored as string representation of list)
        if isinstance(value, str) and value.startswith("["):
            try:
                parsed_value = ast.literal_eval(value)
                if isinstance(parsed_value, list):
                    # Join list items with commas for ingredients
                    value = ", ".join(str(item).strip() for item in parsed_value if str(item).strip())
                else:
                    value = str(parsed_value).strip()
            except Exception as e:
                logger.warning(f"Error parsing list-like string in column '{col}': {e}")
                value = str(value).strip()
        elif isinstance(value, list):
            # Join list items with commas for ingredients
            value = ", ".join(str(item).strip() for item in value if str(item).strip())
        else:
            value = str(value).strip() if pd.notna(value) else ""

        if value:
            ingredients_parts.append(value)

    # Create structured format: dish_name\nIngredients:\ningredient_list
    if not dish_name:
        dish_name = "Unknown Dish"
    
    if ingredients_parts:
        ingredients_text = ", ".join(ingredients_parts)
        combined_text = f"{dish_name}\n{ingredients_text}"
    else:
        combined_text = f"{dish_name}\nNone specified"
    return combined_text.lower().strip()


def create_records_from_dataframe(
    df: pd.DataFrame, text_column: str = "combined_text"
) -> list[dict[str, Any]]:
    """Convert dataframe to records format for SemHash."""
    records = []
    for idx, row in df.iterrows():
        record = {"id": idx, text_column: row[text_column], "original_index": idx}
        # Add all original columns
        for col in df.columns:
            if col != text_column:
                record[col] = row[col]

        records.append(record)

    logger.debug(f"Created {len(records)} records from dataframe")
    return records


def deduplicate_records(
    records: list[dict[str, Any]],
    threshold: float = 0.9,
    text_columns: list[str] = ["combined_text"],
    reuse_model: semhash.SemHash = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicate records using SemHash."""
    logger.debug(
        f"Starting deduplication with threshold {threshold} on columns {text_columns}"
    )
    try:
        if reuse_model is not None:
            # Use existing model but with new records
            sh = semhash.SemHash.from_records(
                records=records, columns=text_columns, use_ann=True, model=reuse_model.model
            )
        else:
            sh = semhash.SemHash.from_records(
                records=records, columns=text_columns, use_ann=True
            )
    except Exception as e:
        logger.error(f"Error initializing SemHash: {e}")
        logger.error(f"Records sample for debugging: {records}")
        raise e
    dedup_result = sh.self_deduplicate(threshold=threshold)

    logger.debug(f"Original: {len(records)}")
    logger.debug(f"After dedup: {len(dedup_result.selected)}")
    logger.debug(
        f"Removed: {len(dedup_result.filtered)} ({len(dedup_result.filtered) / len(records) * 100:.1f}%)"
    )

    return dedup_result.selected, dedup_result.filtered

def remove_null_rows(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Remove rows with null or empty values in specified columns."""
    initial_count = len(df)
    for col in columns:
        if col in df.columns:
            df = df[df[col].notna() & (df[col].astype(str).str.strip() != "")]
        else:
            logger.warning(f"Column '{col}' not found in dataframe")
    final_count = len(df)
    logger.info(f"Removed {initial_count - final_count} rows with null/empty values in columns {columns}")
    return df


def prepare_consumable_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare DataFrame for insertion into consumable table.
    Maps MM-Food-100K columns to consumable table schema.
    
    Args:
        df: Source DataFrame with food data
        
    Returns:
        DataFrame ready for database insertion
    """
    # Create a copy to avoid modifying original
    df_prep = df.copy()
    
    # Map columns to database schema (no ID needed - auto-increment)
    column_mapping = {
        "image_url": "image_url",
        "dish_name": "consumable_name", 
        "food_type": "consumable_type",
        "ingredients": "consumable_ingredients",
        "portion_size": "consumable_portion_size",
        "nutritional_profile": "consumable_nutritional_profile",
        "cooking_method": "consumable_cooking_method",
    }
    
    # Check for missing columns
    missing_cols = [col for col in column_mapping.keys() if col not in df_prep.columns]
    if missing_cols:
        logger.warning(f"Missing columns in source data: {missing_cols}")
    
    # Select and rename columns that exist
    available_cols = {k: v for k, v in column_mapping.items() if k in df_prep.columns}
    df_prep = df_prep[available_cols.keys()].rename(columns=available_cols)
    
    # Clean and validate data
    df_prep = _clean_consumable_data(df_prep)
    
    logger.info(f"Prepared {len(df_prep)} records for consumable table insertion")
    return df_prep


def _clean_consumable_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate data for database insertion."""
    
    # Remove rows with missing required fields
    required_fields = ["image_url", "consumable_name"]
    initial_count = len(df)
    df = df.dropna(subset=required_fields)
    dropped_count = initial_count - len(df)
    
    if dropped_count > 0:
        logger.warning(f"Dropped {dropped_count} rows with missing required fields")
    
    # Handle JSON fields - convert string representation to actual JSON
    if "consumable_nutritional_profile" in df.columns:
        df["consumable_nutritional_profile"] = df["consumable_nutritional_profile"].apply(_parse_json_field)
    
    # Clean text fields
    text_fields = ["consumable_name", "consumable_type", "consumable_cooking_method"]
    for field in text_fields:
        if field in df.columns:
            df[field] = df[field].astype(str).str.strip()
            df[field] = df[field].replace("nan", None)
    
    return df


def _parse_json_field(value) -> dict[str, Any]:
    """Parse JSON field, handling string representations."""
    if pd.isna(value):
        return None
    
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    
    return value

def deduplicate_dataset(
    input_file: str,
    output_file: str,
    combine_columns: list[str] = ["dish_name", "ingredients"],
    threshold: float = 0.9,
    data_folder: str | None = None,
    combined_text_column: str = "combined_text",
) -> pd.DataFrame:
    """
    Complete deduplication pipeline for any dataset.

    Args:
        input_file: Name of input CSV file
        output_file: Name of output CSV file
        combine_columns: List of column names to combine for similarity comparison (default: ['dish_name', 'ingredients'])
        threshold: Similarity threshold for deduplication (0.9 default)
        data_folder: Data folder path (uses env var if None)
        combined_text_column: Name for the combined text column

    Returns:
        Deduplicated dataframe
    """
    load_dotenv()

    if data_folder is None:
        data_folder = os.environ.get("data_folder")
        if not data_folder:
            raise ValueError(
                "data_folder not provided and not found in environment variables"
            )

    logger.info(f"Loading data from {data_folder}")

    # Load data
    input_path = os.path.join(data_folder, input_file)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} records from {input_file}")

    # Validate columns exist
    missing_cols = [col for col in combine_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataset: {missing_cols}")

    # Remove rows with null/empty dish_name
    df = remove_null_rows(df, ["dish_name"])

    logger.info(f"Combining columns: {combine_columns}")

    # Create combined text field
    df[combined_text_column] = df.apply(
        lambda row: combine_text_columns(row, combine_columns), axis=1
    )

    # Convert to records
    records = create_records_from_dataframe(df, combined_text_column)

    # Deduplicate
    selected_records, filtered_records = deduplicate_records(
        records, threshold, [combined_text_column]
    )

    # Create deduplicated dataframe
    kept_indices = [record["original_index"] for record in selected_records]
    df_deduplicated = df.iloc[kept_indices].copy()
    # Create filtered dataframe from filtered records
    # Note: filtered_records are DuplicateRecord objects with record attribute
    # debug log 1 filtered record
    logger.debug(f"Example filtered record: {filtered_records[0]}")  # type: ignore
    if filtered_records:
        filtered_indices = [record.record["original_index"] for record in filtered_records]
        df_filtered = df.iloc[filtered_indices].copy()
    else:
        df_filtered = pd.DataFrame()  # Empty dataframe if no filtered records
    logger.info(f"Deduplicated dataset has {len(df_deduplicated)} records")
    logger.info(f"Filtered dataset has {len(df_filtered)} records")

    # Save result
    output_path = os.path.join(data_folder, output_file)
    df_deduplicated.to_csv(output_path, index=False)
    logger.info(
        f"Saved deduplicated dataset: {len(df_deduplicated)} records to {output_file}"
    )

    return df_deduplicated, df_filtered


def hybrid_deduplicate_dataset(
    input_file: str,
    output_file: str,
    combine_columns: list[str] = ["dish_name", "ingredients"],
    threshold: float = 0.9,
    data_folder: str | None = None,
    combined_text_column: str = "combined_text",
) -> pd.DataFrame:
    """
    Hybrid deduplication: only deduplicate semantically within same dish names.
    Preserves dish name diversity while removing semantic duplicates within each dish type.
    
    Args:
        input_file: Name of input CSV file
        output_file: Name of output CSV file
        combine_columns: List of column names to combine for similarity comparison (default: ['dish_name', 'ingredients'])
        threshold: Similarity threshold for deduplication (0.9 default)
        data_folder: Data folder path (uses env var if None)
        combined_text_column: Name for the combined text column
    
    Returns:
        Deduplicated dataframe
    """
    load_dotenv()
    
    if data_folder is None:
        data_folder = os.environ.get("data_folder")
        if not data_folder:
            raise ValueError(
                "data_folder not provided and not found in environment variables"
            )
    
    logger.info(f"Loading data from {data_folder}")
    
    # Load data
    input_path = os.path.join(data_folder, input_file)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} records from {input_file}")
    
    # Validate columns exist
    missing_cols = [col for col in combine_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataset: {missing_cols}")
    
    # Remove rows with null/empty dish_name
    df = remove_null_rows(df, ["dish_name"])
    
    logger.info(f"Combining columns: {combine_columns}")
    
    # Create combined text field
    df[combined_text_column] = df.apply(
        lambda row: combine_text_columns(row, combine_columns), axis=1
    )
    
    # Group by dish_name and deduplicate within each group
    all_deduplicated = []
    dish_names = df['dish_name'].unique()
    
    logger.info(f"Processing {len(dish_names)} unique dish names")
    
    # Initialize SemHash model once to avoid reloading for each dish group
    logger.info("Initializing SemHash model...")
    sample_records = create_records_from_dataframe(df.head(100), combined_text_column)
    global_sh = semhash.SemHash.from_records(
        records=sample_records, columns=[combined_text_column], use_ann=True
    )
    logger.info("SemHash model initialized successfully")
    
    for dish_name in tqdm(dish_names, desc="Deduplicating by dish_name"):
        dish_group = df[df['dish_name'] == dish_name].copy()
        
        if len(dish_group) == 1:
            # Single item, keep as is
            all_deduplicated.append(dish_group)
            continue
        
        # Convert to records for this dish group
        records = create_records_from_dataframe(dish_group, combined_text_column)
        
        # Deduplicate within this dish group using the pre-initialized model
        selected_records, _ = deduplicate_records(
            records, threshold, [combined_text_column], reuse_model=global_sh
        )
        
        # Create deduplicated dataframe for this dish
        # Map original indices back to relative positions in the dish group
        dish_group_reset = dish_group.reset_index(drop=True)
        original_to_relative = {orig_idx: rel_idx for rel_idx, orig_idx in enumerate(dish_group.index)}
        kept_relative_indices = [original_to_relative[record["original_index"]] for record in selected_records]
        dish_deduplicated = dish_group_reset.iloc[kept_relative_indices].copy()
        all_deduplicated.append(dish_deduplicated)
        
        logger.debug(f"{dish_name}: {len(dish_group)} -> {len(dish_deduplicated)} records")
    # Combine all deduplicated groups
    df_final = pd.concat(all_deduplicated, ignore_index=True)
    
    # Save result
    output_path = os.path.join(data_folder, output_file)
    df_final.to_csv(output_path, index=False)
    logger.info(
        f"Hybrid deduplication complete: {len(df_final)} records saved to {output_file}"
    )
    
    return df_final


def _process_dish_group(args):
    """Worker function for parallel processing of dish groups."""
    dish_name, dish_group_data, combined_text_column, threshold = args
    
    # Recreate dish_group DataFrame from serialized data
    dish_group = pd.DataFrame(dish_group_data)
    
    if len(dish_group) == 1:
        return dish_group
    
    # Convert to records for this dish group
    records = create_records_from_dataframe(dish_group, combined_text_column)
    
    # Create a new SemHash model for this process (can't share across processes)
    sh = semhash.SemHash.from_records(
        records=records, columns=[combined_text_column], use_ann=True
    )
    
    # Deduplicate within this dish group
    selected_records, _ = deduplicate_records(
        records, threshold, [combined_text_column], reuse_model=sh
    )
    
    # Create deduplicated dataframe for this dish
    dish_group_reset = dish_group.reset_index(drop=True)
    original_to_relative = {orig_idx: rel_idx for rel_idx, orig_idx in enumerate(dish_group.index)}
    kept_relative_indices = [original_to_relative[record["original_index"]] for record in selected_records]
    dish_deduplicated = dish_group_reset.iloc[kept_relative_indices].copy()
    
    logger.debug(f"{dish_name}: {len(dish_group)} -> {len(dish_deduplicated)} records")
    return dish_deduplicated


def hybrid_deduplicate_dataset_parallel(
    input_file: str,
    output_file: str,
    combine_columns: list[str] = ["dish_name", "ingredients"],
    threshold: float = 0.9,
    data_folder: str | None = None,
    combined_text_column: str = "combined_text",
    max_workers: int | None = None,
) -> pd.DataFrame:
    """
    Parallel hybrid deduplication: only deduplicate semantically within same dish names.
    Uses multiprocessing to parallelize dish group processing.
    
    Args:
        input_file: Name of input CSV file
        output_file: Name of output CSV file
        combine_columns: List of column names to combine for similarity comparison
        threshold: Similarity threshold for deduplication
        data_folder: Data folder path (uses env var if None)
        combined_text_column: Name for the combined text column
        max_workers: Maximum number of worker processes (defaults to CPU count)
    
    Returns:
        Deduplicated dataframe
    """
    load_dotenv()
    
    if data_folder is None:
        data_folder = os.environ.get("data_folder")
        if not data_folder:
            raise ValueError(
                "data_folder not provided and not found in environment variables"
            )
    
    logger.info(f"Loading data from {data_folder}")
    
    # Load data
    input_path = os.path.join(data_folder, input_file)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} records from {input_file}")
    
    # Validate columns exist
    missing_cols = [col for col in combine_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataset: {missing_cols}")
    
    # Remove rows with null/empty dish_name
    df = remove_null_rows(df, ["dish_name"])
    
    logger.info(f"Combining columns: {combine_columns}")
    
    # Create combined text field
    df[combined_text_column] = df.apply(
        lambda row: combine_text_columns(row, combine_columns), axis=1
    )
    
    # Group by dish_name
    dish_names = df['dish_name'].unique()
    logger.info(f"Processing {len(dish_names)} unique dish names with parallel processing")
    
    # Prepare arguments for parallel processing
    process_args = []
    for dish_name in dish_names:
        dish_group = df[df['dish_name'] == dish_name].copy()
        # Convert to dict for serialization (multiprocessing requirement)
        dish_group_data = dish_group.to_dict('records')
        process_args.append((dish_name, dish_group_data, combined_text_column, threshold))
    
    # Process in parallel
    all_deduplicated = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_dish = {
            executor.submit(_process_dish_group, args): args[0] 
            for args in process_args
        }
        
        # Collect results with progress bar
        for future in tqdm(as_completed(future_to_dish), total=len(future_to_dish), desc="Deduplicating by dish_name"):
            dish_name = future_to_dish[future]
            try:
                result = future.result()
                all_deduplicated.append(result)
            except Exception as exc:
                logger.error(f"Dish {dish_name} generated an exception: {exc}")
                raise exc
    
    # Combine all deduplicated groups
    df_final = pd.concat(all_deduplicated, ignore_index=True)
    
    # Save result
    output_path = os.path.join(data_folder, output_file)
    df_final.to_csv(output_path, index=False)
    logger.info(
        f"Parallel hybrid deduplication complete: {len(df_final)} records saved to {output_file}"
    )
    
    return df_final


if __name__ == "__main__":
    # Example usage - uses default columns ['dish_name', 'ingredients']
    df_result = deduplicate_dataset(
        input_file="MM-Food-100K.csv",
        output_file="MM-Food-100K-deduplicated.csv",
        threshold=0.9,
    )

    # Example usage for custom columns
    # df_result = deduplicate_dataset(
    #     input_file="your_dataset.csv",
    #     output_file="your_dataset_deduplicated.csv",
    #     combine_columns=['title', 'description', 'tags'],
    #     threshold=0.85
    # )
