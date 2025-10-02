#!/usr/bin/env python3
"""
Initial data generation script for creating realistic food delivery queries.

This script generates realistic food delivery queries using OpenAI's GPT model,
dynamic prompt templating, and DSPy for structured I/O.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Add the project root to Python path for imports
project_root = os.environ.get("root_folder")
print(f"Project root from env: {project_root}")
if project_root:
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, os.path.join(project_root, "esci-dataset"))

from database.utils.db_utils import get_table  # noqa: E402

from src.data_generation.dspy_schemas import (  # noqa: E402
    QueryGenerationOutput,
    QueryGenerator,
    convert_output_to_dataframe,
    setup_dspy_model,
)
from src.data_generation.prompt_template import (  # noqa: E402
    get_esci_label_description,
    prepare_prompt,
)
from src.evals.dietary_evals import apply_complete_dietary_evaluation  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("query_generation.log")],
)
logger = logging.getLogger(__name__)


def setup_argparser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate realistic food delivery queries using OpenAI and DSPy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load 1000 records, process 10 at a time with dietary info
  python src/data_generation/initial_generation.py --esci_label E \\
    --dietary_flag --limit 1000 --batch_size 10

  # Load entire table, process 20 at a time
  python src/data_generation/initial_generation.py --esci_label S --batch_size 20

  # Load 50 records, process 5 at a time with custom output
  python src/data_generation/initial_generation.py --esci_label C \\
    --limit 50 --batch_size 5 --output_path output/queries.json
        """,
    )

    parser.add_argument(
        "--esci_label",
        choices=["E", "S", "C", "I"],
        required=True,
        help="ESCI label filter (E=Exact, S=Substitute, C=Complement, I=Irrelevant)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of records to load from database (default: load all)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Number of records to process per OpenAI API call (default: 10)",
    )

    parser.add_argument(
        "--dietary_flag",
        action="store_true",
        help="Include dietary columns in the output",
    )

    parser.add_argument(
        "--query_examples",
        type=str,
        default=None,
        help="Path to query examples file (optional, if not provided no "
        "examples will be included)",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Where to save generated queries (default: auto-generated filename)",
    )

    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=None,
        help="OpenAI API key (default: use OPENAI_API_KEY env var)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="OpenAI model to use (default: gpt-5-mini)",
    )

    parser.add_argument(
        "--template_path",
        type=str,
        default=None,
        help="Path to prompt template (default: prompts/query_generation/v1.txt),",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.2,
        help="Temperature for LLM generation (default: 1.2)",
    )

    parser.add_argument(
        "--queries_per_item",
        type=int,
        default=5,
        help="Number of queries to generate per food item (default: 5)",
    )

    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Starting index for resuming interrupted jobs (default: 0)",
    )

    # Output format is always CSV in modernized version

    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum retries for failed generations (default: 3)",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive if provided")

    if args.temperature < 0 or args.temperature > 2:
        raise ValueError("temperature must be between 0 and 2")

    if args.max_retries < 1:
        raise ValueError("max_retries must be at least 1")

    if args.queries_per_item < 1:
        raise ValueError("queries_per_item must be at least 1")

    if args.start_idx < 0:
        raise ValueError("start_idx must be non-negative")


def get_api_key(args: argparse.Namespace) -> str:
    """Get OpenAI API key from args or environment."""
    api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key not provided. Use --openai_api_key or set "
            "OPENAI_API_KEY environment variable"
        )
    return api_key


def get_template_path(args: argparse.Namespace) -> str:
    """Get template path, using default if not provided."""
    if args.template_path:
        if project_root:
            full_path = os.path.join(project_root, args.template_path)
        else:
            raise ValueError(
                "Project root not set. Cannot determine full template path."
            )
        return full_path

    # Default template path relative to project root
    if project_root:
        default_path = os.path.join(
            project_root, "prompts", "query_generation", "v1.txt"
        )
    else:
        raise ValueError(
            "Project root not set. Cannot determine default template path."
        )
    return default_path


def generate_output_filename(args: argparse.Namespace) -> str:
    """Generate output filename based on parameters."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dietary_suffix = "_dietary" if args.dietary_flag else ""
    limit_suffix = f"_limit{args.limit}" if args.limit else ""
    start_suffix = f"_start{args.start_idx}" if args.start_idx > 0 else ""

    # Extract prompt version from template path
    prompt_version = "v1"  # default
    if args.template_path:
        template_name = os.path.basename(args.template_path)
        if template_name.startswith("v") and ".txt" in template_name:
            prompt_version = template_name.replace(".txt", "")

    # Always use CSV format in modernized version
    filename = (
        f"queries_{args.esci_label}_batch{args.batch_size}{limit_suffix}"
        f"{start_suffix}{dietary_suffix}_{prompt_version}_{timestamp}.csv"
    )

    # Create output directory if it doesn't exist
    if project_root:
        output_dir = os.path.join(project_root, "output")
    else:
        output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    return os.path.join(output_dir, filename)


def load_and_process_data(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    """Load consumable data and apply dietary evaluation if requested."""
    logger.info("Loading consumable data from database...")

    try:
        # Load ENTIRE table first for true randomization
        df = get_table("consumable", limit=None)
        logger.info(f"Loaded {len(df)} total records from consumable table")

        if len(df) == 0:
            raise ValueError("No data found in consumable table")

        # Shuffle the ENTIRE dataframe with fixed seed for reproducibility
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info("Shuffled entire dataset with seed=42")

        # Apply start_idx and limit AFTER shuffling
        if args.start_idx > 0:
            df = df.iloc[args.start_idx :]
            logger.info(
                f"Resumed from index {args.start_idx}, {len(df)} records remaining"
            )

        if args.limit is not None:
            df = df.head(args.limit)
            logger.info(f"Selected top {len(df)} records after shuffling and start_idx")

        # Always apply dietary evaluation for enhanced JSON structure
        logger.info("Applying dietary evaluation...")
        df, dietary_columns = apply_complete_dietary_evaluation(df)
        logger.info(f"Added dietary columns: {dietary_columns}")

        logger.info(
            f"Ready to process {len(df)} records in batches of {args.batch_size}"
        )

        return df, dietary_columns

    except Exception as e:
        logger.error(f"Error loading/processing data: {e}")
        raise


def generate_queries_with_retry(
    generator: QueryGenerator,
    prompt: str,
    esci_label: str,
    max_retries: int,
    batch_num: int = None,
) -> QueryGenerationOutput:
    """Generate queries with retry logic using structured output."""
    batch_info = f" (batch {batch_num})" if batch_num is not None else ""
    for attempt in range(max_retries):
        try:
            logger.info(
                f"Generating queries{batch_info} "
                f"(attempt {attempt + 1}/{max_retries})..."
            )
            result = generator(prompt, esci_label)
            logger.info(
                f"Query generation successful{batch_info}: "
                f"{len(result.candidates)} candidates"
            )
            return result
        except Exception as e:
            logger.warning(f"Generation attempt {attempt + 1} failed{batch_info}: {e}")
            if attempt == max_retries - 1:
                logger.error(f"All {max_retries} attempts failed{batch_info}")
                raise


def process_in_batches(
    df: pd.DataFrame,
    args: argparse.Namespace,
    generator: QueryGenerator,
    template_path: str,
    query_examples_path: str,
    dietary_columns: list[str] = None,
) -> dict:
    """Process dataframe in batches and combine results."""
    all_candidates = []
    total_batches = (
        len(df) + args.batch_size - 1
    ) // args.batch_size  # Ceiling division

    logger.info(
        f"Processing {len(df)} records in {total_batches} batches of "
        f"size {args.batch_size}"
    )

    for batch_idx in range(total_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx].copy()

        logger.info(
            f"Processing batch {batch_idx + 1}/{total_batches} "
            f"({len(batch_df)} records)"
        )

        # Prepare prompt for this batch
        prompt = prepare_prompt(
            template_path=template_path,
            df=batch_df,
            esci_label=args.esci_label,
            batch_size=len(batch_df),  # Use actual batch size for prompt
            include_dietary=args.dietary_flag,
            queries_per_item=args.queries_per_item,
            query_examples_path=query_examples_path,
            dietary_columns=dietary_columns,
        )

        logger.info(f"Batch {batch_idx + 1} prompt length: {len(prompt)} characters")

        # Generate queries for this batch
        try:
            structured_output = generate_queries_with_retry(
                generator, prompt, args.esci_label, args.max_retries, batch_idx + 1
            )

            # Extract candidates from structured output
            batch_candidates = structured_output.model_dump().get("candidates", [])
            all_candidates.extend(batch_candidates)

            # Log structured response information
            total_queries = sum(
                len(candidate.get("queries", [])) for candidate in batch_candidates
            )
            logger.info(
                f"Batch {batch_idx + 1} completed: {len(batch_candidates)} "
                f"candidates, {total_queries} queries generated"
            )

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {e}")
            # Continue with next batch instead of failing completely
            continue

    return {"candidates": all_candidates}


def save_results_as_csv(
    output_data: dict,
    output_path: str,
    args: argparse.Namespace,
    original_df: pd.DataFrame = None,
    dietary_columns: list[str] = None,
) -> None:
    """Save generation results to CSV file using structured approach."""
    # Add metadata
    output_data["metadata"] = {
        "generated_at": datetime.now().isoformat(),
        "esci_label": args.esci_label,
        "esci_description": get_esci_label_description(args.esci_label),
        "limit": args.limit,
        "batch_size": args.batch_size,
        "dietary_flag": args.dietary_flag,
        "model": args.model,
        "temperature": args.temperature,
        "total_candidates": len(output_data.get("candidates", [])),
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Always save as CSV using structured approach
    save_as_csv(output_data, output_path, original_df, dietary_columns)
    logger.info(f"Results saved to CSV: {output_path}")


def save_as_csv(
    output_data: dict,
    output_path: str,
    original_df: pd.DataFrame = None,
    dietary_columns: list[str] = None,
) -> None:
    """Save results in CSV format with one query per row using structured approach."""
    candidates = output_data.get("candidates", [])
    metadata = output_data.get("metadata", {})

    if not candidates:
        logger.warning("No candidates found in output_data for CSV export")
        # Create empty DataFrame with expected columns
        df = pd.DataFrame(
            columns=[
                "consumable_id",
                "consumable_name",
                "query",
                "query_filters",
                "esci_label",
                "generated_at",
            ]
        )
    else:
        # Create QueryGenerationOutput from dict data
        try:
            structured_output = QueryGenerationOutput(candidates=candidates)
            # Use the new DataFrame conversion function
            df = convert_output_to_dataframe(structured_output)

            # Create enhanced JSON structure with dietary restrictions
            df = create_enhanced_query_filters(df, original_df, dietary_columns)

            # Keep only essential columns - remove all dim_* columns
            essential_columns = [
                "consumable_id",
                "consumable_name",
                "query",
                "query_filters",
            ]
            # Keep only columns that exist in the DataFrame
            final_columns = [col for col in essential_columns if col in df.columns]
            df = df[final_columns].copy()

            # Add metadata columns
            df["esci_label"] = metadata.get("esci_label", "")
            df["generated_at"] = metadata.get("generated_at", "")

        except Exception as e:
            logger.error(f"Failed to convert output to DataFrame: {e}")
            raise e

    # Save to CSV
    df.to_csv(output_path, index=False, encoding="utf-8")


def create_enhanced_query_filters(
    df: pd.DataFrame,
    original_df: pd.DataFrame = None,
    dietary_columns: list[str] = None,
) -> pd.DataFrame:
    """
    Create enhanced query_filters JSON structure using existing dietary columns.

    Args:
        df: DataFrame from convert_output_to_dataframe()
        original_df: Original DataFrame with dietary columns

    Returns:
        DataFrame with query_filters column containing enhanced JSON structure
    """
    if original_df is None:
        # Fallback - just use dimensions without dietary restrictions
        df["query_filters"] = df["dimensions_json"]
        logger.warning("Original DataFrame not provided, skipping dietary enrichment")
        return df
    if dietary_columns:
        # Create a DataFrame with only id and dietary columns
        dietary_df = original_df[["id"] + dietary_columns].copy()
        # Melt to long format, filter True, and aggregate to lists
        melted = dietary_df.melt(
            id_vars="id", value_vars=dietary_columns, var_name="col", value_name="val"
        )
        filtered = melted[melted["val"]].copy()
        # Humanize column names
        filtered["col"] = (
            filtered["col"]
            .str.replace("is_", "", regex=False)
            .str.replace("_", " ", regex=False)
            .str.title()
            .str.replace("Gluten Free", "Gluten-Free")
            .str.replace("Dairy Free", "Dairy-Free")
            .str.replace("Nut Free", "Nut-Free")
        )
        dietary_lookup = filtered.groupby("id")["col"].apply(list).to_dict()
    else:
        dietary_lookup = {}

    # Function to combine LLM and rule-based dietary restrictions
    def build_query_filters(row):
        candidate_id = row["consumable_id"]
        # Parse LLM dimensions
        try:
            llm_dimensions = (
                json.loads(row["dimensions_json"]) if row["dimensions_json"] else {}
            )
        except Exception as e:
            logger.warning(
                f"Failed to parse dimensions_json for id {candidate_id}: {e}"
            )
            llm_dimensions = {}
        # Get rule-based dietary
        rule_based_dietary = dietary_lookup.get(candidate_id, [])
        return json.dumps(
            {
                "dimensions": llm_dimensions,
                "rule_based_dietary_restrictions": rule_based_dietary,
            }
        )

    df["query_filters"] = df.apply(build_query_filters, axis=1)
    return df


def main():
    """Main function."""
    try:
        # Parse arguments
        parser = setup_argparser()
        args = parser.parse_args()
        validate_args(args)

        logger.info("Starting query generation process...")
        logger.info(
            f"Configuration: ESCI={args.esci_label}, limit={args.limit}, "
            f"batch_size={args.batch_size}, dietary_flag={args.dietary_flag}, "
            f"model={args.model}"
        )

        # Get API key and setup DSPy
        api_key = get_api_key(args)
        setup_dspy_model(api_key, args.model, args.temperature)
        generator = QueryGenerator()

        # Load and process data
        df, dietary_columns = load_and_process_data(args)

        # Get template path
        template_path = get_template_path(args)
        query_examples_path = (
            os.path.join(project_root, args.query_examples)
            if args.query_examples and project_root
            else args.query_examples
        )
        logger.info(f"Using template: {template_path}")

        # Process data in batches
        try:
            output_dict = process_in_batches(
                df, args, generator, template_path, query_examples_path, dietary_columns
            )
            logger.info(
                f"Successfully generated queries for "
                f"{len(output_dict['candidates'])} candidates"
            )
        except Exception as e:
            logger.error(f"Failed to process batches: {e}")
            logger.info("Saving error information for debugging...")
            output_dict = {"candidates": [], "error": str(e)}

        # Determine output path
        output_path = args.output_path or generate_output_filename(args)

        # Save results as CSV with original DataFrame for dietary data
        save_results_as_csv(output_dict, output_path, args, df, dietary_columns)

        logger.info("Query generation completed successfully!")

        # Print summary
        if "candidates" in output_dict:
            total_queries = sum(
                len(candidate.get("queries", []))
                for candidate in output_dict["candidates"]
            )
            print(
                f"\n✅ Success! Generated {total_queries} queries for "
                f"{len(output_dict['candidates'])} candidates"
            )
            print(f"📄 CSV output saved to: {output_path}")
            print(
                f"🏷️  ESCI Label: {args.esci_label} "
                f"({get_esci_label_description(args.esci_label)})"
            )
            print("📊 Format: CSV with one query per row")
        else:
            print(
                f"\n⚠️ Generation completed with errors. Check {output_path} "
                f"for details."
            )

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
