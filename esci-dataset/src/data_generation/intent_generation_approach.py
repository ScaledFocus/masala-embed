#!/usr/bin/env python3
"""
Intent-driven query generation approach.

This script implements a 3-step process to generate realistic food delivery queries:
1. Generate pure user intents (no food bias)
2. Smart 1:1 matching of foods to best-fitting intents
3. Generate both intent queries and food-aware final queries

Key insight: Original intents work as standalone queries ~75% of the time!
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import dspy
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

# Add the project root to Python path for imports
project_root = os.environ.get("root_folder")
print(f"Project root from env: {project_root}")
if project_root:
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, os.path.join(project_root, "esci-dataset"))

# Import after sys.path modifications
from database.utils.db_utils import get_table  # noqa: E402

from src.data_generation.dspy_schemas import setup_dspy_model  # noqa: E402
from src.evals.dietary_evals import apply_complete_dietary_evaluation  # noqa: E402

# Custom Pydantic Models for Intent Generation Approach


class IntentList(BaseModel):
    """Schema for Step 1: List of pure user intents."""

    intents: list[str] = Field(description="List of pure user search intents")


class IntentMatch(BaseModel):
    """Schema for a single intent-to-food match."""

    consumable_id: int = Field(description="Food item ID")
    consumable_name: str = Field(description="Food item name")
    intent: str = Field(description="Matched user intent")
    reasoning: str = Field(description="Reasoning for the match")


class IntentMatches(BaseModel):
    """Schema for Step 2: Collection of intent-to-food matches."""

    matches: list[IntentMatch] = Field(description="List of intent-food matches")


class IntentQueryResult(BaseModel):
    """Schema for a single query result with metadata."""

    consumable_id: int = Field(description="Food item ID")
    consumable_name: str = Field(description="Food item name")
    original_intent: str = Field(description="Original user intent")
    queries: list[str] = Field(description="Generated queries for this food")


class IntentQueryOutput(BaseModel):
    """Schema for Step 3: Complete query generation output."""

    query_results: list[IntentQueryResult] = Field(
        description="List of query results per food"
    )


# DSPy Signatures and Modules


class IntentGenerationSignature(dspy.Signature):
    """DSPy signature for generating pure user intents."""

    prompt: str = dspy.InputField(desc="Intent generation prompt")
    intent_list: IntentList = dspy.OutputField(desc="List of pure user search intents")


class IntentMatchingSignature(dspy.Signature):
    """DSPy signature for matching intents to foods."""

    matching_prompt: str = dspy.InputField(
        desc="Prompt with intents and foods to match"
    )
    intent_matches: IntentMatches = dspy.OutputField(
        desc="Intent-to-food matches with reasoning"
    )


class IntentQuerySignature(dspy.Signature):
    """DSPy signature for generating final queries from matches."""

    query_prompt: str = dspy.InputField(
        desc="Prompt for generating queries from matches"
    )
    query_output: IntentQueryOutput = dspy.OutputField(
        desc="Generated queries for all matched foods"
    )


class IntentGenerator(dspy.Module):
    """DSPy module for Step 1: Generate pure user intents."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(IntentGenerationSignature)

    def forward(self, prompt: str) -> IntentList:
        result = self.generate(prompt=prompt)
        return result.intent_list


class IntentMatcher(dspy.Module):
    """DSPy module for Step 2: Match intents to foods."""

    def __init__(self):
        super().__init__()
        self.match = dspy.ChainOfThought(IntentMatchingSignature)

    def forward(self, matching_prompt: str) -> IntentMatches:
        result = self.match(matching_prompt=matching_prompt)
        return result.intent_matches


class IntentQueryGenerator(dspy.Module):
    """DSPy module for Step 3: Generate final queries."""

    def __init__(self):
        super().__init__()
        self.generate_queries = dspy.ChainOfThought(IntentQuerySignature)

    def forward(self, query_prompt: str) -> IntentQueryOutput:
        result = self.generate_queries(query_prompt=query_prompt)
        return result.query_output


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("intent_query_generation.log"),
    ],
)
logger = logging.getLogger(__name__)


def setup_dspy_client(model: str = "gpt-5", temperature: float = 1.0):
    """Setup DSPy with OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    setup_dspy_model(api_key, model, temperature)
    logger.info(f"DSPy setup complete with model: {model}, temperature: {temperature}")


def step1_generate_intents(num_intents: int = 50) -> list[str]:
    """Step 1: Generate pure user intents with no food bias using DSPy."""
    logger.info(f"Step 1: Generating {num_intents} pure user intents")

    # Load v1.1 intent generation prompt
    prompt_path = "prompts/intent_generation/v1.1_intent_generation.txt"
    prompt_path = os.path.join(project_root, prompt_path)
    try:
        with open(prompt_path, encoding="utf-8") as f:
            intent_prompt = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Intent generation prompt not found at: {prompt_path}")

    # Replace the default number with the requested number
    intent_prompt = intent_prompt.replace(
        "Generate 50 diverse", f"Generate {num_intents} diverse"
    )
    intent_prompt = intent_prompt.replace(
        "Return a simple list of 50 search queries",
        f"Return a simple list of {num_intents} search queries",
    )

    # Generate intents using DSPy
    try:
        generator = IntentGenerator()
        result = generator.forward(intent_prompt)
        intents = result.intents

        logger.info(f"Generated {len(intents)} user intents")
        return intents

    except Exception as e:
        logger.error(f"Error in step1_generate_intents: {e}")
        raise


def load_food_data(limit: int | None = None, dietary_flag: bool = False) -> pd.DataFrame:
    """Load food candidates data from database."""
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

        # Apply limit AFTER shuffling to get truly random subset
        if limit is not None:
            df = df.head(limit)
            logger.info(f"Selected top {len(df)} records after shuffling")

        # Map database column names to expected format
        if 'id' in df.columns and 'consumable_id' not in df.columns:
            df = df.rename(columns={'id': 'consumable_id'})
            logger.info("Mapped 'id' column to 'consumable_id'")

        # Apply dietary evaluation if requested
        if dietary_flag:
            logger.info("Applying dietary evaluation...")
            df, dietary_columns = apply_complete_dietary_evaluation(df)
            logger.info(f"Added dietary columns: {dietary_columns}")

        # Keep original column names (consumable_id, consumable_name)

        logger.info(f"Successfully processed {len(df)} food items")
        return df

    except Exception as e:
        logger.error(f"Error loading food data: {e}")
        raise


def step2_match_intents_to_foods(intents: list[str], food_df: pd.DataFrame, dietary_flag: bool = False) -> dict:
    """Step 2: Smart 1:1 matching of foods to best-fitting intents using DSPy."""
    logger.info(f"Step 2: Matching {len(food_df)} foods to {len(intents)} intents")
    logger.info(f"Food dataframe columns: {food_df.columns.tolist()}")

    # Load v1.2 intent matching prompt
    prompt_path = "prompts/intent_generation/v1.2_intent_matching.txt"
    prompt_path = os.path.join(project_root, prompt_path)
    try:
        with open(prompt_path, encoding="utf-8") as f:
            matching_prompt_template = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Intent matching prompt not found at: {prompt_path}")

    # Format the prompt with actual data
    intents_list = chr(10).join(
        [f"{i + 1}. {intent}" for i, intent in enumerate(intents)]
    )

    # Include only relevant columns for matching (include dietary if flag is set)
    if dietary_flag:
        # Include dietary columns in the dataframe for matching
        display_df = food_df.copy()
    else:
        # Only include basic columns for matching
        display_df = food_df[['consumable_id', 'consumable_name']].copy()

    food_dataframe = display_df.to_markdown(index=False)

    matching_prompt = matching_prompt_template.format(
        intents_list=intents_list, food_dataframe=food_dataframe
    )

    # Get matches using DSPy
    try:
        matcher = IntentMatcher()
        result = matcher.forward(matching_prompt)

        # Convert Pydantic result to dict format for backward compatibility
        matches = {"matches": [match.dict() for match in result.matches]}

        logger.info(f"Successfully matched {len(matches['matches'])} foods to intents")
        return matches

    except Exception as e:
        logger.error(f"Error in step2_match_intents_to_foods: {e}")
        raise


def step3_generate_final_queries(
    matches: dict, food_df: pd.DataFrame, queries_per_item: int = 3, dietary_flag: bool = False
) -> list[dict]:
    """Step 3: Generate final queries for all matched pairs using DSPy."""
    logger.info(
        f"Step 3: Generating {queries_per_item} queries for "
        f"{len(matches['matches'])} matched pairs"
    )

    # Load v1.3 intent query generation prompt
    prompt_path = "prompts/intent_generation/v1.3_intent_query_generation.txt"
    prompt_path = os.path.join(project_root, prompt_path)
    try:
        with open(prompt_path, encoding="utf-8") as f:
            batch_prompt_template = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Intent query generation prompt not found at: {prompt_path}"
        )

    # Format intent-food pairs
    intent_food_pairs = ""
    for i, match in enumerate(matches["matches"], 1):
        food_info = f'{match["consumable_name"]} (ID: {match["consumable_id"]})'

        # Add dietary information if flag is enabled
        if dietary_flag:
            food_row = food_df[food_df['consumable_id'] == match["consumable_id"]]
            if not food_row.empty:
                dietary_cols = [col for col in food_df.columns if col.startswith(('is_', 'contains_', 'dietary_'))]
                dietary_info = []
                for col in dietary_cols:
                    if food_row[col].iloc[0]:  # Only show True values
                        dietary_info.append(col.replace('_', ' ').title())
                if dietary_info:
                    food_info += f" [{', '.join(dietary_info)}]"

        intent_food_pairs += f"""
{i}. Intent: "{match["intent"]}"
   Food: {food_info}
"""

    # Format the prompt with actual data
    batch_prompt = batch_prompt_template.format(
        intent_food_pairs=intent_food_pairs, queries_per_item=queries_per_item
    )

    # Generate queries using DSPy
    try:
        query_generator = IntentQueryGenerator()
        result = query_generator.forward(batch_prompt)

        # Debug: Check the result structure
        logger.info(f"Result type: {type(result)}")
        logger.info(f"Result has query_results: {hasattr(result, 'query_results')}")
        if hasattr(result, 'query_results'):
            logger.info(f"Number of query_results: {len(result.query_results)}")
            if result.query_results:
                logger.info(f"First query_result type: {type(result.query_results[0])}")
                logger.info(f"First query_result: {result.query_results[0]}")

        # Convert to flat list format
        final_queries = []
        for query_result in result.query_results:
            try:
                # Base query record
                base_record = {
                    "consumable_id": query_result.consumable_id,
                    "consumable_name": query_result.consumable_name,
                    "original_intent": query_result.original_intent,
                    "generated_at": datetime.now().isoformat(),
                }
            except AttributeError as e:
                logger.error(f"Error accessing query_result fields: {e}")
                logger.error(f"query_result type: {type(query_result)}")
                logger.error(f"query_result content: {query_result}")
                raise

            # Add dietary columns if flag is enabled
            if dietary_flag:
                food_row = food_df[food_df['consumable_id'] == query_result.consumable_id]
                if not food_row.empty:
                    dietary_cols = [col for col in food_df.columns if col.startswith(('is_', 'contains_', 'dietary_'))]
                    for col in dietary_cols:
                        base_record[col] = food_row[col].iloc[0]

            # Add intent as standalone query
            intent_record = base_record.copy()
            intent_record.update({
                "query": query_result.original_intent,
                "query_type": "intent",
            })
            final_queries.append(intent_record)

            # Add bridged queries
            for query in query_result.queries:
                bridged_record = base_record.copy()
                bridged_record.update({
                    "query": query,
                    "query_type": "bridged",
                })
                final_queries.append(bridged_record)

        intent_count = len([q for q in final_queries if q["query_type"] == "intent"])
        bridged_count = len([q for q in final_queries if q["query_type"] == "bridged"])
        logger.info(
            f"Generated {len(final_queries)} total queries "
            f"({intent_count} intent + {bridged_count} bridged)"
        )

        return final_queries

    except Exception as e:
        logger.error(f"Error in step3_generate_final_queries: {e}")
        raise


def save_results(
    intents: list[str],
    matches: dict,
    final_queries: list[dict],
    food_df: pd.DataFrame,
    output_dir: str = "output",
    stop_at_intents: bool = False,
    dietary_flag: bool = False,
):
    """Save all results to files."""
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save original intents
    intents_path = f"{output_dir}/intent_generation_intents_{timestamp}.txt"
    with open(intents_path, "w", encoding="utf-8") as f:
        f.write("ORIGINAL USER INTENTS GENERATED (Step 1):\n")
        f.write("=" * 50 + "\n")
        for i, intent in enumerate(intents, 1):
            f.write(f"{i:2d}. {intent}\n")

    # Save matches (only if not stopping at intents, since reasoning is saved in CSV)
    matches_path = None
    if not stop_at_intents:
        matches_path = f"{output_dir}/intent_generation_matches_{timestamp}.json"
        with open(matches_path, "w", encoding="utf-8") as f:
            json.dump(matches, f, indent=2)

    # Save final queries
    queries_path = None
    if not stop_at_intents and final_queries:
        # Save full query results (intent + bridged queries)
        queries_path = f"{output_dir}/intent_generation_queries_{timestamp}.csv"
        df = pd.DataFrame(final_queries)
        df.to_csv(queries_path, index=False)
    elif stop_at_intents and matches:
        # Save intent-only queries when stopping at intents
        queries_path = f"{output_dir}/intent_generation_intent_queries_{timestamp}.csv"
        intent_queries = []
        for match in matches["matches"]:
            query_record = {
                "consumable_id": match["consumable_id"],
                "consumable_name": match["consumable_name"],
                "query": match["intent"],
                "query_type": "intent",
                "original_intent": match["intent"],
                "reasoning": match["reasoning"],
                "generated_at": datetime.now().isoformat(),
            }

            # Add dietary columns if flag is enabled
            if dietary_flag:
                # Get food item from dataframe to include dietary info
                food_row = food_df[food_df['consumable_id'] == match["consumable_id"]]
                if not food_row.empty:
                    # Add dietary columns if they exist
                    dietary_cols = [col for col in food_df.columns if col.startswith(('is_', 'contains_', 'dietary_'))]
                    for col in dietary_cols:
                        query_record[col] = food_row[col].iloc[0]

            intent_queries.append(query_record)
        df = pd.DataFrame(intent_queries)
        df.to_csv(queries_path, index=False)

    if stop_at_intents:
        result_msg = f"Results saved - Intents: {intents_path}, Queries: {queries_path}"
    else:
        result_msg = f"Results saved - Intents: {intents_path}, Matches: {matches_path}"
        if queries_path:
            result_msg += f", Queries: {queries_path}"
    logger.info(result_msg)

    return queries_path


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Intent-driven query generation")
    parser.add_argument("--model", default="gpt-5", help="OpenAI model to use")
    parser.add_argument(
        "--num-intents", type=int, default=50, help="Number of intents to generate"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of food items"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of foods to process per batch",
    )
    parser.add_argument(
        "--queries-per-item",
        type=int,
        default=3,
        help="Number of queries to generate per food item",
    )
    parser.add_argument(
        "--stop-at-intents",
        action="store_true",
        help="Stop after step 2 (intent matching), skip query generation",
    )
    parser.add_argument(
        "--dietary_flag",
        action="store_true",
        help="Include dietary columns in the output",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for model generation",
    )
    parser.add_argument("--output-dir", default="output", help="Output directory")

    args = parser.parse_args()

    logger.info(
        f"Starting intent-driven query generation with model: {args.model}, "
        f"intents: {args.num_intents}, temperature: {args.temperature}"
    )

    try:
        # Setup DSPy
        setup_dspy_client(args.model, args.temperature)

        # Step 1: Generate pure user intents
        intents = step1_generate_intents(args.num_intents)

        # Load food data
        food_df = load_food_data(limit=args.limit, dietary_flag=args.dietary_flag)

        # Process foods in batches
        all_final_queries = []
        all_matches = {"matches": []}
        total_batches = (len(food_df) + args.batch_size - 1) // args.batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(food_df))
            batch_df = food_df.iloc[start_idx:end_idx]

            logger.info(
                f"Processing batch {batch_idx + 1}/{total_batches} "
                f"({len(batch_df)} foods)"
            )

            # Step 2: Smart matching for this batch
            batch_matches = step2_match_intents_to_foods(intents, batch_df, args.dietary_flag)

            # Accumulate matches
            all_matches["matches"].extend(batch_matches["matches"])

            # Step 3: Generate final queries for this batch (if not stopping at intents)
            if not args.stop_at_intents:
                batch_queries = step3_generate_final_queries(
                    batch_matches, batch_df, args.queries_per_item, args.dietary_flag
                )
                all_final_queries.extend(batch_queries)

        final_queries = all_final_queries

        # Save results
        save_results(
            intents, all_matches, final_queries, food_df, args.output_dir, args.stop_at_intents, args.dietary_flag
        )

        # Summary
        if args.stop_at_intents:
            logger.info(
                f"Success! Generated {len(intents)} intents matched to "
                f"{len(all_matches['matches'])} foods (stopped at intents)"
            )
        else:
            intent_queries = len(
                [q for q in final_queries if q["query_type"] == "intent"]
            )
            bridged_queries = len(
                [q for q in final_queries if q["query_type"] == "bridged"]
            )
            logger.info(
                f"Success! Generated {len(final_queries)} total queries "
                f"({intent_queries} intent + {bridged_queries} bridged)"
            )

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
