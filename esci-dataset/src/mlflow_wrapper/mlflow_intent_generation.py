#!/usr/bin/env python3
"""
MLflow wrapper for intent-driven query generation approach.

This script wraps the intent_generation_approach.py script with MLflow tracking,
logging parameters, metrics, and artifacts for experiment management.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import mlflow
from dotenv import load_dotenv

load_dotenv()

# Add the project root to Python path for imports
project_root = os.environ.get("root_folder")
if project_root:
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, os.path.join(project_root, "esci-dataset"))

# Import the original script functions
from src.data_generation.intent_generation_approach import (
    setup_dspy_client,
    step1_generate_intents,
    load_food_data,
    step2_match_intents_to_foods,
    step3_generate_final_queries,
    save_results,
    load_intent_sets,
    get_intent_set_for_batch,
)

from src.utils import get_git_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mlflow_intent_generation.log"),
    ],
)
logger = logging.getLogger(__name__)




def setup_mlflow(experiment_name: str = "intent-generation") -> None:
    """Setup MLflow tracking."""
    # Set tracking URI to local mlruns directory
    mlflow_tracking_uri = os.path.join(project_root, "mlruns") if project_root else "./mlruns"
    mlflow.set_tracking_uri(f"file://{mlflow_tracking_uri}")

    # Set or create experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new MLflow experiment: {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})")

        mlflow.set_experiment(experiment_name)
    except Exception as e:
        logger.error(f"Failed to setup MLflow experiment: {e}")
        raise


def log_parameters(args: argparse.Namespace, prompt_versions: Dict[str, str], intent_set_usage: Dict[str, int] = None) -> None:
    """Log all parameters to MLflow."""
    # Log all CLI arguments
    mlflow.log_param("script_name", "intent_generation_approach.py")
    mlflow.log_param("model", args.model)
    mlflow.log_param("num_intents", args.num_intents)
    mlflow.log_param("limit", args.limit)
    mlflow.log_param("batch_size", args.batch_size)
    mlflow.log_param("queries_per_item", args.queries_per_item)
    mlflow.log_param("stop_at_intents", args.stop_at_intents)
    mlflow.log_param("dietary_flag", args.dietary_flag)
    mlflow.log_param("temperature", args.temperature)
    mlflow.log_param("output_dir", args.output_dir)
    mlflow.log_param("step1_prompt", args.step1_prompt)
    mlflow.log_param("step2_prompt", args.step2_prompt)
    mlflow.log_param("step3_prompt", args.step3_prompt)
    mlflow.log_param("start_idx", args.start_idx)
    mlflow.log_param("use_intent_sets", args.use_intent_sets or "false")
    mlflow.log_param("intent_set_rotation", args.intent_set_rotation)

    # Log derived parameters
    mlflow.log_param("prompt_versions_used", json.dumps(prompt_versions))

    # Log intent set usage if available
    if intent_set_usage:
        mlflow.log_param("intent_sets_used", json.dumps(intent_set_usage))

    # Log generation approach parameters for migration tracking
    mlflow.log_param("generation_approach", "intent")
    if args.stop_at_intents:
        mlflow.log_param("step", "1_2")  # Stops after step 2 (intent matching)
        mlflow.log_param("output_type", "intent_matches")
    else:
        mlflow.log_param("step", "complete")  # Does all three steps
        mlflow.log_param("output_type", "enhanced_json")

    # Log data_gen_hash and MLflow run ID for database migration tracking
    git_info = get_git_info()
    run_id = mlflow.active_run().info.run_id
    mlflow.log_param("data_gen_hash", git_info["commit_hash"])
    mlflow.log_param("mlflow_run_id", run_id)


def log_step_metrics(step: int, execution_time: float, **kwargs) -> None:
    """Log metrics for a specific step."""
    mlflow.log_metric(f"step{step}_time", execution_time)
    for key, value in kwargs.items():
        mlflow.log_metric(key, value)


def log_final_metrics(total_time: float, steps_executed: int, stopped_at_intents: bool,
                     intents_count: int, matches_count: int, queries_count: int,
                     intent_set_usage: Dict[str, int] = None) -> None:
    """Log final summary metrics."""
    mlflow.log_metric("total_runtime_seconds", total_time)
    mlflow.log_metric("steps_executed", steps_executed)
    mlflow.log_metric("stopped_at_intents", 1 if stopped_at_intents else 0)
    mlflow.log_metric("intents_generated", intents_count)
    mlflow.log_metric("successful_matches", matches_count)
    mlflow.log_metric("total_queries_generated", queries_count)

    # Log intent set usage metrics
    if intent_set_usage:
        mlflow.log_metric("intent_sets_used_count", len(intent_set_usage))
        for set_key, usage_count in intent_set_usage.items():
            mlflow.log_metric(f"intent_{set_key}_usage", usage_count)


def log_artifacts(output_paths: Dict[str, str]) -> None:
    """Log output files as MLflow artifacts."""
    for artifact_name, file_path in output_paths.items():
        if file_path and os.path.exists(file_path):
            try:
                mlflow.log_artifact(file_path, f"outputs/{artifact_name}")
                logger.info(f"Logged artifact: {artifact_name} -> {file_path}")
            except Exception as e:
                logger.warning(f"Failed to log artifact {artifact_name}: {e}")


def save_processed_prompts(processed_prompts: Dict[str, str], output_dir: str) -> List[str]:
    """Save processed prompts as separate TXT files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = []

    for step_name, prompt_content in processed_prompts.items():
        if prompt_content:
            filename = f"{step_name}_processed_{timestamp}.txt"
            file_path = os.path.join(output_dir, filename)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(prompt_content)

            saved_files.append(file_path)

    return saved_files


def save_config_snapshot(args: argparse.Namespace, output_dir: str) -> str:
    """Save configuration snapshot as JSON."""
    config = {
        "script_name": "intent_generation_approach.py",
        "parameters": vars(args),
        "execution_time": datetime.now().isoformat(),
        "git_info": get_git_info(),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = os.path.join(output_dir, f"config_snapshot_{timestamp}.json")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return config_path


def generate_run_name(args: argparse.Namespace) -> str:
    """Generate MLflow run name based on parameters."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_suffix = f"-start{args.start_idx}" if args.start_idx > 0 else ""
    return (
        f"intent-{args.model}-{args.num_intents}-"
        f"batch{args.batch_size}-limit{args.limit}{start_suffix}-"
        f"qpi{args.queries_per_item}-{timestamp}"
    )


def setup_argparser() -> argparse.ArgumentParser:
    """Setup command line argument parser with MLflow additions."""
    parser = argparse.ArgumentParser(description="MLflow-wrapped intent-driven query generation")

    # Original script arguments
    parser.add_argument("--model", default="gpt-5-mini", help="OpenAI model to use")
    parser.add_argument("--num-intents", type=int, default=50, help="Number of intents to generate")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of food items")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of foods to process per batch")
    parser.add_argument("--queries-per-item", type=int, default=3, help="Number of queries to generate per food item")
    parser.add_argument("--stop-at-intents", action="store_true", help="Stop after step 2 (intent matching)")
    parser.add_argument("--dietary_flag", action="store_true", help="Include dietary columns in the output")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for model generation")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--step1-prompt", default=None, help="Path to step1 intent generation prompt (relative to project root)")
    parser.add_argument("--step2-prompt", default=None, help="Path to step2 intent matching prompt (relative to project root)")
    parser.add_argument("--step3-prompt", default=None, help="Path to step3 query generation prompt (relative to project root)")
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Starting index for resuming interrupted jobs (default: 0)",
    )
    parser.add_argument(
        "--use-intent-sets",
        default=None,
        help="Path to pre-generated intent sets directory (enables rotation)",
    )
    parser.add_argument(
        "--intent-set-rotation",
        type=int,
        default=1,
        help="Change intent set every N batches (default: 1)",
    )

    # MLflow-specific arguments
    parser.add_argument("--experiment-name", default="intent-generation", help="MLflow experiment name")
    parser.add_argument("--run-name", default=None, help="MLflow run name (auto-generated if not provided)")

    return parser


def main():
    """Main execution with MLflow tracking."""
    parser = setup_argparser()
    args = parser.parse_args()

    # Validate start_idx
    if args.start_idx < 0:
        raise ValueError("start_idx must be non-negative")

    # Validate intent set rotation parameters
    if args.intent_set_rotation < 1:
        raise ValueError("intent_set_rotation must be at least 1")

    if args.use_intent_sets:
        intent_sets_full_path = os.path.join(project_root, args.use_intent_sets)
        if not os.path.exists(intent_sets_full_path):
            raise ValueError(f"Intent sets directory not found: {intent_sets_full_path}")

    # Setup MLflow
    setup_mlflow(args.experiment_name)

    # Generate run name if not provided
    run_name = args.run_name or generate_run_name(args)

    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"Started MLflow run: {run.info.run_id}")

        # Set tags
        git_info = get_git_info()
        mlflow.set_tag("mlflow.source.git.commit", git_info["commit_hash"])
        mlflow.set_tag("mlflow.source.git.branch", git_info["branch_name"])
        mlflow.set_tag("intent-generation", "true")
        mlflow.set_tag("model", args.model)
        mlflow.set_tag(f"{args.num_intents}-intents", "true")
        mlflow.set_tag(f"batch-{args.batch_size}", "true")
        mlflow.set_tag(f"limit-{args.limit}", "true")

        # Add resume/restart tracking
        if args.start_idx > 0:
            mlflow.set_tag("resumed_job", "true")
            mlflow.set_tag(f"start-idx-{args.start_idx}", "true")
        else:
            mlflow.set_tag("resumed_job", "false")

        # Add intent set rotation tracking
        if args.use_intent_sets:
            mlflow.set_tag("uses_rotating_intent_sets", "true")
            mlflow.set_tag(f"rotation_frequency_{args.intent_set_rotation}", "true")
        else:
            mlflow.set_tag("uses_rotating_intent_sets", "false")

        # Set approval tag for migration workflow
        mlflow.set_tag("data_status", "pending_review")

        try:
            total_start_time = time.time()

            # Create output directory
            Path(args.output_dir).mkdir(exist_ok=True)

            # Save config snapshot
            config_path = save_config_snapshot(args, args.output_dir)

            # Setup DSPy
            setup_dspy_client(args.model, args.temperature)

            # Step 1: Generate intents and capture processed prompt
            logger.info("Starting Step 1: Generate intents")
            step1_start = time.time()

            # Intent management: either load pre-generated sets or generate new intents
            if args.use_intent_sets:
                # Load pre-generated intent sets for rotation
                intent_sets_full_path = os.path.join(project_root, args.use_intent_sets)
                intent_sets, intent_set_metadata = load_intent_sets(intent_sets_full_path)
                logger.info(f"Loaded {len(intent_sets)} pre-generated intent sets")
                logger.info(f"Intent set rotation frequency: every {args.intent_set_rotation} batch(es)")
                intents = None  # Will be set per batch
                intent_set_usage = {}  # Track which sets are used
                step1_processed_prompt = "Using pre-generated intent sets (step1 prompt not applicable)"
            else:
                # Generate intents (original behavior) and capture processed prompt
                # Capture processed step1 prompt for logging
                step1_prompt_path = args.step1_prompt or "prompts/intent_generation/v1.1_intent_generation.txt"
                step1_full_path = os.path.join(project_root, step1_prompt_path)
                try:
                    with open(step1_full_path, encoding="utf-8") as f:
                        step1_processed_prompt = f.read()
                    # Apply the same replacements as in step1_generate_intents
                    step1_processed_prompt = step1_processed_prompt.replace(
                        "Generate 50 diverse", f"Generate {args.num_intents} diverse"
                    )
                    step1_processed_prompt = step1_processed_prompt.replace(
                        "Return a simple list of 50 search queries",
                        f"Return a simple list of {args.num_intents} search queries",
                    )
                except FileNotFoundError:
                    step1_processed_prompt = f"Error: Prompt file not found at {step1_full_path}"

                intents = step1_generate_intents(args.num_intents, args.step1_prompt)
                intent_sets = None
                intent_set_metadata = None
                intent_set_usage = None

            step1_time = time.time() - step1_start
            intents_count = len(intents) if intents else 0
            log_step_metrics(1, step1_time, intents_generated=intents_count)

            # Load food data
            food_df, dietary_columns = load_food_data(limit=args.limit, dietary_flag=args.dietary_flag, start_idx=args.start_idx)

            # Process foods in batches
            all_final_queries = []
            all_matches = {"matches": []}
            total_batches = (len(food_df) + args.batch_size - 1) // args.batch_size
            processed_prompts = {"step1": step1_processed_prompt}

            # Step 2: Intent matching
            logger.info("Starting Step 2: Intent matching")
            step2_start = time.time()

            # Capture processed step2 prompt (using first batch for example)
            if total_batches > 0:
                first_batch_df = food_df.iloc[:min(args.batch_size, len(food_df))]
                step2_prompt_path = args.step2_prompt or "prompts/intent_generation/v1.2_intent_matching.txt"
                step2_full_path = os.path.join(project_root, step2_prompt_path)
                try:
                    with open(step2_full_path, encoding="utf-8") as f:
                        step2_template = f.read()

                    # Create the processed prompt with actual data (like step2_match_intents_to_foods does)
                    intents_list = "\n".join([f"{i + 1}. {intent}" for i, intent in enumerate(intents)])
                    display_df = first_batch_df[['consumable_id', 'consumable_name']].copy() if not args.dietary_flag else first_batch_df.copy()
                    food_dataframe = display_df.to_markdown(index=False)
                    step2_processed_prompt = step2_template.format(
                        intents_list=intents_list, food_dataframe=food_dataframe
                    )
                    processed_prompts["step2"] = step2_processed_prompt
                except (FileNotFoundError, Exception) as e:
                    processed_prompts["step2"] = f"Error processing step2 prompt: {e}"

            for batch_idx in range(total_batches):
                start_idx = batch_idx * args.batch_size
                end_idx = min(start_idx + args.batch_size, len(food_df))
                batch_df = food_df.iloc[start_idx:end_idx]

                # Determine which intents to use for this batch
                if intent_sets:
                    # Use rotating intent sets
                    current_intents, intent_set_index = get_intent_set_for_batch(
                        batch_idx, intent_sets, args.intent_set_rotation
                    )
                    logger.info(
                        f"Batch {batch_idx + 1}/{total_batches} using intent set #{intent_set_index + 1}"
                    )
                    # Track intent set usage
                    set_key = f"set_{intent_set_index + 1}"
                    intent_set_usage[set_key] = intent_set_usage.get(set_key, 0) + 1
                else:
                    # Use single generated intents (original behavior)
                    current_intents = intents

                # Smart matching for this batch
                batch_matches = step2_match_intents_to_foods(current_intents, batch_df, args.dietary_flag, args.step2_prompt)
                all_matches["matches"].extend(batch_matches["matches"])

            step2_time = time.time() - step2_start
            log_step_metrics(2, step2_time, successful_matches=len(all_matches["matches"]))

            # Step 3: Generate final queries (if not stopping at intents)
            step3_time = 0
            if not args.stop_at_intents:
                logger.info("Starting Step 3: Generate final queries")
                step3_start = time.time()

                # Capture processed step3 prompt (using first batch for example)
                if len(all_matches["matches"]) > 0:
                    step3_prompt_path = args.step3_prompt or "prompts/intent_generation/v1.3_intent_query_generation.txt"
                    step3_full_path = os.path.join(project_root, step3_prompt_path)
                    try:
                        with open(step3_full_path, encoding="utf-8") as f:
                            step3_template = f.read()

                        # Create example intent-food pairs (like step3_generate_final_queries does)
                        sample_matches = all_matches["matches"][:3]  # First 3 matches for example
                        intent_food_pairs = ""
                        for i, match in enumerate(sample_matches, 1):
                            food_info = f'{match["consumable_name"]} (ID: {match["consumable_id"]})'
                            if args.dietary_flag:
                                food_row = food_df[food_df['consumable_id'] == match["consumable_id"]]
                                if not food_row.empty:
                                    dietary_cols = [col for col in food_df.columns if col.startswith(('is_', 'contains_', 'dietary_'))]
                                    dietary_info = [col.replace('_', ' ').title() for col in dietary_cols if food_row[col].iloc[0]]
                                    if dietary_info:
                                        food_info += f" [{', '.join(dietary_info)}]"
                            intent_food_pairs += f'\n{i}. Intent: "{match["intent"]}"\n   Food: {food_info}\n'

                        step3_processed_prompt = step3_template.format(
                            intent_food_pairs=intent_food_pairs, queries_per_item=args.queries_per_item
                        )
                        processed_prompts["step3"] = step3_processed_prompt
                    except (FileNotFoundError, Exception) as e:
                        processed_prompts["step3"] = f"Error processing step3 prompt: {e}"

                for batch_idx in range(total_batches):
                    start_idx = batch_idx * args.batch_size
                    end_idx = min(start_idx + args.batch_size, len(food_df))
                    batch_df = food_df.iloc[start_idx:end_idx]

                    # Get matches for this batch
                    batch_matches = {
                        "matches": all_matches["matches"][start_idx:end_idx]
                    }

                    batch_queries = step3_generate_final_queries(
                        batch_matches, batch_df, args.queries_per_item, args.dietary_flag, dietary_columns, args.step3_prompt
                    )
                    all_final_queries.extend(batch_queries)

                step3_time = time.time() - step3_start
                log_step_metrics(3, step3_time, total_queries_generated=len(all_final_queries))

            # Save results
            output_paths = save_results(
                intents, all_matches, all_final_queries, food_df,
                args.output_dir, args.stop_at_intents, args.dietary_flag, dietary_columns,
                intent_set_usage
            )

            # Save processed prompts as separate TXT files
            processed_prompt_files = save_processed_prompts(processed_prompts, args.output_dir)

            # Add config to output paths
            output_paths["config_snapshot"] = config_path

            # Add each processed prompt file to output paths
            for i, prompt_file in enumerate(processed_prompt_files):
                step_name = os.path.basename(prompt_file).split('_')[0]  # Extract step1, step2, etc.
                output_paths[f"processed_prompt_{step_name}"] = prompt_file

            # Log parameters and prompt versions
            prompt_versions = {
                "step1": args.step1_prompt or "v1.1_intent_generation.txt",
                "step2": args.step2_prompt or "v1.2_intent_matching.txt",
                "step3": (args.step3_prompt or "v1.3_intent_query_generation.txt") if not args.stop_at_intents else None
            }
            log_parameters(args, prompt_versions, intent_set_usage)

            # Log final metrics
            total_time = time.time() - total_start_time
            steps_executed = 2 if args.stop_at_intents else 3
            queries_count = len(all_final_queries) if not args.stop_at_intents else len(all_matches["matches"])

            intents_count = len(intents) if intents else (len(intent_sets) if intent_sets else 0)
            log_final_metrics(
                total_time, steps_executed, args.stop_at_intents,
                intents_count, len(all_matches["matches"]), queries_count, intent_set_usage
            )

            # Log artifacts
            log_artifacts(output_paths)

            # Success summary
            logger.info(f"MLflow run completed successfully: {run.info.run_id}")
            if args.stop_at_intents:
                intents_count_msg = len(intents) if intents else len(intent_sets) if intent_sets else 0
                logger.info(
                    f"Used {intents_count_msg} intents matched to "
                    f"{len(all_matches['matches'])} foods (stopped at intents)"
                )
            else:
                intent_queries = len([q for q in all_final_queries if q["query_type"] == "intent"])
                bridged_queries = len([q for q in all_final_queries if q["query_type"] == "bridged"])
                logger.info(
                    f"Generated {len(all_final_queries)} total queries "
                    f"({intent_queries} intent + {bridged_queries} bridged)"
                )

            print(f"✅ MLflow run completed: {run.info.run_id}")
            print(f"📊 View results: mlflow ui --backend-store-uri file://{mlflow.get_tracking_uri().replace('file://', '')}")

        except Exception as e:
            logger.error(f"Error during execution: {e}")
            mlflow.log_param("error", str(e))
            raise

        return 0


if __name__ == "__main__":
    exit(main())