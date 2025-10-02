#!/usr/bin/env python3
"""
MLflow wrapper for initial data generation script.

This script wraps the initial_generation.py script with comprehensive MLflow tracking,
logging parameters, metrics, and artifacts for experiment management.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Add the project root to Python path for imports
project_root = os.environ.get("root_folder")
if project_root:
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, os.path.join(project_root, "esci-dataset"))

# Import the original script functions
from src.data_generation.dspy_schemas import (  # noqa: E402
    QueryGenerator,
    setup_dspy_model,
)
from src.data_generation.initial_generation import (  # noqa: E402
    generate_output_filename,
    generate_queries_with_retry,
    get_api_key,
    get_template_path,
    load_and_process_data,
    save_results_as_csv,
    validate_args,
)
from src.data_generation.initial_generation import (  # noqa: E402
    setup_argparser as setup_original_argparser,
)
from src.utils import get_git_info  # noqa: E402

# Configure logging
root_path = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
log_dir = root_path / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = log_dir / f"mlflow_initial_generation_{log_suffix}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file_path),
    ],
    force=True,
)
logger = logging.getLogger(__name__)


def setup_mlflow(experiment_name: str) -> None:
    """Setup MLflow tracking."""
    # Set tracking URI to local mlruns directory
    mlflow_tracking_uri = (
        os.path.join(project_root, "mlruns") if project_root else "./mlruns"
    )
    mlflow.set_tracking_uri(f"file://{mlflow_tracking_uri}")

    # Set or create experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(
                f"Created new MLflow experiment: {experiment_name} "
                f"(ID: {experiment_id})"
            )
        else:
            experiment_id = experiment.experiment_id
            logger.info(
                f"Using existing MLflow experiment: {experiment_name} "
                f"(ID: {experiment_id})"
            )

        mlflow.set_experiment(experiment_name)
    except Exception as e:
        logger.error(f"Failed to setup MLflow experiment: {e}")
        raise


def log_parameters(
    args: argparse.Namespace, template_path: str, query_examples_path: str
) -> None:
    """Log all parameters to MLflow."""
    # Log all CLI arguments
    mlflow.log_param("script_name", "initial_generation.py")
    mlflow.log_param("esci_label", args.esci_label)
    mlflow.log_param("limit", args.limit)
    mlflow.log_param("batch_size", args.batch_size)
    mlflow.log_param("dietary_flag", args.dietary_flag)
    mlflow.log_param("output_path", args.output_path)
    mlflow.log_param("model", args.model)
    mlflow.log_param("temperature", args.temperature)
    mlflow.log_param("queries_per_item", args.queries_per_item)
    mlflow.log_param("max_retries", args.max_retries)
    mlflow.log_param("start_idx", args.start_idx)
    mlflow.log_param("parallel_threads", args.parallel)

    # Log paths
    mlflow.log_param("template_path", template_path)
    mlflow.log_param("query_examples_path", query_examples_path or "none")

    # Log derived parameters
    template_name = os.path.basename(template_path) if template_path else "unknown"
    mlflow.log_param("template_version", template_name)

    # Log generation approach parameters for migration tracking
    mlflow.log_param("generation_approach", "initial")
    mlflow.log_param("step", "complete")
    mlflow.log_param("output_type", "enhanced_json")

    # Log data_gen_hash and MLflow run ID for database migration tracking
    git_info = get_git_info()
    run_id = mlflow.active_run().info.run_id
    mlflow.log_param("data_gen_hash", git_info["commit_hash"])
    mlflow.log_param("mlflow_run_id", run_id)


def log_template_content_as_artifact(template_path: str) -> None:
    """Log template content as MLflow artifact."""
    if template_path and os.path.exists(template_path):
        try:
            mlflow.log_artifact(template_path, "templates")
            logger.info(f"Logged template artifact: {template_path}")
        except Exception as e:
            logger.warning(f"Failed to log template artifact: {e}")


def log_query_examples_as_artifact(query_examples_path: str) -> None:
    """Log query examples as MLflow artifact."""
    if query_examples_path and os.path.exists(query_examples_path):
        try:
            mlflow.log_artifact(query_examples_path, "examples")
            logger.info(f"Logged query examples artifact: {query_examples_path}")
        except Exception as e:
            logger.warning(f"Failed to log query examples artifact: {e}")


def save_processed_prompts_as_artifact(
    processed_prompts: dict, output_dir: str
) -> list[str]:
    """Save processed prompts as separate TXT files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = []

    for batch_key, batch_data in processed_prompts.items():
        if batch_data and "prompt" in batch_data:
            batch_num = batch_data["batch_number"]
            filename = f"batch{batch_num}_processed_{timestamp}.txt"
            file_path = os.path.join(output_dir, filename)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(batch_data["prompt"])

            mlflow.log_artifact(file_path, "prompts")
            logger.info(f"Logged processed prompt artifact: {file_path}")
            saved_files.append(file_path)

    return saved_files


def log_timing_metrics(total_runtime: float, batch_times: list[float]) -> None:
    """Log timing metrics to MLflow."""
    mlflow.log_metric("total_runtime_seconds", total_runtime)

    if batch_times:
        mlflow.log_metric("avg_batch_time", sum(batch_times) / len(batch_times))
        mlflow.log_metric("min_batch_time", min(batch_times))
        mlflow.log_metric("max_batch_time", max(batch_times))
        mlflow.log_metric("total_batches", len(batch_times))


def log_success_metrics(
    total_candidates: int,
    total_examples: int,
    total_queries: int,
    successful_batches: int,
    failed_batches: int,
) -> None:
    """Log success metrics to MLflow."""
    mlflow.log_metric("total_candidates_generated", total_candidates)
    mlflow.log_metric("total_examples_generated", total_examples)
    mlflow.log_metric("total_queries_generated", total_queries)
    mlflow.log_metric("successful_batches", successful_batches)
    mlflow.log_metric("failed_batches", failed_batches)

    total_batches = successful_batches + failed_batches
    if total_batches > 0:
        success_rate = (successful_batches / total_batches) * 100
        mlflow.log_metric("success_rate_percent", success_rate)


def save_batch_details(batch_details: list[dict], output_dir: str) -> str:
    """Save detailed batch information as JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_details_path = os.path.join(output_dir, f"batch_details_{timestamp}.json")

    with open(batch_details_path, "w", encoding="utf-8") as f:
        json.dump(batch_details, f, indent=2)

    return batch_details_path


def save_batch_failures(failures: list[dict], output_dir: str) -> str:
    """Save batch failure information as JSON."""
    if not failures:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    failures_path = os.path.join(output_dir, f"batch_failures_{timestamp}.json")

    with open(failures_path, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)

    return failures_path


def save_config_snapshot(
    args: argparse.Namespace,
    output_dir: str,
    template_path: str,
    query_examples_path: str,
) -> str:
    """Save configuration snapshot as JSON."""
    config = {
        "script_name": "initial_generation.py",
        "parameters": vars(args),
        "template_path": template_path,
        "query_examples_path": query_examples_path,
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
    template_version = "v1"  # default
    if args.template_path:
        template_name = os.path.basename(args.template_path)
        if template_name.startswith("v") and ".txt" in template_name:
            template_version = template_name.replace(".txt", "")

    start_suffix = f"-start{args.start_idx}" if args.start_idx > 0 else ""
    return (
        f"initial-{args.esci_label}-{args.model}-"
        f"batch{args.batch_size}-limit{args.limit}{start_suffix}-"
        f"qpi{args.queries_per_item}-{template_version}-{timestamp}"
    )


def process_in_batches_with_tracking(
    df: pd.DataFrame,
    args: argparse.Namespace,
    generator: QueryGenerator,
    template_path: str,
    query_examples_path: str,
    output_dir: str,
    dietary_columns: list[str] = None,
) -> tuple[dict, list[dict], list[dict], list[float], dict]:
    """Enhanced batch processing with detailed tracking."""
    all_candidates = []
    batch_details = []
    batch_failures = []
    batch_times = []
    processed_prompts = {}  # Store processed prompts for each batch

    total_batches = (len(df) + args.batch_size - 1) // args.batch_size

    logger.info(
        f"Processing {len(df)} records in {total_batches} batches of "
        f"size {args.batch_size}"
    )

    for batch_idx in range(total_batches):
        batch_start_time = time.time()
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx].copy()

        logger.info(
            f"Processing batch {batch_idx + 1}/{total_batches} "
            f"({len(batch_df)} records)"
        )

        batch_detail = {
            "batch_number": batch_idx + 1,
            "start_index": start_idx,
            "end_index": end_idx,
            "records_count": len(batch_df),
            "start_time": datetime.now().isoformat(),
        }

        # Prepare prompt for this batch
        from src.data_generation.prompt_template import prepare_prompt

        prompt = prepare_prompt(
            template_path=template_path,
            df=batch_df,
            esci_label=args.esci_label,
            batch_size=len(batch_df),
            include_dietary=args.dietary_flag,
            queries_per_item=args.queries_per_item,
            query_examples_path=query_examples_path,
            dietary_columns=dietary_columns,
        )

        logger.info(f"Batch {batch_idx + 1} prompt length: {len(prompt)} characters")
        batch_detail["prompt_length"] = len(prompt)

        # Store the processed prompt for this batch
        processed_prompts[f"batch_{batch_idx + 1}"] = {
            "batch_number": batch_idx + 1,
            "prompt": prompt,
            "prompt_length": len(prompt),
            "records_count": len(batch_df),
        }

        # Generate queries for this batch
        try:
            structured_output = generate_queries_with_retry(
                generator, prompt, args.esci_label, args.max_retries, batch_idx + 1
            )

            # Extract candidates from structured output
            batch_candidates = structured_output.model_dump().get("candidates", [])
            all_candidates.extend(batch_candidates)

            # Calculate metrics for this batch
            total_queries = sum(
                len(candidate.get("queries", [])) for candidate in batch_candidates
            )

            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)

            batch_detail.update(
                {
                    "status": "success",
                    "candidates_generated": len(batch_candidates),
                    "queries_generated": total_queries,
                    "execution_time_seconds": batch_time,
                    "end_time": datetime.now().isoformat(),
                }
            )

            logger.info(
                f"Batch {batch_idx + 1} completed: {len(batch_candidates)} "
                f"candidates, {total_queries} queries generated in {batch_time:.2f}s"
            )

        except Exception as e:
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)

            batch_detail.update(
                {
                    "status": "failed",
                    "error": str(e),
                    "execution_time_seconds": batch_time,
                    "end_time": datetime.now().isoformat(),
                }
            )

            batch_failures.append(
                {
                    "batch_number": batch_idx + 1,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "start_index": start_idx,
                    "end_index": end_idx,
                    "records_count": len(batch_df),
                }
            )

            logger.error(f"Batch {batch_idx + 1} failed: {e}")
            # Continue with next batch instead of failing completely

        batch_details.append(batch_detail)

    return (
        {"candidates": all_candidates},
        batch_details,
        batch_failures,
        batch_times,
        processed_prompts,
    )


def process_single_batch_for_parallel(
    batch_data: dict,
    args: argparse.Namespace,
    template_path: str,
    query_examples_path: str,
    dietary_columns: list[str] = None,
) -> tuple[dict, dict, float]:
    """Process a single batch for parallel execution."""
    batch_idx = batch_data["batch_idx"]
    batch_df = batch_data["batch_df"]

    batch_start_time = time.time()

    logger.info(
        f"Processing batch {batch_idx + 1} (parallel) - {len(batch_df)} records"
    )

    batch_detail = {
        "batch_number": batch_idx + 1,
        "start_index": batch_data["start_idx"],
        "end_index": batch_data["end_idx"],
        "records_count": len(batch_df),
        "start_time": datetime.now().isoformat(),
    }

    try:
        # Prepare prompt for this batch
        from src.data_generation.prompt_template import prepare_prompt

        prompt = prepare_prompt(
            template_path=template_path,
            df=batch_df,
            esci_label=args.esci_label,
            batch_size=len(batch_df),
            include_dietary=args.dietary_flag,
            queries_per_item=args.queries_per_item,
            query_examples_path=query_examples_path,
            dietary_columns=dietary_columns,
        )

        batch_detail["prompt_length"] = len(prompt)

        # Create a new generator instance for this thread
        generator = QueryGenerator()

        # Generate queries for this batch
        structured_output = generate_queries_with_retry(
            generator, prompt, args.esci_label, args.max_retries, batch_idx + 1
        )

        # Extract candidates from structured output
        batch_candidates = structured_output.model_dump().get("candidates", [])

        # Calculate metrics for this batch
        total_queries = sum(
            len(candidate.get("queries", [])) for candidate in batch_candidates
        )

        batch_time = time.time() - batch_start_time

        batch_detail.update(
            {
                "status": "success",
                "candidates_generated": len(batch_candidates),
                "queries_generated": total_queries,
                "execution_time_seconds": batch_time,
                "end_time": datetime.now().isoformat(),
            }
        )

        logger.info(
            f"Batch {batch_idx + 1} completed: {len(batch_candidates)} "
            f"candidates, {total_queries} queries generated in {batch_time:.2f}s"
        )

        return {
            "candidates": batch_candidates,
            "batch_detail": batch_detail,
            "batch_time": batch_time,
            "processed_prompt": {
                f"batch_{batch_idx + 1}": {
                    "batch_number": batch_idx + 1,
                    "prompt": prompt,
                    "prompt_length": len(prompt),
                    "records_count": len(batch_df),
                }
            },
        }

    except Exception as e:
        batch_time = time.time() - batch_start_time

        batch_detail.update(
            {
                "status": "failed",
                "error": str(e),
                "execution_time_seconds": batch_time,
                "end_time": datetime.now().isoformat(),
            }
        )

        batch_failure = {
            "batch_number": batch_idx + 1,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "start_index": batch_data["start_idx"],
            "end_index": batch_data["end_idx"],
            "records_count": len(batch_df),
        }

        logger.error(f"Batch {batch_idx + 1} failed: {e}")

        return {
            "candidates": [],
            "batch_detail": batch_detail,
            "batch_time": batch_time,
            "batch_failure": batch_failure,
            "processed_prompt": {},
        }


def process_in_batches_parallel(
    df: pd.DataFrame,
    args: argparse.Namespace,
    generator: QueryGenerator,
    template_path: str,
    query_examples_path: str,
    output_dir: str,
    dietary_columns: list[str] = None,
) -> tuple[dict, list[dict], list[dict], list[float], dict]:
    """Enhanced batch processing with parallel execution using dspy.Parallel."""
    all_candidates = []
    batch_details = []
    batch_failures = []
    batch_times = []
    processed_prompts = {}

    total_batches = (len(df) + args.batch_size - 1) // args.batch_size

    logger.info(
        f"Processing {len(df)} records in {total_batches} batches of "
        f"size {args.batch_size} using {args.parallel} parallel threads"
    )

    # Prepare batch data for parallel processing
    batch_data_list = []
    for batch_idx in range(total_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx].copy()

        batch_data_list.append(
            {
                "batch_idx": batch_idx,
                "batch_df": batch_df,
                "start_idx": start_idx,
                "end_idx": end_idx,
            }
        )

    # Execute batches in parallel using concurrent.futures for better control
    import functools
    from concurrent.futures import ThreadPoolExecutor

    logger.info(f"Starting parallel execution with {args.parallel} threads...")

    # Create a partial function with fixed arguments
    partial_func = functools.partial(
        process_single_batch_for_parallel,
        args=args,
        template_path=template_path,
        query_examples_path=query_examples_path,
        dietary_columns=dietary_columns,
    )

    # Execute batches in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        results = list(executor.map(partial_func, batch_data_list))

    # Process results and aggregate data
    for result in results:
        # Add candidates
        all_candidates.extend(result["candidates"])

        # Add batch details
        batch_details.append(result["batch_detail"])

        # Add batch times
        batch_times.append(result["batch_time"])

        # Add failures if any
        if "batch_failure" in result:
            batch_failures.append(result["batch_failure"])

        # Add processed prompts
        processed_prompts.update(result["processed_prompt"])

    return (
        {"candidates": all_candidates},
        batch_details,
        batch_failures,
        batch_times,
        processed_prompts,
    )


def setup_argparser() -> argparse.ArgumentParser:
    """Setup command line argument parser with MLflow additions."""
    # Get the original parser
    parser = setup_original_argparser()

    # Add MLflow-specific arguments
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="MLflow experiment name (default: initial-generation-{esci_label})",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="MLflow run name (auto-generated if not provided)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel threads for batch processing "
        "(default: 1 for sequential)",
    )

    return parser


def main():
    """Main execution with MLflow tracking."""
    parser = setup_argparser()
    args = parser.parse_args()

    # Validate arguments using original validation
    validate_args(args)

    # Additional validation for parallel processing
    if args.parallel < 1:
        raise ValueError("--parallel must be >= 1")
    if args.parallel > 32:  # Reasonable upper limit
        raise ValueError(
            "--parallel must be <= 32 (too many threads can hurt performance)"
        )

    # Generate default experiment name based on ESCI label
    experiment_name = args.experiment_name or f"initial-generation-{args.esci_label}"

    # Setup MLflow
    setup_mlflow(experiment_name)

    # Generate run name if not provided
    run_name = args.run_name or generate_run_name(args)

    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"Started MLflow run: {run.info.run_id}")

        # Set tags
        git_info = get_git_info()
        mlflow.set_tag("mlflow.source.git.commit", git_info["commit_hash"])
        mlflow.set_tag("mlflow.source.git.branch", git_info["branch_name"])
        mlflow.set_tag("initial-generation", "true")
        mlflow.set_tag("esci_label", args.esci_label)
        mlflow.set_tag("model", args.model)
        mlflow.set_tag(f"batch-{args.batch_size}", "true")
        mlflow.set_tag(f"limit-{args.limit}", "true")
        mlflow.set_tag(f"parallel-{args.parallel}", "true")

        # Add processing mode tag
        if args.parallel > 1:
            mlflow.set_tag("processing_mode", "parallel")
        else:
            mlflow.set_tag("processing_mode", "sequential")

        # Add resume/restart tracking
        if args.start_idx > 0:
            mlflow.set_tag("resumed_job", "true")
            mlflow.set_tag(f"start-idx-{args.start_idx}", "true")
        else:
            mlflow.set_tag("resumed_job", "false")

        # Set approval tag for migration workflow
        mlflow.set_tag("data_status", "pending_review")

        try:
            total_start_time = time.time()

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

            # Create output directory for artifacts
            output_dir = (
                os.path.dirname(args.output_path) if args.output_path else "output"
            )
            if project_root and not os.path.isabs(output_dir):
                output_dir = os.path.join(project_root, output_dir)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Save config snapshot
            config_path = save_config_snapshot(
                args, output_dir, template_path, query_examples_path
            )

            # Log parameters and artifacts
            log_parameters(args, template_path, query_examples_path)
            log_template_content_as_artifact(template_path)
            log_query_examples_as_artifact(query_examples_path)

            # Choose processing method based on parallel flag
            if args.parallel > 1:
                # Use parallel processing
                (
                    output_dict,
                    batch_details,
                    batch_failures,
                    batch_times,
                    processed_prompts,
                ) = process_in_batches_parallel(
                    df,
                    args,
                    generator,
                    template_path,
                    query_examples_path,
                    output_dir,
                    dietary_columns,
                )
            else:
                # Use sequential processing
                (
                    output_dict,
                    batch_details,
                    batch_failures,
                    batch_times,
                    processed_prompts,
                ) = process_in_batches_with_tracking(
                    df,
                    args,
                    generator,
                    template_path,
                    query_examples_path,
                    output_dir,
                    dietary_columns,
                )

            # Calculate metrics
            total_candidates = len(output_dict.get("candidates", []))
            total_examples = sum(
                len(candidate.get("queries", []))
                for candidate in output_dict.get("candidates", [])
            )
            # total queries is unique query strings across all candidates
            # logger.info(output_dict.get("candidates", []))
            total_queries = len(
                set(
                    query["query"]
                    for candidate in output_dict.get("candidates", [])
                    for query in candidate.get("queries", [])
                )
            )
            successful_batches = len(
                [bd for bd in batch_details if bd["status"] == "success"]
            )
            failed_batches = len(
                [bd for bd in batch_details if bd["status"] == "failed"]
            )

            # Log timing and success metrics
            total_runtime = time.time() - total_start_time
            log_timing_metrics(total_runtime, batch_times)
            log_success_metrics(
                total_candidates,
                total_examples,
                total_queries,
                successful_batches,
                failed_batches,
            )

            # Handle failures
            if batch_failures:
                mlflow.set_tag("has_failures", "true")
                mlflow.set_tag("partial_failure", "true")
                failures_path = save_batch_failures(batch_failures, output_dir)
                if failures_path:
                    mlflow.log_artifact(failures_path, "failures")

            # Save and log batch details
            batch_details_path = save_batch_details(batch_details, output_dir)
            mlflow.log_artifact(batch_details_path, "details")

            # Save and log processed prompts
            save_processed_prompts_as_artifact(processed_prompts, output_dir)

            # Determine output path and save results
            output_path = args.output_path or generate_output_filename(args)
            save_results_as_csv(output_dict, output_path, args, df, dietary_columns)

            # Log output as artifact
            if os.path.exists(output_path):
                mlflow.log_artifact(output_path, "outputs")

            # Log config artifact
            mlflow.log_artifact(config_path, "config")

            logger.info("Query generation completed successfully!")

            # Print summary
            if successful_batches == len(batch_details):
                logger.info("✅ All batches completed successfully!")
            else:
                logger.warning(
                    f"⚠️ {failed_batches} out of {len(batch_details)} batches failed"
                )

            logger.info(
                f"Generated {total_queries} queries for "
                f"{total_candidates} candidates in {total_runtime:.2f}s"
            )

            print(f"✅ MLflow run completed: {run.info.run_id}")
            print(
                f"📊 View results: mlflow ui --backend-store-uri file://"
                f"{mlflow.get_tracking_uri().replace('file://', '')}"
            )
            print(f"📄 Output saved to: {output_path}")
            print(f"🏷️ ESCI Label: {args.esci_label}")
            print(f"📈 Success Rate: {successful_batches}/{len(batch_details)} batches")

        except Exception as e:
            logger.error(f"Error during execution: {e}")
            mlflow.log_param("execution_error", str(e))
            mlflow.set_tag("execution_failed", "true")
            raise

    return 0


if __name__ == "__main__":
    exit(main())
