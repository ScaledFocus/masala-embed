#!/usr/bin/env python3
"""
Generate diverse intent sets using weighted variations of the base prompt.

This script creates 10 shuffled intent sets by generating themed intents
and mixing them naturally, providing maximum diversity while maintaining
consistency with the existing prompt structure.
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Add the project root to Python path for imports
project_root = os.environ.get("root_folder")
print(f"Project root from env: {project_root}")
if project_root:
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, os.path.join(project_root, "esci-dataset"))

from src.data_generation.intent_generation_approach import (  # noqa: E402
    IntentGenerator,
    setup_dspy_client,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("generate_intent_sets.log"),
    ],
)
logger = logging.getLogger(__name__)


def get_theme_configurations() -> dict[str, dict[str, Any]]:
    """Define 10 theme configurations for prompt weighting."""
    return {
        "quick_convenient": {
            "description": "Fast food, convenience, time-pressed scenarios",
            "emphasize_personas": ["Busy Professional", "Lazy Student"],
            "pattern_weights": {"Need-Based": 20, "Situational": 25, "Practical": 35, "Craving": 20},
            "add_examples": [
                "fast food near me", "quick lunch delivery", "express delivery",
                "food in under 30 minutes", "grab and go meals"
            ],
        },
        "health_wellness": {
            "description": "Nutrition, diet-conscious, fitness-oriented",
            "emphasize_personas": ["Health-Conscious", "Busy Professional"],
            "pattern_weights": {"Need-Based": 35, "Situational": 15, "Practical": 10, "Craving": 40},
            "add_examples": [
                "healthy options near me", "low calorie meals", "protein rich food",
                "nutritious delivery", "clean eating options"
            ],
        },
        "comfort_indulgent": {
            "description": "Comfort food, treats, indulgent cravings",
            "emphasize_personas": ["Comfort Seeker", "Craving-Driven"],
            "pattern_weights": {"Need-Based": 30, "Situational": 15, "Practical": 10, "Craving": 45},
            "add_examples": [
                "comfort food delivery", "indulgent treats", "cheat meal tonight",
                "guilty pleasure food", "decadent desserts"
            ],
        },
        "international_cuisine": {
            "description": "Ethnic foods, cultural cravings, travel-inspired",
            "emphasize_personas": ["Craving-Driven", "Social Diner"],
            "pattern_weights": {"Need-Based": 20, "Situational": 20, "Practical": 15, "Craving": 45},
            "add_examples": [
                "authentic ethnic food", "international cuisine", "exotic flavors",
                "traditional dishes", "cultural food experiences"
            ],
        },
        "social_sharing": {
            "description": "Party food, group meals, social dining",
            "emphasize_personas": ["Party Planner", "Social Diner"],
            "pattern_weights": {"Need-Based": 15, "Situational": 40, "Practical": 15, "Craving": 30},
            "add_examples": [
                "food for groups", "party platters", "sharing meals",
                "social dining options", "crowd pleasers"
            ],
        },
        "budget_conscious": {
            "description": "Cheap eats, value meals, student-friendly",
            "emphasize_personas": ["Budget-Conscious", "Lazy Student"],
            "pattern_weights": {"Need-Based": 25, "Situational": 20, "Practical": 40, "Craving": 15},
            "add_examples": [
                "cheap eats", "value meals", "budget friendly food",
                "affordable options", "student discounts"
            ],
        },
        "premium_gourmet": {
            "description": "High-end, special occasions, quality-focused",
            "emphasize_personas": ["Social Diner", "Craving-Driven"],
            "pattern_weights": {"Need-Based": 30, "Situational": 25, "Practical": 10, "Craving": 35},
            "add_examples": [
                "gourmet delivery", "premium options", "fine dining",
                "upscale food", "special occasion meals"
            ],
        },
        "dietary_restrictions": {
            "description": "Vegan, gluten-free, allergy-conscious",
            "emphasize_personas": ["Health-Conscious", "Craving-Driven"],
            "pattern_weights": {"Need-Based": 40, "Situational": 10, "Practical": 15, "Craving": 35},
            "add_examples": [
                "vegan options", "gluten free food", "allergy friendly",
                "dietary restrictions", "special diet meals"
            ],
        },
        "mood_occasion": {
            "description": "Breakfast, late-night, hangover, celebration",
            "emphasize_personas": ["Comfort Seeker", "Lazy Student"],
            "pattern_weights": {"Need-Based": 25, "Situational": 35, "Practical": 15, "Craving": 25},
            "add_examples": [
                "late night food", "hangover cure", "breakfast delivery",
                "celebration meals", "mood food"
            ],
        },
        "seasonal_fresh": {
            "description": "Seasonal ingredients, fresh produce, weather-based",
            "emphasize_personas": ["Health-Conscious", "Craving-Driven"],
            "pattern_weights": {"Need-Based": 30, "Situational": 25, "Practical": 15, "Craving": 30},
            "add_examples": [
                "fresh ingredients", "seasonal specials", "locally sourced",
                "farm to table", "weather appropriate food"
            ],
        },
    }


def load_base_prompt(prompt_path: str) -> str:
    """Load the base intent generation prompt."""
    try:
        with open(prompt_path, encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Base prompt not found at: {prompt_path}")


def create_themed_prompt_variation(base_prompt: str, theme_config: dict[str, Any]) -> str:
    """Create a themed variation of the base prompt by adjusting emphasis and examples."""

    # Start with the base prompt
    themed_prompt = base_prompt

    # Add theme-specific emphasis
    theme_intro = f"\n**THEME FOCUS: {theme_config['description'].upper()}**\n"
    theme_intro += f"Pay extra attention to these personas: {', '.join(theme_config['emphasize_personas'])}\n"

    # Add theme-specific examples
    if theme_config['add_examples']:
        theme_intro += "Include variations of these theme-specific examples:\n"
        for example in theme_config['add_examples']:
            theme_intro += f"- \"{example}\"\n"

    # Insert theme intro after the main description
    insertion_point = themed_prompt.find("## **REAL USER SCENARIOS**")
    if insertion_point != -1:
        themed_prompt = (
            themed_prompt[:insertion_point] +
            theme_intro + "\n" +
            themed_prompt[insertion_point:]
        )

    # Adjust pattern percentages
    pattern_weights = theme_config['pattern_weights']
    for pattern_name, weight in pattern_weights.items():
        # Find and replace the percentage
        old_pattern = f"**{pattern_name} ({weight}%):**"
        themed_prompt = themed_prompt.replace(f"**{pattern_name} (", f"**{pattern_name} ({weight}%):**\nEMPHASIZED: ")
        themed_prompt = themed_prompt.replace(f"**{pattern_name} ({weight}%):**\nEMPHASIZED: ", old_pattern)

    return themed_prompt


def generate_themed_intents(theme_name: str, theme_config: dict[str, Any],
                          base_prompt: str, num_intents: int = 50) -> list[str]:
    """Generate intents for a specific theme using DSPy."""
    logger.info(f"Generating {num_intents} intents for theme: {theme_name}")

    # Create themed prompt variation
    themed_prompt = create_themed_prompt_variation(base_prompt, theme_config)

    # Replace the default number with the requested number
    themed_prompt = themed_prompt.replace("Generate 100 diverse", f"Generate {num_intents} diverse")
    themed_prompt = themed_prompt.replace(
        "Return a simple list of 100 search queries",
        f"Return a simple list of {num_intents} search queries"
    )

    try:
        generator = IntentGenerator()
        result = generator(themed_prompt)
        intents = result.intents

        logger.info(f"Generated {len(intents)} intents for theme: {theme_name}")
        return intents

    except Exception as e:
        logger.error(f"Error generating intents for theme {theme_name}: {e}")
        raise


def create_shuffled_intent_sets(all_themed_intents: dict[str, list[str]],
                              num_sets: int = 10) -> list[dict[str, Any]]:
    """Create shuffled intent sets from all themed intents."""
    logger.info(f"Creating {num_sets} shuffled intent sets")

    # Flatten all themed intents into one list
    all_intents = []
    theme_sources = []  # Track which theme each intent came from

    for theme_name, intents in all_themed_intents.items():
        for intent in intents:
            all_intents.append(intent)
            theme_sources.append(theme_name)

    logger.info(f"Total themed intents collected: {len(all_intents)}")

    # Shuffle the combined set once
    random.seed(42)
    combined_data = list(zip(all_intents, theme_sources))
    random.shuffle(combined_data)
    shuffled_intents, shuffled_themes = zip(*combined_data)

    # Divide into equal parts
    total_intents = len(shuffled_intents)
    intents_per_set_actual = total_intents // num_sets

    shuffled_sets = []
    for i in range(num_sets):
        start_idx = i * intents_per_set_actual
        end_idx = start_idx + intents_per_set_actual if i < num_sets - 1 else total_intents

        selected_intents = list(shuffled_intents[start_idx:end_idx])
        selected_themes = list(shuffled_themes[start_idx:end_idx])

        # Calculate theme distribution for metadata
        theme_distribution = {}
        for theme in selected_themes:
            theme_distribution[theme] = theme_distribution.get(theme, 0) + 1

        intent_set = {
            "metadata": {
                "set_number": i + 1,
                "creation_date": datetime.now().isoformat(),
                "seed": 42,
                "theme_distribution": theme_distribution,
                "total_intents": len(selected_intents),
            },
            "intents": selected_intents
        }

        shuffled_sets.append(intent_set)

        logger.info(f"Created intent set {i + 1} with {len(selected_intents)} intents")
        logger.info(f"Theme distribution: {theme_distribution}")

    return shuffled_sets


def save_intent_sets(intent_sets: list[dict[str, Any]], output_dir: str) -> list[str]:
    """Save intent sets as JSON files."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    saved_files = []
    for intent_set in intent_sets:
        set_number = intent_set["metadata"]["set_number"]
        filename = f"intent_set_{set_number:02d}.json"
        file_path = os.path.join(output_dir, filename)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(intent_set, f, indent=2, ensure_ascii=False)

        saved_files.append(file_path)
        logger.info(f"Saved intent set {set_number} to {file_path}")

    # Save summary file
    summary = {
        "generation_info": {
            "total_sets": len(intent_sets),
            "creation_date": datetime.now().isoformat(),
            "intents_per_set": intent_sets[0]["metadata"]["intents_per_set"] if intent_sets else 0,
        },
        "theme_info": list(get_theme_configurations().keys()),
        "files": [os.path.basename(f) for f in saved_files]
    }

    summary_path = os.path.join(output_dir, "intent_sets_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    saved_files.append(summary_path)
    logger.info(f"Saved summary to {summary_path}")

    return saved_files


def setup_argparser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate diverse intent sets using themed prompt variations"
    )

    parser.add_argument(
        "--output-dir",
        default="intent_sets",
        help="Output directory for intent sets (default: intent_sets)"
    )

    parser.add_argument(
        "--base-prompt",
        default=None,
        help="Path to base intent generation prompt (default: prompts/intent_generation/v1.1_intent_generation.txt)"
    )

    parser.add_argument(
        "--num-sets",
        type=int,
        default=10,
        help="Number of shuffled intent sets to create (default: 10)"
    )

    parser.add_argument(
        "--intents-per-theme",
        type=int,
        default=50,
        help="Number of intents to generate per theme (default: 50)"
    )

    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="OpenAI model to use (default: gpt-5-mini)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for model generation (default: 1.0)"
    )

    return parser


def main():
    """Main execution."""
    parser = setup_argparser()
    args = parser.parse_args()

    logger.info("Starting intent set generation...")
    logger.info(f"Configuration: sets={args.num_sets}, intents_per_theme={args.intents_per_theme}, model={args.model}")

    try:
        # Setup DSPy
        setup_dspy_client(args.model, args.temperature)

        # Get base prompt path
        base_prompt_path = args.base_prompt
        if not base_prompt_path:
            base_prompt_path = os.path.join(
                project_root, "prompts", "intent_generation", "v1.1_intent_generation.txt"
            )
        elif not os.path.isabs(base_prompt_path):
            base_prompt_path = os.path.join(project_root, base_prompt_path)

        logger.info(f"Using base prompt: {base_prompt_path}")

        # Load base prompt
        base_prompt = load_base_prompt(base_prompt_path)

        # Get theme configurations
        theme_configs = get_theme_configurations()
        logger.info(f"Generating intents for {len(theme_configs)} themes")

        # Generate themed intents
        all_themed_intents = {}
        for theme_name, theme_config in theme_configs.items():
            themed_intents = generate_themed_intents(
                theme_name, theme_config, base_prompt, args.intents_per_theme
            )
            all_themed_intents[theme_name] = themed_intents

        # Create shuffled intent sets
        intent_sets = create_shuffled_intent_sets(
            all_themed_intents, args.num_sets
        )

        # Save intent sets
        saved_files = save_intent_sets(intent_sets, args.output_dir)

        logger.info("Intent set generation completed successfully!")

        # Print summary
        total_themed_intents = sum(len(intents) for intents in all_themed_intents.values())
        print(f"\n✅ Success! Generated {len(intent_sets)} shuffled intent sets")
        print(f"📊 Total themed intents: {total_themed_intents}")
        print(f"📁 Intent sets saved to: {args.output_dir}")
        print("🔀 Each set contains mixed intents from all themes (divided equally)")
        print(f"📄 Files created: {len(saved_files)}")

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
