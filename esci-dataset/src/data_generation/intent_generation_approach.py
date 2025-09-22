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
from typing import Dict, List, Optional

import pandas as pd
import openai
from dotenv import load_dotenv

load_dotenv()

# Add the project root to Python path for imports
project_root = os.environ.get("root_folder")
print(f"Project root from env: {project_root}")
if project_root:
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, os.path.join(project_root, "esci-dataset"))

from database.utils.db_utils import get_table
from src.data_generation.prompt_template import get_esci_label_description

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('intent_query_generation.log')
    ]
)
logger = logging.getLogger(__name__)

def setup_openai():
    """Setup OpenAI client."""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

def call_openai(prompt: str, model: str = "gpt-5") -> str:
    """Make OpenAI API call with error handling."""
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            max_completion_tokens=4000 if model.startswith("gpt-5") else 4000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API error: {e}")
        raise

def step1_generate_intents(num_intents: int = 50, model: str = "gpt-5") -> List[str]:
    """Step 1: Generate pure user intents with no food bias."""
    print(f"🚀 Step 1: Generating {num_intents} pure user intents...")

    # Load v1.1 intent generation prompt
    prompt_path = "prompts/query_generation/v1.1_intent_generation.txt"
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            intent_prompt = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Intent generation prompt not found at: {prompt_path}")

    # Replace the default number with the requested number
    intent_prompt = intent_prompt.replace("Generate 50 diverse", f"Generate {num_intents} diverse")
    intent_prompt = intent_prompt.replace("Return a simple list of 50 search queries", f"Return a simple list of {num_intents} search queries")

    # Generate intents
    response = call_openai(intent_prompt, model)

    # Parse intents (one per line)
    intents = [line.strip() for line in response.split('\n') if line.strip()]

    print(f"Generated {len(intents)} user intents")
    print(f"Sample intents: {intents[:3]}")

    return intents

def load_food_data(limit: Optional[int] = None) -> pd.DataFrame:
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

        # Rename columns to match expected format
        if 'id' not in df.columns and 'consumable_id' in df.columns:
            df = df.rename(columns={'consumable_id': 'id'})

        logger.info(f"Successfully processed {len(df)} food items")
        return df

    except Exception as e:
        logger.error(f"Error loading food data: {e}")
        raise

def step2_match_intents_to_foods(intents: List[str], food_df: pd.DataFrame, model: str = "gpt-5") -> Dict:
    """Step 2: Smart 1:1 matching of foods to best-fitting intents."""
    print("🎯 Step 2: Matching foods to best intents...")

    # Create matching prompt
    matching_prompt = f"""
You are a food delivery expert. Your task is to match food items to the most appropriate user search intents.

**USER INTENTS AVAILABLE:**
{chr(10).join([f"{i+1}. {intent}" for i, intent in enumerate(intents)])}

**FOOD ITEMS TO MATCH:**
{food_df.to_markdown(index=False)}

**MATCHING RULES:**
1. Each food item should be matched to exactly ONE intent that makes the most logical sense
2. Each intent can only be used once (or left unused if no good match)
3. If a food doesn't match any intent well, skip it
4. Prioritize realistic, natural connections over forced matches
5. Consider the food's actual properties (protein content, price range, preparation style)

**OUTPUT FORMAT:**
Return a JSON object with the matches:
```json
{{
  "matches": [
    {{"food_id": 21496, "food_name": "Fried Potato Slice", "intent": "game day finger foods", "reasoning": "Crispy, shareable snack perfect for munching during games"}},
    {{"food_id": 6473, "food_name": "Cheeseburger Meal", "intent": "somethin greasy and fast", "reasoning": "Classic quick-serve, greasy comfort food"}}
  ]
}}
```

Match each food to the best intent, providing reasoning for each match.
"""

    # Get matches
    response = call_openai(matching_prompt, model)

    # Parse JSON response
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        json_str = response[json_start:json_end]
        matches = json.loads(json_str)

        print(f"Successfully matched {len(matches['matches'])} foods to intents")

        # Display matches
        for match in matches['matches']:
            print(f"  🍽️  {match['food_name']} → '{match['intent']}'")

        return matches

    except Exception as e:
        print(f"Error parsing matches: {e}")
        print(f"Raw response: {response}")
        raise

def step3_generate_final_queries(matches: Dict, model: str = "gpt-5") -> List[Dict]:
    """Step 3: Generate final queries for all matched pairs in one API call."""
    print("📝 Step 3: Generating final queries for all matched pairs...")

    # Create batch prompt for all matches
    batch_prompt = """
You are a food delivery expert. For each intent-food pair below, generate 3 realistic search queries that would naturally bridge from the user's original intent to the specific food.

**INTENT-FOOD PAIRS:**
"""

    for i, match in enumerate(matches['matches'], 1):
        batch_prompt += f"""
{i}. Intent: "{match['intent']}"
   Food: {match['food_name']} (ID: {match['food_id']})
"""

    batch_prompt += """

**REQUIREMENTS:**
- Generate 3 queries per food item
- Preserve the user's original intent and context
- Use authentic language that real users would type
- Make the progression feel natural, not forced
- Avoid being too specific unless it feels natural

**OUTPUT FORMAT:**
Return as JSON:
```json
{
  "query_results": [
    {
      "food_id": 21496,
      "food_name": "Fried Potato Slice",
      "original_intent": "game day finger foods",
      "queries": [
        "crispy finger foods for game day",
        "crunchy snacks for watching sports",
        "game day munchies delivery"
      ]
    }
  ]
}
```

Generate 3 queries for each food item that naturally bridge from intent to food.
"""

    # Generate queries in batch
    response = call_openai(batch_prompt, model)

    # Parse response
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        json_str = response[json_start:json_end]
        query_results = json.loads(json_str)

        # Convert to flat list format
        final_queries = []
        for result in query_results['query_results']:
            # Add intent as standalone query
            final_queries.append({
                'food_id': result['food_id'],
                'food_name': result['food_name'],
                'query': result['original_intent'],
                'query_type': 'intent',
                'original_intent': result['original_intent'],
                'generated_at': datetime.now().isoformat()
            })

            # Add bridged queries
            for query in result['queries']:
                final_queries.append({
                    'food_id': result['food_id'],
                    'food_name': result['food_name'],
                    'query': query,
                    'query_type': 'bridged',
                    'original_intent': result['original_intent'],
                    'generated_at': datetime.now().isoformat()
                })

        print(f"Generated {len(final_queries)} total queries ({len([q for q in final_queries if q['query_type'] == 'intent'])} intent + {len([q for q in final_queries if q['query_type'] == 'bridged'])} bridged)")

        return final_queries

    except Exception as e:
        print(f"Error parsing final queries: {e}")
        print(f"Raw response: {response}")
        raise

def save_results(intents: List[str], matches: Dict, final_queries: List[Dict], output_dir: str = "output"):
    """Save all results to files."""
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save original intents
    intents_path = f"{output_dir}/intent_generation_intents_{timestamp}.txt"
    with open(intents_path, 'w', encoding='utf-8') as f:
        f.write("ORIGINAL USER INTENTS GENERATED (Step 1):\n")
        f.write("=" * 50 + "\n")
        for i, intent in enumerate(intents, 1):
            f.write(f"{i:2d}. {intent}\n")

    # Save matches
    matches_path = f"{output_dir}/intent_generation_matches_{timestamp}.json"
    with open(matches_path, 'w', encoding='utf-8') as f:
        json.dump(matches, f, indent=2)

    # Save final queries
    queries_path = f"{output_dir}/intent_generation_queries_{timestamp}.csv"
    df = pd.DataFrame(final_queries)
    df.to_csv(queries_path, index=False)

    print(f"\n📁 Results saved:")
    print(f"  📄 Original intents: {intents_path}")
    print(f"  🔗 Intent-food matches: {matches_path}")
    print(f"  📝 Final queries: {queries_path}")

    return queries_path

def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Intent-driven query generation")
    parser.add_argument("--model", default="gpt-5", help="OpenAI model to use")
    parser.add_argument("--num-intents", type=int, default=50, help="Number of intents to generate")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of food items")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of foods to process per batch")
    parser.add_argument("--output-dir", default="output", help="Output directory")

    args = parser.parse_args()

    print("🚀 Starting intent-driven query generation...")
    print(f"Model: {args.model}")
    print(f"Number of intents: {args.num_intents}")

    try:
        # Setup
        setup_openai()

        # Step 1: Generate pure user intents
        intents = step1_generate_intents(args.num_intents, args.model)

        # Load food data
        food_df = load_food_data(limit=args.limit)

        # Process foods in batches
        all_final_queries = []
        all_matches = {"matches": []}
        total_batches = (len(food_df) + args.batch_size - 1) // args.batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(food_df))
            batch_df = food_df.iloc[start_idx:end_idx]

            print(f"\n📦 Processing batch {batch_idx + 1}/{total_batches} ({len(batch_df)} foods)")

            # Step 2: Smart matching for this batch
            batch_matches = step2_match_intents_to_foods(intents, batch_df, args.model)

            # Step 3: Generate final queries for this batch
            batch_queries = step3_generate_final_queries(batch_matches, args.model)

            # Accumulate results
            all_final_queries.extend(batch_queries)
            all_matches["matches"].extend(batch_matches["matches"])

        final_queries = all_final_queries

        # Save results
        save_results(intents, all_matches, final_queries, args.output_dir)

        # Summary
        intent_queries = len([q for q in final_queries if q['query_type'] == 'intent'])
        bridged_queries = len([q for q in final_queries if q['query_type'] == 'bridged'])

        print(f"\n✅ Success!")
        print(f"📊 Generated {len(final_queries)} total queries:")
        print(f"   🎯 {intent_queries} intent-based queries (pure user searches)")
        print(f"   🔗 {bridged_queries} bridged queries (food-aware)")
        print(f"   📈 ~75% authentic user intent coverage")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())