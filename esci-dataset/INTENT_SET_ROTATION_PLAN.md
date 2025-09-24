# Intent Set Rotation System - Implementation Plan

## Overview
Enhance the intent generation approach to pre-generate 10 diverse intent sets and rotate through them during step 2 (matching) and step 3 (query generation), providing maximum diversity while maintaining efficiency.

## Current Analysis
- **Current approach**: Generates 1 intent set per run (typically 50 intents) and uses it for all food batches
- **Current prompt**: Uses personas like "Lazy Student", "Busy Professional", "Party Planner", "Health-Conscious"
- **Current flow**: Step 1 → Step 2 (batched) → Step 3 (batched)

## Proposed Enhancement

### 1. Pre-Generate 10 Themed Intent Sets
Create a new script `generate_intent_sets.py` that generates 10 diverse intent sets with different themes:

**Intent Set Themes:**
1. **Quick & Convenient** - Fast food, convenience, time-pressed scenarios
2. **Health & Wellness** - Nutrition, diet-conscious, fitness-oriented
3. **Comfort & Indulgent** - Comfort food, treats, indulgent cravings
4. **International Cuisine** - Ethnic foods, cultural cravings, travel-inspired
5. **Social & Sharing** - Party food, group meals, social dining
6. **Budget Conscious** - Cheap eats, value meals, student-friendly
7. **Premium & Gourmet** - High-end, special occasions, quality-focused
8. **Dietary Restrictions** - Vegan, gluten-free, allergy-conscious
9. **Mood & Occasion** - Breakfast, late-night, hangover, celebration
10. **Seasonal & Fresh** - Seasonal ingredients, fresh produce, weather-based

### 2. Modify Intent Generation Approach
Update `intent_generation_approach.py` to:
- Accept pre-generated intent sets instead of generating new ones
- Rotate through intent sets during batch processing
- Track which intent set was used for each batch

### 3. Update MLflow Wrapper
Enhance `mlflow_intent_generation.py` to:
- Support intent set rotation parameters
- Log which intent sets were used
- Track intent set usage across batches

### 4. Key Benefits
- **Diversity**: 10x more diverse intent coverage across runs
- **Consistency**: Reproducible intent sets for comparison
- **Efficiency**: No repeated intent generation during large runs
- **Thematic Coverage**: Comprehensive coverage of user scenarios
- **Scalability**: Works seamlessly with existing batch processing

### 5. Implementation Structure
```python
# New files to create:
- src/data_generation/generate_intent_sets.py
- prompts/intent_generation/themes/01_quick_convenient.txt
- prompts/intent_generation/themes/02_health_wellness.txt
- prompts/intent_generation/themes/03_comfort_indulgent.txt
- prompts/intent_generation/themes/04_international_cuisine.txt
- prompts/intent_generation/themes/05_social_sharing.txt
- prompts/intent_generation/themes/06_budget_conscious.txt
- prompts/intent_generation/themes/07_premium_gourmet.txt
- prompts/intent_generation/themes/08_dietary_restrictions.txt
- prompts/intent_generation/themes/09_mood_occasion.txt
- prompts/intent_generation/themes/10_seasonal_fresh.txt

# Files to modify:
- src/data_generation/intent_generation_approach.py
- src/mlflow_wrapper/mlflow_intent_generation.py

# New CLI parameters:
--use-intent-sets path/to/intent/sets/
--intent-rotation-frequency 5  # Change intent set every N batches
```

### 6. Usage Examples
```bash
# Generate 10 intent sets once
python src/data_generation/generate_intent_sets.py --output-dir intent_sets/

# Use rotating intent sets in processing
python src/mlflow_wrapper/mlflow_intent_generation.py \
  --limit 1000 --batch-size 100 \
  --use-intent-sets intent_sets/ \
  --intent-rotation-frequency 3
```

### 7. Rotation Logic Pseudo-code
```python
def run_intent_generation_with_rotation():
    intent_sets = load_all_intent_sets("intent_sets/")  # Load 10 sets
    total_foods = get_total_food_count()
    batch_size = 100
    rotation_frequency = 5  # Change every 5 batches

    for batch_num in range(0, total_foods, batch_size):
        # Rotate intent set every N batches
        intent_set_index = (batch_num // batch_size // rotation_frequency) % len(intent_sets)
        current_intents = intent_sets[intent_set_index]

        # Process batch with current intent set
        batch_results = process_batch(batch_foods, current_intents)
        log_intent_set_usage(intent_set_index, batch_num)
```

### 8. Expected Impact
- **For 25,574 foods with batch_size=100, rotation_frequency=5:**
  - 256 batches total
  - Each intent set used ~25-26 times
  - Maximum thematic diversity across the entire dataset
  - Reproducible and trackable intent usage

### 9. Integration with Dietary Evaluations
The system can leverage existing dietary evaluation functions like:
- `get_dietary_column_mapping()` for dietary restriction themes
- `get_all_dietary_combinations()` for comprehensive dietary coverage
- Existing dietary flags for themed intent generation

## Next Steps
1. Create themed prompt templates for each of the 10 intent sets
2. Implement `generate_intent_sets.py` script
3. Modify intent generation approach to support rotation
4. Update MLflow wrapper with new parameters
5. Test with small batches before full-scale deployment

This enhancement provides maximum diversity while maintaining the existing proven workflow!