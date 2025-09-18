import pandas as pd
from rapidfuzz import process
from tqdm import tqdm

# Load datasets
recipes_df = pd.read_csv("3A2M.csv")  # Recipes dataset
food_df    = pd.read_csv("MM-Food-100K.csv")  # Hugging Face dataset

# Take 5k sample from both (your code had 50k comment but samples are 5k)
recipes_sample = recipes_df.sample(n=50000, random_state=42)
food_sample    = food_df.sample(n=50000, random_state=42)

print("Recipes dataset:", recipes_sample.shape)
print("Food dataset:", food_sample.shape)

# prepare choices as a list and lowercase for more consistent fuzzy matches
food_names = food_sample['dish_name'].dropna().unique()
food_names = [str(n).strip().lower() for n in food_names]

matches = []
# iterate over unique recipe titles (lowercased)
titles = recipes_sample['title'].dropna().unique()
for i, title in enumerate(tqdm(titles, desc="Matching titles")):
    t = str(title).strip().lower()
    # assign result then check for None
    match = process.extractOne(t, food_names, score_cutoff=95)
    if match is not None:
        # match is a tuple like (best_match, score, index)
        best_match, score, _ = match
        matches.append((title, best_match, score))
        # debug print every 500 matches processed (not every iteration)
        if len(matches) % 500 == 0:
            print(f"Found {len(matches)} matches so far. Last: {title} <-> {best_match} (score={score})")

# Build Final Dataset (sample for check)
final_data = []
for title, dish_name, score in matches:  # first 20 matches
    # find the first matching recipe row (original title match)
    rec_row = recipes_sample[recipes_sample['title'].str.strip().str.lower() == str(title).strip().lower()].iloc[0]
    # find the food row by dish_name (we stored lowercase names)
    food_row = food_sample[food_sample['dish_name'].str.strip().str.lower() == dish_name].iloc[0]

    final_data.append({
        "Dish Name (Recipe)": title,
        "Dish Name (MMFood)": dish_name,
        "Recipe": rec_row.get('directions', None),
        "Ingredients(Recipe)": rec_row.get('NER', None),
        "Ingredients(MMFood)": food_row.get('ingredients', None),
        "Nutritional Profile": food_row.get('nutritional_profile', None),
        "Cooking Method": food_row.get('cooking_method', None),
        "Food Type": food_row.get('food_type', None),
        "Genre": rec_row.get('genre', None),
    })

final_df = pd.DataFrame(final_data)
print("Final dataset sample (first 10 rows):")
print(final_df.head(10))
final_df.to_csv('final_df.csv')