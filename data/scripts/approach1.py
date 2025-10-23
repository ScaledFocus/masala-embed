import os
import pickle
import tempfile
from pathlib import Path
import pandas as pd
from rapidfuzz import process
from tqdm import tqdm
import signal

RECIPES_CSV = "3A2M.csv"
FOOD_CSV    = "MM-Food-100K.csv"

MATCH_CHECKPOINT = "matches_checkpoint.pkl"
PROGRESS_STATE   = "progress_state.pkl"
FINAL_CSV        = "final_df.csv"
SAVE_EVERY = 500
ATOMIC_TMP = ".tmp_save"
SCORE_CUTOFF = 95

def atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ATOMIC_TMP)
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)

def atomic_save_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(path.suffix + ATOMIC_TMP)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def load_pickle_if_exists(path: Path, default):
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return default

shutdown = False
def _handle_signal(sig, frame):
    global shutdown
    shutdown = True
signal.signal(signal.SIGINT, _handle_signal)

recipes_df = pd.read_csv(RECIPES_CSV)
food_df    = pd.read_csv(FOOD_CSV)

recipes_sample = recipes_df
food_sample    = food_df

print("Recipes dataset:", recipes_sample.shape)
print("Food dataset:", food_sample.shape)

food_names = food_sample['dish_name'].dropna().unique()
food_names = [str(n).strip().lower() for n in food_names]

matches_path = Path(MATCH_CHECKPOINT)
state_path   = Path(PROGRESS_STATE)
final_path   = Path(FINAL_CSV)

matches = load_pickle_if_exists(matches_path, default=[])
state = load_pickle_if_exists(state_path, default={"last_idx": 0})

start_idx = state.get("last_idx", 0)
print(f"Resuming matching from index {start_idx}. Already found {len(matches)} matches.")

titles = recipes_sample['title'].dropna().unique()
total_titles = len(titles)

for offset, raw_title in enumerate(titles[start_idx:], start=start_idx):
    if shutdown:
        print("Shutdown requested; saving state and exiting matching loop.")
        state['last_idx'] = offset
        atomic_save(state, state_path)
        atomic_save(matches, matches_path)
        break

    title = str(raw_title).strip().lower()
    match = process.extractOne(title, food_names, score_cutoff=SCORE_CUTOFF)
    if match is not None:
        best_match, score, _ = match
        matches.append((raw_title, best_match, score))

        if len(matches) % SAVE_EVERY == 0:
            print(f"[Checkpoint] Found {len(matches)} matches. Saving...")
            state['last_idx'] = offset + 1
            atomic_save(matches, matches_path)
            atomic_save(state, state_path)

else:
    state['last_idx'] = total_titles
    atomic_save(matches, matches_path)
    atomic_save(state, state_path)
    print("Completed matching loop. Saved final matches checkpoint.")

matches = load_pickle_if_exists(matches_path, default=matches)
print(f"Total matches to process into final dataset: {len(matches)}")

if final_path.exists():
    existing_df = pd.read_csv(final_path)
    processed_count = len(existing_df)
    final_data = existing_df.to_dict(orient="records")
    print(f"Found existing final CSV with {processed_count} rows. Resuming from there.")
else:
    final_data = []
    processed_count = 0

for idx, (title, dish_name_lower, score) in enumerate(tqdm(matches, desc="Building final dataset"), start=0):
    if idx < processed_count:
        continue

    rec_rows = recipes_sample[recipes_sample['title'].str.strip().str.lower() == str(title).strip().lower()]
    if rec_rows.empty:
        recipe_row = {}
    else:
        recipe_row = rec_rows.iloc[0]

    food_rows = food_sample[food_sample['dish_name'].str.strip().str.lower() == dish_name_lower]
    if food_rows.empty:
        food_row = {}
    else:
        food_row = food_rows.iloc[0]

    raw_url = food_row.get('image_url', None) if isinstance(food_row, pd.Series) else None
    image_file = raw_url.replace("https://file.b18a.io/", "") if raw_url else None

    final_data.append({
        "dish_name(Recipe)": title,
        "dish_name(MMFood)": dish_name_lower,
        "file_path": image_file,
        "recipe": recipe_row.get('directions', None) if isinstance(recipe_row, pd.Series) else None,
        "ingredients(Recipe)": recipe_row.get('NER', None) if isinstance(recipe_row, pd.Series) else None,
        "ingredients(MMFood)": food_row.get('ingredients', None) if isinstance(food_row, pd.Series) else None,
        "nutritional_profile": food_row.get('nutritional_profile', None) if isinstance(food_row, pd.Series) else None,
        "cooking_method": food_row.get('cooking_method', None) if isinstance(food_row, pd.Series) else None,
        "food_type": food_row.get('food_type', None) if isinstance(food_row, pd.Series) else None,
        "genre": recipe_row.get('genre', None) if isinstance(recipe_row, pd.Series) else None,
        "score": score
    })

    if (idx + 1) % SAVE_EVERY == 0:
        print(f"[Final CSV checkpoint] Saving {len(final_data)} rows to {FINAL_CSV}")
        df_partial = pd.DataFrame(final_data)
        atomic_save_csv(df_partial, final_path)

    if shutdown:
        print("Shutdown requested; saving final CSV and exiting.")
        df_partial = pd.DataFrame(final_data)
        atomic_save_csv(df_partial, final_path)
        break

df_final = pd.DataFrame(final_data)
atomic_save_csv(df_final, final_path)
print(f"Final dataset saved to {FINAL_CSV} ({len(df_final)} rows).")
