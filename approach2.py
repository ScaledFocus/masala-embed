import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def create_merged_food_dataset():
    print("Loading CSV files...")
    mm_food_df = pd.read_csv('MM-Food-100K.csv')
    a2m_df = pd.read_csv('3A2M.csv')
    
    print(f"MM Food dataset shape: {mm_food_df.shape}")
    print(f"3A2M dataset shape: {a2m_df.shape}")
    
    mm_food_names = mm_food_df['dish_name'].fillna('').astype(str).tolist()
    a2m_names = a2m_df['title'].fillna('').astype(str).tolist()
    
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Generating embeddings for MM Food dishes...")
    mm_embeddings = []
    for name in tqdm(mm_food_names, desc="MM Food embeddings"):
        embedding = model.encode([name])[0]
        mm_embeddings.append(embedding)
    mm_embeddings = np.array(mm_embeddings)
    
    print("Generating embeddings for 3A2M titles...")
    a2m_embeddings = []
    for name in tqdm(a2m_names, desc="3A2M embeddings"):
        embedding = model.encode([name])[0]
        a2m_embeddings.append(embedding)
    a2m_embeddings = np.array(a2m_embeddings)
    
    print("Finding matches with similarity threshold 0.99...")
    matches = []
    threshold = 0.99
    
    for i, a2m_embedding in enumerate(tqdm(a2m_embeddings, desc="Finding matches")):
        similarities = cosine_similarity([a2m_embedding], mm_embeddings)[0]
        max_similarity_idx = np.argmax(similarities)
        max_similarity = similarities[max_similarity_idx]
        
        if max_similarity >= threshold:
            matches.append({
                'a2m_idx': i,
                'mm_idx': max_similarity_idx,
                'similarity': max_similarity,
                'a2m_title': a2m_names[i],
                'mm_dish_name': mm_food_names[max_similarity_idx]
            })
    
    print(f"Found {len(matches)} matches above threshold {threshold}")
    
    print("Creating final merged dataset...")
    final_rows = []
    
    for match in tqdm(matches, desc="Creating merged rows"):
        a2m_row = a2m_df.iloc[match['a2m_idx']].to_dict()
        mm_row = mm_food_df.iloc[match['mm_idx']].to_dict()
        
        merged_row = {}
        for col, val in a2m_row.items():
            merged_row[f'3A2M_{col}'] = val
        for col, val in mm_row.items():
            merged_row[f'MM_Food_{col}'] = val
        
        merged_row['similarity_score'] = match['similarity']
        final_rows.append(merged_row)
    
    final_df = pd.DataFrame(final_rows)
    
    print(f"Final dataset shape: {final_df.shape}")
    final_df.to_csv('merged_food_dataset.csv', index=False)
    print("Saved merged dataset to 'merged_food_dataset.csv'")
    
    return final_df

if __name__ == "__main__":
    merged_df = create_merged_food_dataset()
