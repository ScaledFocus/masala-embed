import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def create_faiss_food_merger():
    print("Loading CSV files...")
    mm_food_df = pd.read_csv('MM-Food-100K.csv')
    a2m_df = pd.read_csv('3A2M.csv')
    
    print(f"MM Food dataset shape: {mm_food_df.shape}")
    print(f"3A2M dataset shape: {a2m_df.shape}")
    
    mm_food_names = mm_food_df['dish_name'].fillna('').astype(str).tolist()
    a2m_names = a2m_df['title'].fillna('').astype(str).tolist()
    
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_dim = 384
    
    print("Creating embeddings for MM Food dishes...")
    mm_embeddings = []
    for name in tqdm(mm_food_names, desc="MM Food embeddings"):
        embedding = model.encode(name)
        mm_embeddings.append(embedding)
    
    mm_embeddings = np.array(mm_embeddings).astype('float32')
    faiss.normalize_L2(mm_embeddings)
    
    print("Building FAISS index for MM Food...")
    mm_index = faiss.IndexFlatIP(embedding_dim)
    mm_index.add(mm_embeddings)
    
    print("Saving MM Food FAISS index...")
    faiss.write_index(mm_index, 'mm_food_index.faiss')
    mm_metadata = {
        'names': mm_food_names,
        'dataframe_indices': list(range(len(mm_food_names)))
    }
    with open('mm_food_metadata.pkl', 'wb') as f:
        pickle.dump(mm_metadata, f)
    
    print("Creating embeddings for 3A2M titles...")
    a2m_embeddings = []
    for name in tqdm(a2m_names, desc="3A2M embeddings"):
        embedding = model.encode(name)
        a2m_embeddings.append(embedding)
    
    a2m_embeddings = np.array(a2m_embeddings).astype('float32')
    faiss.normalize_L2(a2m_embeddings)
    
    print("Building FAISS index for 3A2M...")
    a2m_index = faiss.IndexFlatIP(embedding_dim)
    a2m_index.add(a2m_embeddings)
    
    print("Saving 3A2M FAISS index...")
    faiss.write_index(a2m_index, 'a2m_index.faiss')
    a2m_metadata = {
        'names': a2m_names,
        'dataframe_indices': list(range(len(a2m_names)))
    }
    with open('a2m_metadata.pkl', 'wb') as f:
        pickle.dump(a2m_metadata, f)
    
    print("Finding closest matches for each 3A2M item...")
    final_rows = []
    threshold = 0.99
    matches_above_threshold = 0
    
    batch_size = 100
    for i in tqdm(range(0, len(a2m_embeddings), batch_size), desc="Processing matches"):
        batch_end = min(i + batch_size, len(a2m_embeddings))
        batch_embeddings = a2m_embeddings[i:batch_end]
        
        similarities, indices = mm_index.search(batch_embeddings, 1)
        
        for j, (similarity, mm_idx) in enumerate(zip(similarities, indices)):
            original_a2m_idx = i + j
            similarity_score = float(similarity[0])
            mm_match_idx = int(mm_idx[0])
            
            if similarity_score >= threshold:
                matches_above_threshold += 1
            
            a2m_row = a2m_df.iloc[original_a2m_idx].to_dict()
            mm_row = mm_food_df.iloc[mm_match_idx].to_dict()
            
            merged_row = {}
            
            for col, val in a2m_row.items():
                merged_row[f'3A2M_{col}'] = val
            
            for col, val in mm_row.items():
                merged_row[f'MM_Food_{col}'] = val
            
            merged_row['similarity_score'] = similarity_score
            merged_row['above_threshold'] = similarity_score >= threshold
            
            final_rows.append(merged_row)
    
    print(f"Total matches processed: {len(final_rows)}")
    print(f"Matches above threshold {threshold}: {matches_above_threshold}")
    
    final_df = pd.DataFrame(final_rows)
    
    print(f"Final dataset shape: {final_df.shape}")
    print(f"Columns in final dataset: {len(final_df.columns)}")
    
    final_df.to_csv('merged_food_dataset_faiss.csv', index=False)
    print("Saved complete merged dataset to 'merged_food_dataset_faiss.csv'")
    
    high_similarity_df = final_df[final_df['above_threshold'] == True]
    if len(high_similarity_df) > 0:
        high_similarity_df.to_csv('high_similarity_matches.csv', index=False)
        print(f"Saved {len(high_similarity_df)} high similarity matches to 'high_similarity_matches.csv'")
    
    return final_df

def search_food_similarity(query_food_name, dataset='mm_food', top_k=5):    
    if dataset == 'mm_food':
        index_file = 'mm_food_index.faiss'
        metadata_file = 'mm_food_metadata.pkl'
    else:
        index_file = 'a2m_index.faiss'
        metadata_file = 'a2m_metadata.pkl'
    
    if not os.path.exists(index_file):
        print(f"Index file {index_file} not found. Run create_faiss_food_merger() first.")
        return None
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index(index_file)
    
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    
    query_embedding = model.encode(query_food_name).astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_embedding)
    
    similarities, indices = index.search(query_embedding, top_k)
    
    results = []
    for sim, idx in zip(similarities[0], indices[0]):
        results.append({
            'food_name': metadata['names'][idx],
            'similarity': float(sim),
            'index': int(idx)
        })
    
    return results

if __name__ == "__main__":
    merged_df = create_faiss_food_merger()
    
    print("\nSample of merged data:")
    if len(merged_df) > 0:
        print(merged_df[['3A2M_title', 'MM_Food_dish_name', 'similarity_score', 'above_threshold']].head())
    
    print(f"\nTotal rows in final dataset: {len(merged_df)}")
    print(f"High similarity matches (>= 0.99): {len(merged_df[merged_df['above_threshold'] == True])}")
    print(f"Average similarity score: {merged_df['similarity_score'].mean():.4f}")
