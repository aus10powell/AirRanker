import os
from data_loader import AirbnbDataLoader
from listing_ranker import ListingRanker
from recommender_evaluator import RecommenderEvaluator, run_evaluation
from pprint import pprint
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import time

def load_or_compute_cache(cache_path, compute_func, *args, **kwargs):
    """Load from cache or compute and save to cache"""
    if os.path.exists(cache_path):
        print(f"Loading from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print(f"Computing and caching: {cache_path}")
    result = compute_func(*args, **kwargs)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)
    return result

def run_recommender_evaluation(listings_df, reviews_df, llm_model='phi4', embedding_model='all-MiniLM-L6-v2', sample_size=20):
    """
    Run the recommender evaluation pipeline
    
    Args:
        listings_df: DataFrame containing listing information
        reviews_df: DataFrame containing review information
        llm_model: Language model to use for LLM ranker
        embedding_model: Sentence transformer model to use for embeddings
        sample_size: Number of users to evaluate
    """
    print("\nRunning recommender evaluation...")
    print(f"Sample size: {sample_size} users")
    print(f"Using LLM model: {llm_model}")
    print(f"Using embedding model: {embedding_model}")
    
    # Run the evaluation
    results = run_evaluation(
        df_listings=listings_df,
        df_reviews=reviews_df,
        llm_model=llm_model,
        embedding_model=embedding_model,
        sample_size=sample_size
    )
    
    if results is not None:
        print("\nEvaluation Results:")
        print(results)
    else:
        print("Evaluation failed to complete.")

def main():
    start_time = time.time()
    
    # Initialize data loader
    data_loader = AirbnbDataLoader(data_dir='data/seattle')
    
    # Load data with caching
    print("Loading data...")
    cache_dir = Path('cache')
    cache_dir.mkdir(exist_ok=True)
    
    listings_cache = cache_dir / "listings_df.pkl"
    reviews_cache = cache_dir / "reviews_df.pkl"
    
    if os.path.exists(listings_cache) and os.path.exists(reviews_cache):
        print("Loading data from cache...")
        with open(listings_cache, 'rb') as f:
            listings_df = pickle.load(f)
        with open(reviews_cache, 'rb') as f:
            reviews_df = pickle.load(f)
    else:
        print("Loading data from source...")
        listings_df, reviews_df, _ = data_loader.load_data(city='seattle')
        if listings_df is not None and reviews_df is not None:
            with open(listings_cache, 'wb') as f:
                pickle.dump(listings_df, f)
            with open(reviews_cache, 'wb') as f:
                pickle.dump(reviews_df, f)
    
    if listings_df is None or reviews_df is None:
        print("Failed to load data!")
        return
        
    print("Data loaded successfully!")
    print(f"Loaded {len(listings_df)} listings and {len(reviews_df)} reviews")
    
    # Initialize ranker with optimized parameters
    ranker = ListingRanker(
        listings_df=listings_df, 
        reviews_df=reviews_df,
        llm_model='phi4',  # Using phi4 for better performance
        max_concurrent_calls=10  # Increased concurrent calls
    )

    # Example: Get recommendations for a user
    user_id = reviews_df['reviewer_id'].iloc[0]
    print(f"\nGetting recommendations for user: {user_id}")
    
    # Get user history (first 2 reviews)
    user_history = listings_df[listings_df['listing_id'].isin(
        reviews_df[reviews_df['reviewer_id'] == user_id]['listing_id'].head(2)
    )]
    
    # Get 10 random candidates (excluding user history)
    candidates = listings_df[~listings_df['listing_id'].isin(user_history['listing_id'])].sample(n=10)
    
    print(f"\nUser history ({len(user_history)} items):")
    print(user_history[['listing_id', 'name', 'price', 'room_type']].to_string())
    
    print(f"\nCandidate items ({len(candidates)} items):")
    print(candidates[['listing_id', 'name', 'price', 'room_type']].to_string())
    
    # Get recommendations with progress tracking
    print("\nComputing recommendations...")
    recommendations = ranker.retrieve_candidates(
        user_history=user_history,
        candidates=candidates,
        interaction_data=reviews_df,
        top_k=5  # Get top 5 recommendations
    )
    
    print("\nTop 5 recommendations:")
    print(recommendations[['listing_id', 'name', 'price', 'room_type', 'score']].to_string())
    
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
