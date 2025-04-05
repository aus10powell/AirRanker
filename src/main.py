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
import random
import json

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

    print("\n=== Testing Single User Recommendations ===")
    
    # Example: Get recommendations for a user
    user_id = reviews_df['reviewer_id'].iloc[0]
    print(f"\nGetting recommendations for user: {user_id}")
    
    # Get user history (first 2 reviews)
    user_history = listings_df[listings_df['listing_id'].isin(
        reviews_df[reviews_df['reviewer_id'] == user_id]['listing_id'].head(2)
    )]
    
    # TARGETED CANDIDATE SELECTION STRATEGY
    print("\nSelecting targeted candidates based on user preferences...")
    
    # 1. Extract user preferences from history
    avg_price = user_history['price'].mean()
    price_std = user_history['price'].std() if len(user_history) > 1 else 0
    price_range = 0.5  # 50% above and below average price
    min_price = avg_price * (1 - price_range)
    max_price = avg_price * (1 + price_range)
    
    # Get preferred room types
    preferred_room_types = user_history['room_type'].value_counts().index.tolist()
    
    # Get preferred neighborhoods if available
    preferred_neighborhoods = []
    if 'neighbourhood' in user_history.columns:
        preferred_neighborhoods = user_history['neighbourhood'].value_counts().index.tolist()
    
    # Get preferred amenities if available
    preferred_amenities = []
    if 'amenities' in user_history.columns:
        # Extract amenities from history
        all_amenities = []
        for amenities_str in user_history['amenities'].dropna():
            try:
                amenities = json.loads(amenities_str.replace("'", '"'))
                all_amenities.extend(amenities)
            except:
                pass
        # Get most common amenities
        if all_amenities:
            from collections import Counter
            amenity_counts = Counter(all_amenities)
            preferred_amenities = [item for item, count in amenity_counts.most_common(5)]
    
    # Get preferred review scores
    avg_cleanliness = user_history['review_scores_cleanliness'].mean() if 'review_scores_cleanliness' in user_history.columns else None
    avg_location = user_history['review_scores_location'].mean() if 'review_scores_location' in user_history.columns else None
    avg_rating = user_history['review_scores_rating'].mean() if 'review_scores_rating' in user_history.columns else None
    
    # 2. Filter candidates based on preferences
    potential_candidates = listings_df[~listings_df['listing_id'].isin(user_history['listing_id'])]
    
    # Apply initial filters
    filtered_candidates = potential_candidates[
        # Price filter (with handling for NaN values)
        ((potential_candidates['price'] >= min_price) & (potential_candidates['price'] <= max_price)) |
        (potential_candidates['price'].isna())
    ]
    
    # Room type filter
    if preferred_room_types:
        filtered_candidates = filtered_candidates[
            filtered_candidates['room_type'].isin(preferred_room_types)
        ]
    
    # Neighborhood filter if available
    if preferred_neighborhoods and 'neighbourhood' in filtered_candidates.columns:
        filtered_candidates = filtered_candidates[
            filtered_candidates['neighbourhood'].isin(preferred_neighborhoods)
        ]
    
    # 3. If we have too few candidates after filtering, gradually relax constraints
    if len(filtered_candidates) < 20:
        print(f"Only found {len(filtered_candidates)} candidates after strict filtering, relaxing constraints...")
        
        # Relax room type constraint first
        if preferred_room_types:
            filtered_candidates = potential_candidates[
                ((potential_candidates['price'] >= min_price) & (potential_candidates['price'] <= max_price)) |
                (potential_candidates['price'].isna())
            ]
        
        # If still too few, relax price constraint
        if len(filtered_candidates) < 20:
            print("Relaxing price constraints...")
            filtered_candidates = potential_candidates
    
    # 4. Score candidates based on multiple factors
    print(f"Scoring {len(filtered_candidates)} candidates based on multiple factors...")
    
    # Initialize score column
    filtered_candidates['candidate_score'] = 0.0
    
    # Price score (closer to user's average price is better)
    if not filtered_candidates['price'].isna().all():
        filtered_candidates['price_diff'] = abs(filtered_candidates['price'] - avg_price)
        max_price_diff = filtered_candidates['price_diff'].max()
        if max_price_diff > 0:
            filtered_candidates['price_score'] = 1 - (filtered_candidates['price_diff'] / max_price_diff)
        else:
            filtered_candidates['price_score'] = 1.0
        filtered_candidates['candidate_score'] += 0.3 * filtered_candidates['price_score']
    
    # Room type score
    if preferred_room_types and 'room_type' in filtered_candidates.columns:
        filtered_candidates['room_type_score'] = filtered_candidates['room_type'].isin(preferred_room_types).astype(float)
        filtered_candidates['candidate_score'] += 0.2 * filtered_candidates['room_type_score']
    
    # Neighborhood score
    if preferred_neighborhoods and 'neighbourhood' in filtered_candidates.columns:
        filtered_candidates['neighborhood_score'] = filtered_candidates['neighbourhood'].isin(preferred_neighborhoods).astype(float)
        filtered_candidates['candidate_score'] += 0.2 * filtered_candidates['neighborhood_score']
    
    # Review scores
    if avg_cleanliness is not None and 'review_scores_cleanliness' in filtered_candidates.columns:
        filtered_candidates['cleanliness_score'] = filtered_candidates['review_scores_cleanliness'] / 10.0
        filtered_candidates['candidate_score'] += 0.1 * filtered_candidates['cleanliness_score']
    
    if avg_location is not None and 'review_scores_location' in filtered_candidates.columns:
        filtered_candidates['location_score'] = filtered_candidates['review_scores_location'] / 10.0
        filtered_candidates['candidate_score'] += 0.1 * filtered_candidates['location_score']
    
    if avg_rating is not None and 'review_scores_rating' in filtered_candidates.columns:
        filtered_candidates['rating_score'] = filtered_candidates['review_scores_rating'] / 10.0
        filtered_candidates['candidate_score'] += 0.1 * filtered_candidates['rating_score']
    
    # 5. Select final candidates (up to 50 for better diversity)
    # Sort by candidate score and select top candidates
    filtered_candidates = filtered_candidates.sort_values('candidate_score', ascending=False)
    num_candidates = min(50, len(filtered_candidates))
    candidates = filtered_candidates.head(num_candidates)
    
    print(f"\nUser history ({len(user_history)} items):")
    print(user_history[['listing_id', 'name', 'price', 'room_type']].to_string())
    
    print(f"\nSelected {len(candidates)} targeted candidates:")
    print(candidates[['listing_id', 'name', 'price', 'room_type', 'candidate_score']].to_string())
    
    # Get recommendations with progress tracking
    print("\nComputing recommendations...")
    recommendations = ranker.retrieve_candidates(
        user_history=user_history,
        candidates=candidates,
        interaction_data=reviews_df,
        top_k=10  # Get top 10 recommendations
    )
    
    print("\nTop 10 recommendations:")
    print(recommendations[['listing_id', 'name', 'price', 'room_type', 'score']].to_string())
    
    print("\n=== Running Evaluation on Sample Users ===")
    
    # Initialize evaluator
    evaluator = RecommenderEvaluator(ranker, listings_df, reviews_df)
    
    # Run evaluation on a small sample of users
    print("\nEvaluating recommender system on 5 random users...")
    sample_users = random.sample(list(reviews_df['reviewer_id'].unique()), 5)
    
    all_metrics = []
    for user_id in sample_users:
        print(f"\nEvaluating user {user_id}...")
        # Get user's reviews
        user_reviews = reviews_df[reviews_df['reviewer_id'] == user_id].sort_values('date')
        
        # Split into history and holdout
        history_size = max(2, len(user_reviews) // 2)  # Use at least 2 items for history
        history = user_reviews.head(history_size)
        holdout = user_reviews.tail(len(user_reviews) - history_size)
        
        # Get user history listings
        history_listings = listings_df[listings_df['listing_id'].isin(history['listing_id'])]
        holdout_listings = listings_df[listings_df['listing_id'].isin(holdout['listing_id'])]
        
        # Select targeted candidates for evaluation
        # Extract user preferences from history
        avg_price = history_listings['price'].mean()
        price_std = history_listings['price'].std() if len(history_listings) > 1 else 0
        price_range = 0.5  # 50% above and below average price
        min_price = avg_price * (1 - price_range)
        max_price = avg_price * (1 + price_range)
        
        # Get preferred room types
        preferred_room_types = history_listings['room_type'].value_counts().index.tolist()
        
        # Get preferred neighborhoods if available
        preferred_neighborhoods = []
        if 'neighbourhood' in history_listings.columns:
            preferred_neighborhoods = history_listings['neighbourhood'].value_counts().index.tolist()
        
        # Get preferred amenities if available
        preferred_amenities = []
        if 'amenities' in history_listings.columns:
            # Extract amenities from history
            all_amenities = []
            for amenities_str in history_listings['amenities'].dropna():
                try:
                    amenities = json.loads(amenities_str.replace("'", '"'))
                    all_amenities.extend(amenities)
                except:
                    pass
            # Get most common amenities
            if all_amenities:
                from collections import Counter
                amenity_counts = Counter(all_amenities)
                preferred_amenities = [item for item, count in amenity_counts.most_common(5)]
        
        # Get preferred review scores
        avg_cleanliness = history_listings['review_scores_cleanliness'].mean() if 'review_scores_cleanliness' in history_listings.columns else None
        avg_location = history_listings['review_scores_location'].mean() if 'review_scores_location' in history_listings.columns else None
        avg_rating = history_listings['review_scores_rating'].mean() if 'review_scores_rating' in history_listings.columns else None
        
        # Filter candidates based on preferences
        potential_candidates = listings_df[~listings_df['listing_id'].isin(history_listings['listing_id'])]
        
        # Apply initial filters
        filtered_candidates = potential_candidates[
            # Price filter (with handling for NaN values)
            ((potential_candidates['price'] >= min_price) & (potential_candidates['price'] <= max_price)) |
            (potential_candidates['price'].isna())
        ]
        
        # Room type filter
        if preferred_room_types:
            filtered_candidates = filtered_candidates[
                filtered_candidates['room_type'].isin(preferred_room_types)
            ]
        
        # Neighborhood filter if available
        if preferred_neighborhoods and 'neighbourhood' in filtered_candidates.columns:
            filtered_candidates = filtered_candidates[
                filtered_candidates['neighbourhood'].isin(preferred_neighborhoods)
            ]
        
        # If we have too few candidates after filtering, gradually relax constraints
        if len(filtered_candidates) < 50:
            # Relax room type constraint first
            if preferred_room_types:
                filtered_candidates = potential_candidates[
                    ((potential_candidates['price'] >= min_price) & (potential_candidates['price'] <= max_price)) |
                    (potential_candidates['price'].isna())
                ]
            
            # If still too few, relax price constraint
            if len(filtered_candidates) < 50:
                filtered_candidates = potential_candidates
        
        # Score candidates based on multiple factors
        # Initialize score column
        filtered_candidates['candidate_score'] = 0.0
        
        # Price score (closer to user's average price is better)
        if not filtered_candidates['price'].isna().all():
            filtered_candidates['price_diff'] = abs(filtered_candidates['price'] - avg_price)
            max_price_diff = filtered_candidates['price_diff'].max()
            if max_price_diff > 0:
                filtered_candidates['price_score'] = 1 - (filtered_candidates['price_diff'] / max_price_diff)
            else:
                filtered_candidates['price_score'] = 1.0
            filtered_candidates['candidate_score'] += 0.3 * filtered_candidates['price_score']
        
        # Room type score
        if preferred_room_types and 'room_type' in filtered_candidates.columns:
            filtered_candidates['room_type_score'] = filtered_candidates['room_type'].isin(preferred_room_types).astype(float)
            filtered_candidates['candidate_score'] += 0.2 * filtered_candidates['room_type_score']
        
        # Neighborhood score
        if preferred_neighborhoods and 'neighbourhood' in filtered_candidates.columns:
            filtered_candidates['neighborhood_score'] = filtered_candidates['neighbourhood'].isin(preferred_neighborhoods).astype(float)
            filtered_candidates['candidate_score'] += 0.2 * filtered_candidates['neighborhood_score']
        
        # Review scores
        if avg_cleanliness is not None and 'review_scores_cleanliness' in filtered_candidates.columns:
            filtered_candidates['cleanliness_score'] = filtered_candidates['review_scores_cleanliness'] / 10.0
            filtered_candidates['candidate_score'] += 0.1 * filtered_candidates['cleanliness_score']
        
        if avg_location is not None and 'review_scores_location' in filtered_candidates.columns:
            filtered_candidates['location_score'] = filtered_candidates['review_scores_location'] / 10.0
            filtered_candidates['candidate_score'] += 0.1 * filtered_candidates['location_score']
        
        if avg_rating is not None and 'review_scores_rating' in filtered_candidates.columns:
            filtered_candidates['rating_score'] = filtered_candidates['review_scores_rating'] / 10.0
            filtered_candidates['candidate_score'] += 0.1 * filtered_candidates['rating_score']
        
        # Select final candidates
        # Sort by candidate score and select top candidates
        filtered_candidates = filtered_candidates.sort_values('candidate_score', ascending=False)
        
        # Ensure we have enough candidates for evaluation with k=100
        # If we have fewer than k candidates, we'll use all available candidates
        k_value = 100  # This should match the k parameter in evaluate_user
        num_candidates = max(k_value, min(100, len(filtered_candidates)))
        eval_candidates = filtered_candidates.head(num_candidates)
        
        print(f"Selected {len(eval_candidates)} candidates for evaluation with k={k_value}")
        
        # Evaluate
        metrics, _ = evaluator.evaluate_user(user_id, history_listings, holdout_listings, k=k_value, candidates=eval_candidates)
        print("Metrics:", metrics)
        all_metrics.append(metrics)
    
    # Compute average metrics
    print("\nAverage metrics across all users:")
    avg_metrics = {}
    for metric in all_metrics[0].keys():
        values = [m[metric] for m in all_metrics if m[metric] is not None]
        if values:
            avg_metrics[metric] = sum(values) / len(values)
            print(f"{metric}: {avg_metrics[metric]:.3f}")
    
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
