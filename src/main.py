import os
from data_loader import AirbnbDataLoader
from listing_ranker import ListingRanker
from recommender_evaluator import RecommenderEvaluator, run_evaluation
from pprint import pprint
import pandas as pd
import numpy as np

def run_recommender_evaluation(listings_df, reviews_df, sample_size=20):
    """
    Run the recommender evaluation pipeline
    
    Args:
        listings_df: DataFrame containing listing information
        reviews_df: DataFrame containing review information
        sample_size: Number of users to evaluate
    """
    print("\nRunning recommender evaluation...")
    print(f"Sample size: {sample_size} users")
    
    # Run the evaluation
    results = run_evaluation(
        df_listings=listings_df,
        df_reviews=reviews_df,
        llm_model='phi3',  # You can change this to other models
        sample_size=sample_size
    )
    
    if results is not None:
        print("\nEvaluation Results:")
        print(results)
    else:
        print("Evaluation failed to complete.")

# Example Usage
def main():
    # Initialize data loader
    data_loader = AirbnbDataLoader(data_dir='data/seattle')
    
    # Load data
    print("Loading data...")
    listings_df, reviews_df, _ = data_loader.load_data(city='seattle')
    if listings_df is None or reviews_df is None:
        print("Failed to load data!")
        return
        
    print("Data loaded successfully!")
    print("\nListings columns:", listings_df.columns.tolist())
    print("\nReviews columns:", reviews_df.columns.tolist())
    print("\nReviews data:")
    print("reviews_df.head():", reviews_df.head())

    # Initialize ranker with data
    ranker = ListingRanker(listings_df=listings_df, reviews_df=reviews_df)

    # Example: Get recommendations for a user
    user_id = reviews_df['reviewer_id'].iloc[0]  # Get first user as example
    print(f"\nUser ID: {user_id}")
    print(f"User history: {reviews_df[reviews_df['reviewer_id'] == user_id]['listing_id'].tolist()}")
    user_history = listings_df[listings_df['listing_id'].isin(
        reviews_df[reviews_df['reviewer_id'] == user_id]['listing_id']
    )]
    
    # Get candidate items (excluding user history)
    candidates = listings_df[~listings_df['listing_id'].isin(user_history['listing_id'])]
    
    # Get recommendations
    recommendations = ranker.retrieve_candidates(
        user_history=user_history,
        candidates=candidates,
        interaction_data=reviews_df,
        top_k=5
    )
    
    print("\nTop 5 recommendations:")
    print(recommendations)
    
    # Run the recommender evaluation
    run_recommender_evaluation(listings_df, reviews_df)

if __name__ == '__main__':
    main()
