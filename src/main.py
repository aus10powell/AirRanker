import os
from data_loader import AirbnbDataLoader
from listing_ranker import ListingRanker
from pprint import pprint
import pandas as pd
import numpy as np

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

if __name__ == '__main__':
    main()
