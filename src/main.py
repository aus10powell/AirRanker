from listing_ranker import ListingRanker
from data_loader import AirbnbDataLoader
from pprint import pprint
import pandas as pd
import os

# Example Usage
def main():
    DATA_DIR = 'data/seattle'

    # List contents of the data directory
    if os.path.exists(DATA_DIR):
        files = os.listdir(DATA_DIR)
        print("Files in data directory:")
        for file in files:
            print(f" - {file}")
    else:
        print(f"Directory {DATA_DIR} does not exist")

    # Initialize data loader
    data_loader = AirbnbDataLoader(data_dir=DATA_DIR)

    # Load data using the data loader
    listings_df, reviews_df, calendar_df = data_loader.load_data(city='seattle')

    if listings_df is None or reviews_df is None:
        print("Failed to load data!")
        return

    print("Data loaded successfully!")
    print("Reviews data:")
    print("reviews_df.head():", reviews_df.head())
    
    # Assume we have a user's history and candidate listings
    # Using a different user ID since we're now using Seattle data
    user_history = reviews_df[reviews_df['reviewer_id'] == 166478]  # Example user from Seattle dataset
    candidate_listings = listings_df
    
    # Initialize ranker
    ranker = ListingRanker(model='phi3')
    
    # Get ranked listings
    ranked_recommendations = ranker.rank_listings(
        candidate_listings, 
        user_history
    )
    print(f"Number of ranked recommendations: {len(ranked_recommendations)}")

    print("Ranked Recommendations:")
    #pprint(ranked_recommendations)

    # Print the first 10 ranked listings by their index
    for idx, (_, listing) in enumerate(ranked_recommendations[:10].iterrows(), 1):
        print(f"\nRank {idx}:")
        print(f"  Name: {listing['name']}")
        print(f"  Price: ${listing['price']}/night")
        print(f"  URL: {listing['listing_url']}")

if __name__ == '__main__':
    main()
