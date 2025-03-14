from listing_ranker import ListingRanker
from pprint import pprint
import pandas as pd

# Example Usage
def main():

    import os

    DATA_DIR = '../data/'

    # List contents of the data directory
    if os.path.exists(DATA_DIR):
        files = os.listdir(DATA_DIR)
        print("Files in data directory:")
        for file in files:
            print(f" - {file}")
    else:
        print(f"Directory {DATA_DIR} does not exist")


    # Load data
    reviews_df = pd.read_csv(os.path.join(DATA_DIR, 'airbnb_oakland_reviews.csv'))
    listings_df = pd.read_csv(os.path.join(DATA_DIR,'airbnb_oakland_listings.csv'))

    print("Data loaded successfully!")
    print("Reviews data:")
    print("reviews_df.head():",reviews_df.head())
    
    # Assume we have a user's history and candidate listings
    user_history = reviews_df[reviews_df['id'] == 21720882]
    candidate_listings = listings_df#.head(10)
    
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
    for idx, (_, listing) in enumerate(ranked_recommendations[:10].iterrows(), 1,):
        print(f"\nRank {idx}:")
        print(f"  Name: {listing['name']}")
        print(f"  Price: ${listing['price']}/night")
        print(f"  URL: {listing['listing_url']}")

if __name__ == '__main__':
    main()
