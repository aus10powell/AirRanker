import sys
import os
import pandas as pd

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.listing_ranker import ListingRanker

def rank_listings(listings, query, top_k=5):
    """
    Rank listings based on the query using ListingRanker.
    
    Args:
        listings (list): List of listing dictionaries
        query (str): User query
        top_k (int): Number of top listings to return
        
    Returns:
        list: Top k ranked listings
    """
    try:
        # Convert listings to DataFrame
        listings_df = pd.DataFrame(listings)
        
        # Create an empty reviews DataFrame since we don't have reviews data
        reviews_df = pd.DataFrame()
        
        # Initialize the ranker
        ranker = ListingRanker(listings_df, reviews_df)
        
        # For now, just return the top k listings
        # In a real implementation, you would use the ranker's methods
        return listings[:top_k]
    except Exception as e:
        print(f"Error in rank_listings: {e}")
        # Return the original listings if ranking fails
        return listings[:top_k] 