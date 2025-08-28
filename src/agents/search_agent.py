import os
import json
from typing import List
import ollama
import sys
import os
import pandas as pd
from pprint import pprint

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.listing_ranker import ListingRanker
from src.agents.tools.listing_filter import filter_listings
from src.agents.tools.ranker_tool import rank_listings

# Load Seattle listings from processed data
def load_seattle_listings():
    listings_df = pd.read_parquet('../../data/seattle/listings.parquet')
    return listings_df.to_dict(orient='records')

SEATTLE_LISTINGS = load_seattle_listings()

def ask_agent(user_query: str, listings=SEATTLE_LISTINGS) -> List[dict]:
    """
    Main entry point for the agent. Handles user query, parses it, filters, ranks, and returns top listings.
    """
    print(f"USER: {user_query}\n")

    # Step 1: Ask LLM to extract filters and preferences
    filter_prompt = f"""
You are a helpful assistant helping users find Airbnb listings.
Extract structured preferences from the following user query.

Return a JSON object with:
- keywords: list of relevant keywords
- location: optional location or neighborhood
- price_limit: if mentioned
- amenities: list if mentioned

User query: "{user_query}"

IMPORTANT: Your response must be a valid JSON object only, with no additional text.
"""
    # Use Ollama with llama2 model instead of OpenAI
    try:
        filter_response = ollama.chat(model='llama3.2', messages=[
            {"role": "user", "content": filter_prompt}
        ])
        
        # Extract the content from the response
        response_content = filter_response['message']['content']
        
        # Try to parse the JSON response
        try:
            parsed = json.loads(response_content)
            print(f"Extracted filters: {json.dumps(parsed, indent=2)}")
        except json.JSONDecodeError as e:
            print(f"Failed to parse model response: {e}")
            print(f"Raw response: {response_content}")
            # Provide a default structure if parsing fails
            parsed = {
                "keywords": [],
                "location": None,
                "price_limit": None,
                "amenities": []
            }
    except Exception as e:
        print(f"Error calling Ollama API: {e}")
        # Provide a default structure if API call fails
        parsed = {
            "keywords": [],
            "location": None,
            "price_limit": None,
            "amenities": []
        }

    # Step 2: Filter listings
    filtered = filter_listings(listings, parsed)

    if not filtered:
        print("No listings matched filters.")
        return []

    # Step 3: Rank listings with ListingRanker
    top_listings = rank_listings(filtered, query=user_query, top_k=5)

    # Step 4: Return listing summaries
    for i, listing in enumerate(top_listings, 1):
        print(f"{i}. {listing['name']} — ${listing.get('price', '?')}")
        print(f"   {listing['description'][:150]}...\n")

    return top_listings

if __name__ == "__main__":
    user_query = "I'm looking for a place to stay in Seattle with a pool and a hot tub."
    top_listings = ask_agent(user_query)
    
    for listing in top_listings:
        print(f"{listing['name']} — ${listing.get('price', '?')}")
    print(top_listings,type(top_listings))
