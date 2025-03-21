import ollama
import itertools
import json
import pandas as pd
from pprint import pprint

class ListingRanker:
    def __init__(self, model='phi4'):
        self.llm = model
    
    def create_pairwise_ranking_prompt(self, candidate_listings, user_history):
        """
        Create a structured prompt for pairwise ranking of listings
        
        Args:
            candidate_listings (pd.DataFrame): DataFrame of candidate listings
            user_history (pd.DataFrame): DataFrame of user's historical listings
        
        Returns:
            str: Formatted prompt for pairwise ranking
        """
        # Prepare user history context with more details
        history_context = "User's Previous Bookings and Reviews:\n"
   
        for _, listing in user_history.iterrows():
            history_context += (
                f"- Listing {listing['listing_id']}:\n"
                f"  Comment: {listing.get('comments', 'No comment')}\n"
                f"  Date: {listing.get('date', 'Unknown')}\n"
            )
        
        # Prepare candidate listings with more features
        candidates_context = "\nCandidate Listings to Rank:\n"
        for i, (_, listing) in enumerate(candidate_listings.iterrows(), 1):
            candidates_context += (
                f"[{i}] listing_id: {listing['id']}\n"
                f"    name: {listing['name']}\n"
                f"    price: ${listing['price']}/night\n"
                f"    room_type: {listing.get('room_type', 'Unknown')}\n"
                f"    location_score: {listing.get('review_scores_location', 'Unknown')}\n"
                f"    cleanliness_score: {listing.get('review_scores_cleanliness', 'Unknown')}\n"
                f"    overall_rating: {listing.get('review_scores_rating', 'Unknown')}\n"
            )
        
        # Construct full prompt with explicit instructions
        prompt = f"""You are an AI assistant ranking vacation rental listings based on user preferences.

{history_context}

{candidates_context}

Task: Analyze the user's booking history and rank the candidate listings based on:
1. Similarity to previous bookings (room type, price range, location ratings)
2. Overall listing quality (ratings, reviews)
3. Value for money (price vs. ratings)

IMPORTANT: You must respond ONLY with a JSON object in the following format:
{{
    "ranking": "[1] > [2] > [3] > ..."
}}

The numbers in brackets represent the listing numbers from the candidate list above.
Do not include any other text or explanation in your response.
Example valid response: {{"ranking": "[1] > [3] > [2] > [4]"}}
"""
        return prompt
    
    def generate_response(self, prompt, max_chunks=100):
        """
        Generate response using the LLM model   
        """
        op = "generate_response"
        try:
            # Stream response from ollama
            response = ""
            # Correctly formatted message with role and content
            chunk_counter = 0
            for chunk in ollama.chat(
                model=self.llm, 
                messages=[{"role": "user", "content": prompt}],
                stream=True
            ):
                chunk_counter += 1
                if chunk_counter  % 100 == 0:
                    print(f"Chunk {chunk_counter} received")
                if chunk_counter > max_chunks:
                    break
                response += chunk['message']['content']
            return response
        except Exception as e:
            print(f"op={op}. Error: {str(e)}")
            return None

    def pairwise_llm_ranking(self, candidate_listings, user_history):
        """
        Perform pairwise ranking using LLM
        
        Args:
            candidate_listings (pd.DataFrame): DataFrame of candidate listings
            user_history (pd.DataFrame): DataFrame of user's historical listings
        
        Returns:
            list: Ranked listing identifiers
        """
        op = "pairwise_llm_ranking"
        
        if len(candidate_listings) == 0:
            return []
            
        if len(candidate_listings) == 1:
            return [1]
            
        # Create prompt
        prompt = self.create_pairwise_ranking_prompt(candidate_listings, user_history)
        
        # Get LLM response with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.generate_response(prompt)
                
                # Clean the response - remove any non-JSON text
                response = response.strip()
                if response.find('{') >= 0:
                    response = response[response.find('{'):response.rfind('}')+1]
                
                # Parse JSON response
                ranking_data = json.loads(response)
                ranking_str = ranking_data.get('ranking', '')
                
                # Parse ranking string and validate
                ranked_listings = [
                    int(item.strip('[]')) 
                    for item in ranking_str.split('>')
                ]
                
                # Validate the ranking
                expected_numbers = set(range(1, len(candidate_listings) + 1))
                if set(ranked_listings) == expected_numbers:
                    return ranked_listings
                    
            except Exception as e:
                print(f"Ranking attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    print("All ranking attempts failed, falling back to default ordering")
                    return list(range(1, len(candidate_listings) + 1))
                    
        return list(range(1, len(candidate_listings) + 1))
    
    def rank_listings(self, candidate_listings, user_history):
        """
        Main ranking method
        
        Args:
            candidate_listings (pd.DataFrame): DataFrame of candidate listings
            user_history (pd.DataFrame): DataFrame of user's historical listings
        
        Returns:
            pd.DataFrame: Ranked listings
        """
        # Perform pairwise ranking
        print("user history:", user_history)
        ranked_indices = self.pairwise_llm_ranking(candidate_listings, user_history)
        
        # Reorder listings based on ranking
        ranked_listings = candidate_listings.iloc[
            [idx - 1 for idx in ranked_indices]
        ]
        
        return ranked_listings