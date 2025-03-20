import ollama
import itertools
import json
import pandas as pd
from pprint import pprint

class ListingRanker:
    def __init__(self, model='phi3'):
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
        # Prepare user history context
        history_context = "User's Previous Bookings:\n"
   
        for _, listing in user_history.iterrows():
            history_context += f"- Listing {listing['listing_id']}: \n"
        
        # Prepare candidate listings
        candidates_context = "\nCandidate Listings:\n"
        for i, (_, listing) in enumerate(candidate_listings.iterrows(), 1):
            candidates_context += (
                f"[{i}] listing_id: {listing['id']}\n"
                f"    listing_url: {listing['listing_url']}\n"
                f"    name: {listing['name']}\n"
                f"    Location: {listing['review_scores_location']}\n"
                f"    Price: ${listing['price']}/night\n"
                f"    review_scores_cleanliness: {listing['review_scores_cleanliness']}\n"
            )
        
        # Construct full prompt
        prompt = f"""
        You are an AI assistant helping to rank vacation rental listings based on a user's preferences.

        {history_context}

        {candidates_context}

        Task: Perform a pairwise comparison of the candidate listings. 
        For each pair of listings, determine which one is more likely to be preferred 
        by the user based on their booking history, considering factors like:
        - Similarity to previous bookings
        - Location preferences
        - Amenities
        - Price range
        - Overall suitability

        Provide your ranking as a strict ordering of the listing identifiers.
        Output MUST be in the following JSON format and include the listing_id:
        {{
            "ranking": "[1] > [2] > [3] > ..."
        }}
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
        # Create prompt
        prompt = self.create_pairwise_ranking_prompt(candidate_listings, user_history)
        
        # Get LLM response
        try:
            response = self.generate_response(prompt)
            print(f"op={op}. generate_response finished. response={response}")
            # Parse JSON response
            ranking_data = json.loads(response)
            print(f"op={op}. JSON loaded")
            # Extract ranking order
            ranking_str = ranking_data.get('ranking', '')
            
            # Parse ranking string
            ranked_listings = [
                int(item.strip('[]')) 
                for item in ranking_str.split('>')
            ]
            
            return ranked_listings
        
        except Exception as e:
            print(f"Ranking error: {e}")
            # Fallback to original order if ranking fails
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