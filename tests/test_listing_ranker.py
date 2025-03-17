import pytest
import pandas as pd
from src.listing_ranker import ListingRanker

@pytest.fixture
def sample_listings():
    return pd.DataFrame({
        'listing_id': [1, 2, 3],
        'listing_url': ['url1', 'url2', 'url3'],
        'name': ['Cozy Studio', 'Luxury Apartment', 'Beach House'],
        'price': [100, 200, 300],
        'review_scores_location': [4.5, 4.8, 4.7],
        'review_scores_cleanliness': [4.7, 4.9, 4.8],
        'review_scores_value': [4.6, 4.7, 4.8]
    })

@pytest.fixture
def sample_user_history():
    return pd.DataFrame({
        'listing_id': [4, 5],
        'reviewer_id': [166478, 166478],
        'comments': ['Great location!', 'Very clean'],
        'date': ['2024-01-01', '2024-01-02']
    })

class TestListingRanker:
    def test_init(self):
        ranker = ListingRanker(model='phi3')
        assert ranker.llm == 'phi3'

    def test_create_pairwise_ranking_prompt(self, sample_listings, sample_user_history):
        ranker = ListingRanker()
        prompt = ranker.create_pairwise_ranking_prompt(sample_listings, sample_user_history)
        
        # Check if prompt contains key elements
        assert "User's Previous Bookings" in prompt
        assert "Candidate Listings" in prompt
        assert "Cozy Studio" in prompt
        assert "Luxury Apartment" in prompt
        assert "Beach House" in prompt

    def test_rank_listings_returns_dataframe(self, sample_listings, sample_user_history):
        ranker = ListingRanker()
        ranked_listings = ranker.rank_listings(sample_listings, sample_user_history)
        
        # Check if output is a DataFrame with the same number of rows
        assert isinstance(ranked_listings, pd.DataFrame)
        assert len(ranked_listings) == len(sample_listings)
        
        # Check if all original columns are preserved
        for col in sample_listings.columns:
            assert col in ranked_listings.columns

    @pytest.mark.integration
    def test_end_to_end_ranking(self, sample_listings, sample_user_history):
        ranker = ListingRanker()
        ranked_listings = ranker.rank_listings(sample_listings, sample_user_history)
        
        # Basic validation of the ranking output
        assert len(ranked_listings) > 0
        assert isinstance(ranked_listings, pd.DataFrame)
        
        # Check if all original listings are present
        original_ids = set(sample_listings['listing_id'])
        ranked_ids = set(ranked_listings['listing_id'])
        assert original_ids == ranked_ids 