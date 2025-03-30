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
def sample_reviews():
    return pd.DataFrame({
        'listing_id': [1, 2],
        'reviewer_id': [166478, 166478],
        'comments': ['Great location!', 'Very clean'],
        'date': ['2024-01-01', '2024-01-02']
    })

class TestListingRanker:
    def test_init(self, sample_listings, sample_reviews):
        ranker = ListingRanker(listings_df=sample_listings, reviews_df=sample_reviews)
        assert isinstance(ranker.listings_df, pd.DataFrame)
        assert isinstance(ranker.reviews_df, pd.DataFrame)

    def test_create_pairwise_ranking_prompt(self, sample_listings, sample_reviews):
        ranker = ListingRanker(listings_df=sample_listings, reviews_df=sample_reviews)
        prompt = ranker._construct_item_description(sample_listings.iloc[0])
        
        # Check if prompt contains key elements
        assert "Cozy Studio" in prompt
        assert "100" in prompt
        assert "4.5" in prompt
        assert "4.7" in prompt

    def test_rank_listings_returns_dataframe(self, sample_listings, sample_reviews):
        ranker = ListingRanker(listings_df=sample_listings, reviews_df=sample_reviews)
        user_history = sample_listings.head(1)
        candidates = sample_listings.tail(2)
        
        ranked_listings = ranker.retrieve_candidates(
            user_history=user_history,
            candidates=candidates,
            interaction_data=sample_reviews
        )
        
        # Check if output is a DataFrame with expected properties
        assert isinstance(ranked_listings, pd.DataFrame)
        assert len(ranked_listings) <= len(candidates)
        assert 'score' in ranked_listings.columns

    @pytest.mark.integration
    def test_end_to_end_ranking(self, sample_listings, sample_reviews):
        ranker = ListingRanker(listings_df=sample_listings, reviews_df=sample_reviews)
        user_history = sample_listings.head(1)
        candidates = sample_listings.tail(2)
        
        ranked_listings = ranker.retrieve_candidates(
            user_history=user_history,
            candidates=candidates,
            interaction_data=sample_reviews
        )
        
        # Basic validation of the ranking output
        assert len(ranked_listings) > 0
        assert isinstance(ranked_listings, pd.DataFrame)
        assert 'score' in ranked_listings.columns
        
        # Check if all original listings are present in candidates
        candidate_ids = set(candidates['listing_id'])
        ranked_ids = set(ranked_listings['listing_id'])
        assert ranked_ids.issubset(candidate_ids) 