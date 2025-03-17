import pytest
import pandas as pd
import os
from src.data_loader import AirbnbDataLoader

@pytest.fixture
def sample_listings_data():
    return pd.DataFrame({
        'listing_id': [1, 2],
        'listing_url': ['url1', 'url2'],
        'name': ['Listing 1', 'Listing 2'],
        'price': ['$100', '$200'],
        'review_scores_location': [4.5, 4.8],
        'review_scores_cleanliness': [4.7, 4.9],
        'review_scores_value': [4.6, 4.7],
        'instant_bookable': [True, False]
    })

@pytest.fixture
def sample_reviews_data():
    return pd.DataFrame({
        'listing_id': [1, 1, 2],
        'id': [101, 102, 103],
        'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'reviewer_id': [501, 502, 503],
        'comments': ['Great stay!', 'Awesome!', 'Lovely place']
    })

class TestAirbnbDataLoader:
    def test_init(self):
        loader = AirbnbDataLoader(data_dir='test_data')
        assert loader.data_dir == 'test_data'

    def test_validate_data_with_valid_listings(self, sample_listings_data):
        loader = AirbnbDataLoader()
        assert loader.validate_data(listings_df=sample_listings_data) == True

    def test_validate_data_with_valid_reviews(self, sample_reviews_data):
        loader = AirbnbDataLoader()
        assert loader.validate_data(reviews_df=sample_reviews_data) == True

    def test_validate_data_with_invalid_listings(self):
        invalid_listings = pd.DataFrame({
            'wrong_column': [1, 2]
        })
        loader = AirbnbDataLoader()
        assert loader.validate_data(listings_df=invalid_listings) == False

    def test_process_listings(self, sample_listings_data):
        loader = AirbnbDataLoader()
        processed_df = loader.process_listings(sample_listings_data)
        
        # Check if price is converted to float
        assert isinstance(processed_df['price'].iloc[0], (float, int))
        assert processed_df['price'].iloc[0] == 100.0

    def test_process_reviews(self, sample_reviews_data):
        loader = AirbnbDataLoader()
        processed_df = loader.process_reviews(sample_reviews_data)
        
        # Check if date is converted to datetime
        assert isinstance(processed_df['date'].iloc[0], pd.Timestamp)

    def test_load_data_nonexistent_directory(self):
        loader = AirbnbDataLoader(data_dir='nonexistent_dir')
        listings_df, reviews_df, calendar_df = loader.load_data()
        assert listings_df is None
        assert reviews_df is None
        assert calendar_df is None 