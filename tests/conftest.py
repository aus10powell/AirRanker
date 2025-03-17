import pytest
import os
import tempfile
import pandas as pd

@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture(scope="session")
def setup_test_data(test_data_dir):
    """Create sample data files in the test directory"""
    # Create listings.csv
    listings_df = pd.DataFrame({
        'listing_id': [1, 2],
        'listing_url': ['url1', 'url2'],
        'name': ['Test Listing 1', 'Test Listing 2'],
        'price': ['$100', '$200'],
        'review_scores_location': [4.5, 4.8],
        'review_scores_cleanliness': [4.7, 4.9],
        'review_scores_value': [4.6, 4.7],
        'instant_bookable': [True, False]
    })
    
    # Create reviews.csv
    reviews_df = pd.DataFrame({
        'listing_id': [1, 1, 2],
        'id': [101, 102, 103],
        'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'reviewer_id': [501, 502, 503],
        'comments': ['Great stay!', 'Awesome!', 'Lovely place']
    })
    
    # Save the files
    listings_df.to_csv(os.path.join(test_data_dir, 'listings.csv'), index=False)
    reviews_df.to_csv(os.path.join(test_data_dir, 'reviews.csv'), index=False)
    
    return test_data_dir

@pytest.fixture
def mock_ollama_response():
    """Mock response for ollama API calls"""
    return {
        'response': 'Listing 1 is better than Listing 2 because of its location and value.'
    } 