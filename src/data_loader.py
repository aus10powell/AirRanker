import os
import pandas as pd
from typing import Tuple, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AirbnbDataLoader:
    """
    A class to handle loading and preprocessing of Airbnb data including listings, reviews, and calendar data.
    """
    
    def __init__(self, data_dir: str = '../data'):
        """
        Initialize the AirbnbDataProcessor with the data directory path.
        
        Args:
            data_dir (str): Path to the directory containing the data files
        """
        self.data_dir = data_dir
        
    def validate_data(self, listings_df: Optional[pd.DataFrame] = None, 
                     reviews_df: Optional[pd.DataFrame] = None,
                     calendar_df: Optional[pd.DataFrame] = None) -> bool:
        """
        Validate the loaded data for required columns and data quality.
        """
        if listings_df is not None:
            required_listing_columns = [
                'listing_id', 'listing_url', 'name', 'price', 'review_scores_location',
                'review_scores_cleanliness', 'review_scores_value', 'instant_bookable'
            ]
            if not all(col in listings_df.columns for col in required_listing_columns):
                missing_cols = [col for col in required_listing_columns if col not in listings_df.columns]
                logger.error(f"Missing required columns in listings data: {missing_cols}")
                logger.error(f"Available columns: {listings_df.columns.tolist()}")
                return False
                
        if reviews_df is not None:
            required_review_columns = ['listing_id', 'id', 'date', 'reviewer_id', 'comments']
            if not all(col in reviews_df.columns for col in required_review_columns):
                missing_cols = [col for col in required_review_columns if col not in reviews_df.columns]
                logger.error(f"Missing required columns in reviews data: {missing_cols}")
                logger.error(f"Available columns: {reviews_df.columns.tolist()}")
                return False
                
        if calendar_df is not None:
            required_calendar_columns = ['listing_id', 'date', 'available', 'price', 'minimum_nights']
            if not all(col in calendar_df.columns for col in required_calendar_columns):
                missing_cols = [col for col in required_calendar_columns if col not in calendar_df.columns]
                logger.error(f"Missing required columns in calendar data: {missing_cols}")
                logger.error(f"Available columns: {calendar_df.columns.tolist()}")
                return False
                
        return True
    
    def process_listings(self, listings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process listings data with specific cleaning and transformations.
        """
        try:
            # Rename id to listing_id if it exists
            if 'id' in listings_df.columns and 'listing_id' not in listings_df.columns:
                listings_df = listings_df.rename(columns={'id': 'listing_id'})
            
            # Convert listing_id to int
            listings_df['listing_id'] = listings_df['listing_id'].astype(int)
            
            # Convert price-related columns to float
            price_columns = [col for col in listings_df.columns if 'price' in col.lower()]
            for col in price_columns:
                if col in listings_df.columns:
                    listings_df[col] = listings_df[col].replace('[\$,]', '', regex=True).astype(float)
            
            # Convert review scores to float and handle missing values
            score_columns = [col for col in listings_df.columns if col.startswith('review_scores_')]
            for col in score_columns:
                listings_df[col] = pd.to_numeric(listings_df[col], errors='coerce')
                listings_df[col] = listings_df[col].fillna(listings_df[col].median())
            
            # Convert boolean columns
            bool_columns = ['instant_bookable']
            for col in bool_columns:
                if col in listings_df.columns:
                    listings_df[col] = listings_df[col].map({'t': True, 'f': False})
            
            # Convert count columns to int
            count_columns = [col for col in listings_df.columns if 'count' in col.lower()]
            for col in count_columns:
                if col in listings_df.columns:
                    listings_df[col] = pd.to_numeric(listings_df[col], errors='coerce').fillna(0).astype(int)
            
            return listings_df
            
        except Exception as e:
            logger.error(f"Error processing listings data: {str(e)}")
            return listings_df
    
    def process_reviews(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process reviews data with specific cleaning and transformations.
        """
        try:
            # Convert listing_id and reviewer_id to int
            reviews_df['listing_id'] = reviews_df['listing_id'].astype(int)
            reviews_df['reviewer_id'] = reviews_df['reviewer_id'].astype(int)
            
            # Convert date to datetime
            reviews_df['date'] = pd.to_datetime(reviews_df['date'])
            
            # Clean comments (remove extra whitespace, handle NaN)
            reviews_df['comments'] = reviews_df['comments'].fillna('')
            reviews_df['comments'] = reviews_df['comments'].str.strip()
            
            return reviews_df
            
        except Exception as e:
            logger.error(f"Error processing reviews data: {str(e)}")
            return reviews_df
    
    def process_calendar(self, calendar_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process calendar data with specific cleaning and transformations.
        """
        try:
            # Convert listing_id to int
            calendar_df['listing_id'] = calendar_df['listing_id'].astype(int)
            
            # Convert date to datetime
            calendar_df['date'] = pd.to_datetime(calendar_df['date'])
            
            # Convert price columns
            price_columns = ['price', 'adjusted_price']
            for col in price_columns:
                if col in calendar_df.columns:
                    calendar_df[col] = calendar_df[col].replace('[\$,]', '', regex=True)
                    calendar_df[col] = pd.to_numeric(calendar_df[col], errors='coerce')
            
            # Convert availability to boolean
            calendar_df['available'] = calendar_df['available'].map({'t': True, 'f': False})
            
            # Convert night restrictions to int
            night_columns = ['minimum_nights', 'maximum_nights']
            for col in night_columns:
                if col in calendar_df.columns:
                    calendar_df[col] = pd.to_numeric(calendar_df[col], errors='coerce').fillna(0).astype(int)
            
            return calendar_df
            
        except Exception as e:
            logger.error(f"Error processing calendar data: {str(e)}")
            return calendar_df
    
    def load_data(self, city: str = 'seattle') -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Load and process all Airbnb data for a specified city.
        
        Args:
            city (str): City name for the data files (default: 'seattle')
            
        Returns:
            Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]: 
            Processed listings, reviews, and calendar DataFrames
        """
        try:
            # Construct file paths
            listings_path = os.path.join(self.data_dir, f'listings.parquet')
            reviews_path = os.path.join(self.data_dir, f'reviews.parquet')
            calendar_path = os.path.join(self.data_dir, f'calendar.parquet')
            
            # Load data
            listings_df = pd.read_parquet(listings_path) if os.path.exists(listings_path) else None
            reviews_df = pd.read_parquet(reviews_path) if os.path.exists(reviews_path) else None
            calendar_df = pd.read_parquet(calendar_path) if os.path.exists(calendar_path) else None
            
            # Process each dataset if it exists
            if listings_df is not None:
                listings_df = self.process_listings(listings_df)
            if reviews_df is not None:
                reviews_df = self.process_reviews(reviews_df)
            if calendar_df is not None:
                calendar_df = self.process_calendar(calendar_df)
            
            # Validate data
            if not self.validate_data(listings_df, reviews_df, calendar_df):
                return None, None, None
            
            logger.info(f"Successfully loaded and processed data for {city}")
            return listings_df, reviews_df, calendar_df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None, None, None
