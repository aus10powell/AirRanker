import os
import logging
# To run this script, use: python -m src.evaluations.run_main_evaluation from the project root
from src.data_loader import AirbnbDataLoader
from src.listing_ranker import ListingRanker
from src.evaluations.recommender_evaluator import RecommenderEvaluator, run_evaluation
from pprint import pprint
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('airranker.log')
    ]
)
logger = logging.getLogger(__name__)

def run_recommender_evaluation(listings_df, reviews_df, llm_model='phi3', embedding_model='all-MiniLM-L6-v2', sample_size=20, k=20, use_pairwise=False, min_reviews=None, max_reviews=None):
    """
    Run the recommender evaluation pipeline
    
    Args:
        listings_df: DataFrame containing listing information
        reviews_df: DataFrame containing review information
        llm_model: Language model to use for LLM ranker
        embedding_model: Sentence transformer model to use for embeddings
        sample_size: Number of users to evaluate
        k: Number of recommendations to retrieve
        use_pairwise: Whether to use pair-wise ranking with LLM
        min_reviews: Minimum number of reviews required for a user to be included
        max_reviews: Maximum number of reviews allowed for a user to be included
    """
    op = "run_recommender_evaluation"
    logger.info(f"op={op} Running recommender evaluation...")
    logger.info(f"op={op} Sample size: {sample_size} users")
    logger.info(f"op={op} Using LLM model: {llm_model}")
    logger.info(f"op={op} Using embedding model: {embedding_model}")
    logger.info(f"op={op} Using pair-wise ranking: {use_pairwise}")
    if min_reviews is not None:
        logger.info(f"op={op} Minimum reviews required: {min_reviews}")
    if max_reviews is not None:
        logger.info(f"op={op} Maximum reviews allowed: {max_reviews}")
    
    # Run the evaluation
    results = run_evaluation(
        df_listings=listings_df,
        df_reviews=reviews_df,
        llm_model=llm_model,
        embedding_model=embedding_model,
        sample_size=sample_size,
        k=k,
        use_pairwise=use_pairwise,
        min_reviews=min_reviews,
        max_reviews=max_reviews
    )
    
    if results is not None:
        logger.info(f"op={op} Evaluation Results:")
        logger.info(f"op={op} {results}")
        
        # Log the min/max reviews parameters in a more visible format
        logger.info(f"op={op} ===== REVIEW FILTERS =====")
        logger.info(f"op={op} Min reviews: {min_reviews}")
        logger.info(f"op={op} Max reviews: {max_reviews}")
        logger.info(f"op={op} =========================")
    else:
        logger.error(f"op={op} Evaluation failed to complete.")

# Example Usage
def main():
    # Initialize data loader
    data_loader = AirbnbDataLoader(data_dir='data/seattle')
    
    # Load data
    logger.info(f"op=main Loading data...")
    listings_df, reviews_df, _ = data_loader.load_data(city='seattle')
    if listings_df is None or reviews_df is None:
        logger.error(f"op=main Failed to load data!")
        return
        
    logger.info(f"op=main Data loaded successfully!")
    logger.info(f"op=main Listings columns: {listings_df.columns.tolist()}")
    logger.info(f"op=main Reviews columns: {reviews_df.columns.tolist()}")
    logger.info(f"op=main Reviews data:")
    logger.info(f"op=main reviews_df.head(): {reviews_df.head()}")

    # Initialize ranker with data
    ranker = ListingRanker(listings_df=listings_df, reviews_df=reviews_df, llm_model='llama3.2')

    # Example: Get recommendations for a user
    user_id = reviews_df['reviewer_id'].iloc[0]  # Get first user as example
    logger.info(f"op=main User ID: {user_id}")
    logger.info(f"op=main User history: {reviews_df[reviews_df['reviewer_id'] == user_id]['listing_id'].tolist()}")
    user_history = listings_df[listings_df['listing_id'].isin(
        reviews_df[reviews_df['reviewer_id'] == user_id]['listing_id']
    )]
    
    # Get candidate items (excluding user history)
    candidates = listings_df[~listings_df['listing_id'].isin(user_history['listing_id'])]
    
    # Set retrieval parameters for evaluation
    top_k = 10
    use_pairwise = True
    sample_size = 400
    min_reviews = 2  # Set minimum reviews to 2 to ensure users have at least 2 reviews
    max_reviews = None  # No maximum limit
    logger.info(f"op=main Retrieval parameters: top_k={top_k}, use_pairwise={use_pairwise}, sample_size={sample_size}, min_reviews={min_reviews}, max_reviews={max_reviews}")
    # Get recommendations
    # logger.info("Getting recommendations without pair-wise ranking:")
    # recommendations = ranker.retrieve_candidates(
    #     user_history=user_history,
    #     candidates=candidates,
    #     interaction_data=reviews_df,
    #     top_k=top_k,
    #     use_pairwise=False
    # )
    
    # logger.info("Top 5 recommendations (without pair-wise):")
    # logger.info(recommendations[['name', 'score']].head())
    
    # logger.info("Getting recommendations with pair-wise ranking:")
    # recommendations_with_llm = ranker.retrieve_candidates(
    #     user_history=user_history,
    #     candidates=candidates,
    #     interaction_data=reviews_df,
    #     top_k=top_k,
    #     use_pairwise=use_pairwise
    # )
    
    # logger.info("Top 5 recommendations (with pair-wise):")
    # logger.info(recommendations_with_llm[['name', 'score']].head())

    # Run the recommender evaluation
    run_recommender_evaluation(
        listings_df, 
        reviews_df, 
        llm_model='llama3.2', 
        embedding_model='all-MiniLM-L6-v2', 
        sample_size=sample_size, 
        k=top_k, 
        use_pairwise=use_pairwise,
        min_reviews=min_reviews,
        max_reviews=max_reviews
    )

if __name__ == '__main__':
    main()
