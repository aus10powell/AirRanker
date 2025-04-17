"""
Semantic Evaluation for Airbnb Ranker

This script evaluates the performance of the Airbnb Ranker model in a semantic search scenario.
It tests how well the model can identify the listing a user actually booked based solely on their review text.

The evaluation process:
1. Creates a dataset of one-review users with high-rated listings
2. For each user, uses their review text as a direct query
3. Uses the StarRanker model to rank all listings based on the review text
4. Checks where the actual booked listing appears in the ranking
5. Calculates evaluation metrics (hit rate, mean reciprocal rank, etc.)

This approach tests if the model can effectively understand the semantic meaning of reviews
and match them to the appropriate listings without relying on user history or other contextual information.
"""

import pandas as pd
from pathlib import Path
import json
from typing import Tuple, List, Optional, Dict, Any
import logging
import os
import sys
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime

# Add the parent directory to the path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import AirbnbDataLoader
from listing_ranker import ListingRanker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(data_dir: str, city: str = 'seattle') -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load reviews and listings data using AirbnbDataLoader.
    
    Args:
        data_dir: Path to the data directory
        city: City name for the data files (default: 'seattle')
        
    Returns:
        Tuple of (reviews_df, listings_df)
    """
    logger.info(f"Loading data for {city} using AirbnbDataLoader...")
    data_loader = AirbnbDataLoader(data_dir=data_dir)
    listings_df, reviews_df, _ = data_loader.load_data(city=city)
    
    if listings_df is None or reviews_df is None:
        logger.error("Failed to load data!")
        return None, None
        
    logger.info(f"Successfully loaded data: {len(listings_df)} listings, {len(reviews_df)} reviews")
    return reviews_df, listings_df

def filter_one_review_users(df_reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to keep only users with exactly one review.
    
    Args:
        df_reviews: Reviews dataframe
        
    Returns:
        Filtered dataframe with only one-review users
    """
    logger.info("Filtering for one-review users...")
    reviewer_counts = df_reviews['reviewer_id'].value_counts()
    one_review_users = reviewer_counts[reviewer_counts == 1].index
    return df_reviews[df_reviews['reviewer_id'].isin(one_review_users)]

def create_evaluation_dataset(
    df_reviews: pd.DataFrame,
    df_listings: pd.DataFrame,
    min_rating: float = 4.81,
    min_chars: int = 50,
    num_users: Optional[int] = None,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Create evaluation dataset from reviews and listings data.
    
    Args:
        df_reviews: Reviews dataframe
        df_listings: Listings dataframe
        min_rating: Minimum review score to include
        min_chars: Minimum number of characters in review
        num_users: Number of users to include in the evaluation dataset (if None, include all)
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame containing evaluation dataset
    """
    logger.info("Creating evaluation dataset...")
    
    # Filter one-review users
    df_reviews = filter_one_review_users(df_reviews)
    
    # Join reviews with listings
    df_merged = pd.merge(
        df_reviews,
        df_listings,
        left_on='listing_id',
        right_on='listing_id',
        how='inner'
    )
    
    # Filter by rating
    df_merged = df_merged[df_merged['review_scores_rating'] >= min_rating]
    
    # Filter by comment length (using character count instead of tokens)
    df_merged = df_merged[df_merged['comments'].str.len() >= min_chars]
    
    # Create final dataset
    evaluation_data = []
    for _, row in df_merged.iterrows():
        entry = {
            'query': row['comments'],  # Using original comments without cleaning
            'target_listing_id': row['listing_id'],
            'listing_name': row.get('name', ''),
            'listing_description': row.get('description', ''),
            'review_score': row['review_scores_rating']
        }
        evaluation_data.append(entry)
    
    evaluation_df = pd.DataFrame(evaluation_data)
    
    # Select a subset of users if requested
    if num_users is not None and num_users > 0 and num_users < len(evaluation_df):
        logger.info(f"Selecting {num_users} users for evaluation...")
        np.random.seed(random_seed)
        evaluation_df = evaluation_df.sample(n=num_users, random_state=random_seed)
        logger.info(f"Selected {len(evaluation_df)} users for evaluation")
    
    return evaluation_df

def calculate_ndcg(relevance_scores, k):
    """
    Calculate Normalized Discounted Cumulative Gain at k.
    
    Args:
        relevance_scores: List of binary relevance scores (1 for relevant, 0 for not relevant)
        k: Number of top items to consider
        
    Returns:
        float: NDCG@k score
    """
    if not relevance_scores or k <= 0:
        return 0.0
    
    # Limit to top k items
    relevance_scores = relevance_scores[:k]
    
    # Calculate DCG
    dcg = 0.0
    for i, rel in enumerate(relevance_scores):
        dcg += rel / np.log2(i + 2)  # log2(i+2) because i is 0-indexed
    
    # Calculate IDCG (ideal DCG)
    # Sort relevance scores in descending order
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal_scores):
        idcg += rel / np.log2(i + 2)
    
    # Calculate NDCG
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    return ndcg

def simulate_recommendation_scenario(
    evaluation_df: pd.DataFrame,
    listings_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    top_k: int = 20,
    alpha: float = 0.5,
    use_pairwise: bool = False,
    llm_model: str = 'llama3.2'
) -> Dict[str, Any]:
    """
    Simulate a recommendation scenario for each pseudo-user in the evaluation dataset.
    
    Args:
        evaluation_df: Evaluation dataset with queries and target listings
        listings_df: Full listings dataframe
        reviews_df: Full reviews dataframe
        top_k: Number of top recommendations to consider
        alpha: Weight between semantic and collaborative relationships
        use_pairwise: Whether to use pair-wise ranking with LLM
        llm_model: Name of the LLM model to use for pairwise ranking
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Simulating recommendation scenario...")
    
    # Initialize the ranker with the specified LLM model
    ranker = ListingRanker(listings_df, reviews_df, llm_model=llm_model)
    
    # Metrics
    total_users = len(evaluation_df)
    found_in_top_k = 0
    found_in_top_5 = 0
    found_in_top_10 = 0
    found_in_top_20 = 0
    mean_reciprocal_rank = 0.0
    not_found = 0
    
    # For NDCG calculation
    all_ndcg_scores = []
    
    # For each pseudo-user
    for idx, row in tqdm(evaluation_df.iterrows(), total=total_users, desc="Evaluating users"):
        # Get the target listing (ground truth)
        target_listing_id = row['target_listing_id']
        review_text = row['query']
        
        # Include all listings including the target
        candidates = listings_df.copy()
        
        # Try to rank the candidates using the review text as a direct query
        try:
            ranked_candidates = ranker.retrieve_by_query(
                query_text=review_text,
                candidates=candidates,
                interaction_data=reviews_df,
                top_k=top_k,
                alpha=alpha,
                use_pairwise=use_pairwise
            )
        except ConnectionError as e:
            logger.error(f"Connection error with Ollama: {e}")
            raise ConnectionError(f"Failed to connect to Ollama service: {e}")
        
        # Check if target is in the ranked candidates
        target_rank = None
        relevance_scores = []
        
        for rank, (_, candidate) in enumerate(ranked_candidates.iterrows(), 1):
            if candidate['listing_id'] == target_listing_id:
                target_rank = rank
                relevance_scores.append(1)
            else:
                relevance_scores.append(0)
            
            if rank >= top_k:
                break
        
        # Calculate NDCG for this user
        ndcg_score = calculate_ndcg(relevance_scores, top_k)
        all_ndcg_scores.append(ndcg_score)
        
        # Update metrics
        if target_rank is not None:
            found_in_top_k += 1
            mean_reciprocal_rank += 1.0 / target_rank
            
            if target_rank <= 5:
                found_in_top_5 += 1
            if target_rank <= 10:
                found_in_top_10 += 1
            if target_rank <= 20:
                found_in_top_20 += 1
        else:
            not_found += 1
    
    # Calculate final metrics with high precision
    metrics = {
        'total_users': total_users,
        'found_in_top_k': found_in_top_k,
        'found_in_top_5': found_in_top_5,
        'found_in_top_10': found_in_top_10,
        'found_in_top_20': found_in_top_20,
        'not_found': not_found,
        'hit_rate_at_k': round(found_in_top_k / total_users if total_users > 0 else 0, 5),
        'hit_rate_at_5': round(found_in_top_5 / total_users if total_users > 0 else 0, 5),
        'hit_rate_at_10': round(found_in_top_10 / total_users if total_users > 0 else 0, 5),
        'hit_rate_at_20': round(found_in_top_20 / total_users if total_users > 0 else 0, 5),
        f'mrr@{top_k}': round(mean_reciprocal_rank / total_users if total_users > 0 else 0, 5),
        f'ndcg@{top_k}': round(np.mean(all_ndcg_scores) if all_ndcg_scores else 0, 5)
    }
    
    logger.info(f"Evaluation metrics: {metrics}")
    return metrics

def save_evaluation_results(
    metrics: Dict[str, Any],
    output_path: str,
    llm_model: str,
    use_pairwise: bool,
    top_k: int,
    alpha: float,
    min_rating: float,
    min_chars: int,
    num_users: int
):
    """
    Save evaluation results to file.
    
    Args:
        metrics: Dictionary with evaluation metrics
        output_path: Path to save the file
        llm_model: Name of the LLM model used
        use_pairwise: Whether pair-wise ranking was used
        top_k: Number of top recommendations considered
        alpha: Weight between semantic and collaborative relationships
        min_rating: Minimum rating threshold used
        min_chars: Minimum characters in review text
        num_users: Number of users evaluated
    """
    logger.info(f"Saving evaluation results to {output_path}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    metrics_with_metadata = {
        **metrics,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_info': {
            'llm_model': llm_model,
            'embedding_model': 'all-MiniLM-L6-v2',
            'use_pairwise': use_pairwise
        },
        'evaluation_params': {
            'top_k': top_k,
            'alpha': alpha,
            'min_rating': min_rating,
            'min_chars': min_chars,
            'num_users': num_users
        }
    }
    
    # Use a custom JSON encoder to ensure high precision for floating point values
    class HighPrecisionEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, float):
                return round(obj, 5)
            return super().default(obj)
    
    with open(output_path, 'w') as f:
        json.dump(metrics_with_metadata, f, indent=2, cls=HighPrecisionEncoder)
    
    logger.info(f"Saved evaluation results to {output_path}")
    
    # Also save a copy with timestamp in the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_path = output_path.parent / f"semantic_evaluation_results_{timestamp}.json"
    with open(timestamped_path, 'w') as f:
        json.dump(metrics_with_metadata, f, indent=2, cls=HighPrecisionEncoder)
    logger.info(f"Saved timestamped copy to {timestamped_path}")

def main():
    # Start timing
    start_time = time.time()
    
    # Configuration parameters
    data_dir = "data/seattle"
    city = "seattle"
    output_dir = Path("data/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "semantic_evaluation_results.json"
    
    # Model parameters
    num_users = 2  # Number of users to evaluate (set to 0 for all users)
    min_rating = 4.81
    min_chars = 50
    top_k = 20
    alpha = 0.5
    use_pairwise = True
    llm_model = "llama3.2"  # Using llama3.2 model instead of phi3.5
    random_seed = 42
    
    # Log configuration
    logger.info(f"Starting semantic evaluation with {num_users if num_users > 0 else 'all'} users")
    logger.info(f"Configuration: min_rating={min_rating}, min_chars={min_chars}, top_k={top_k}, alpha={alpha}, use_pairwise={use_pairwise}, llm_model={llm_model}")
    logger.info(f"Results will be saved to {output_path}")
    
    # Load data
    df_reviews, df_listings = load_data(data_dir, city)
    
    if df_reviews is None or df_listings is None:
        logger.error("Failed to load data. Exiting.")
        return
    
    # Create evaluation dataset
    evaluation_df = create_evaluation_dataset(
        df_reviews=df_reviews,
        df_listings=df_listings,
        min_rating=min_rating,
        min_chars=min_chars,
        num_users=num_users if num_users > 0 else None,
        random_seed=random_seed
    )
    
    # Simulate recommendation scenario
    metrics = simulate_recommendation_scenario(
        evaluation_df=evaluation_df,
        listings_df=df_listings,
        reviews_df=df_reviews,
        top_k=top_k,
        alpha=alpha,
        use_pairwise=use_pairwise,
        llm_model=llm_model
    )
    
    # Save evaluation results
    save_evaluation_results(
        metrics=metrics,
        output_path=output_path,
        llm_model=llm_model,
        use_pairwise=use_pairwise,
        top_k=top_k,
        alpha=alpha,
        min_rating=min_rating,
        min_chars=min_chars,
        num_users=num_users
    )
    
    # Calculate and log total execution time
    end_time = time.time()
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.5f}s")
    
    # Print summary of results
    logger.info("\nEvaluation Results Summary:")
    logger.info(f"Total users evaluated: {metrics['total_users']}")
    logger.info(f"Hit rate at k={top_k}: {metrics['hit_rate_at_k']:.5f}")
    logger.info(f"Hit rate at k=5: {metrics['hit_rate_at_5']:.5f}")
    logger.info(f"Hit rate at k=10: {metrics['hit_rate_at_10']:.5f}")
    logger.info(f"Hit rate at k=20: {metrics['hit_rate_at_20']:.5f}")
    logger.info(f"MRR@{top_k}: {metrics[f'mrr@{top_k}']:.5f}")
    logger.info(f"NDCG@{top_k}: {metrics[f'ndcg@{top_k}']:.5f}")
    logger.info(f"Not found: {metrics['not_found']} ({metrics['not_found']/metrics['total_users']*100:.5f}%)")

if __name__ == "__main__":
    main() 