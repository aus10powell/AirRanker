import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from datetime import datetime
import json
import time
from listing_ranker import ListingRanker

class RecommenderEvaluator:
    def __init__(self, ranker, df_listings, df_reviews, metrics=None):
        """
        Initialize the evaluator
        
        Args:
            ranker: The recommendation model to evaluate
            df_listings: DataFrame containing listing information
            df_reviews: DataFrame containing user reviews
            metrics: List of metrics to compute (default: ['ndcg', 'precision', 'recall', 'diversity'])
        """
        self.ranker = ranker
        self.df_listings = df_listings
        self.df_reviews = df_reviews
        self.metrics = metrics or ['ndcg', 'precision', 'recall', 'diversity', 'coverage', 'mrr']
        
        # Extract unique user IDs
        self.user_ids = df_reviews['reviewer_id'].unique()
        
    def prepare_holdout_set(self, test_size=0.2, min_reviews=3, random_state=42):
        """
        Split users into train and test sets
        
        Args:
            test_size: Proportion of users to include in the test set
            min_reviews: Minimum number of reviews required for a user to be included
            random_state: Random seed for reproducibility
            
        Returns:
            dict: Dictionary containing train and test sets
        """
        # Set random seed for reproducibility
        random.seed(random_state)
        np.random.seed(random_state)
        
        # Filter users with enough reviews
        user_review_counts = self.df_reviews['reviewer_id'].value_counts()
        eligible_users = user_review_counts[user_review_counts >= min_reviews].index.tolist()
        
        print(f"Found {len(eligible_users)} users with at least {min_reviews} reviews")
        
        # Skip if not enough eligible users
        if len(eligible_users) < 5:
            print(f"Warning: Only {len(eligible_users)} eligible users found. Need at least 5.")
            # Return an empty holdout set
            return {
                'train_users': [],
                'test_users': [],
                'user_histories': {}
            }
        
        # Split users into train and test
        train_users, test_users = train_test_split(
            eligible_users, 
            test_size=test_size, 
            random_state=random_state
        )
        
        holdout_data = {
            'train_users': train_users,
            'test_users': test_users,
            'user_histories': {}
        }
        
        # For each test user, split their reviews into history and holdout
        for user_id in tqdm(test_users, desc="Preparing holdout data"):
            user_reviews = self.df_reviews[self.df_reviews['reviewer_id'] == user_id].copy()
            
            # Skip if user has no reviews
            if len(user_reviews) == 0:
                continue
            
            # Sort reviews by date
            try:
                user_reviews = user_reviews.sort_values('date')
            except:
                # If date sorting fails, use the original order
                pass
            
            # Use 70% of reviews as history, 30% as holdout
            history_size = max(1, int(len(user_reviews) * 0.7))
            
            history = user_reviews.iloc[:history_size]
            holdout = user_reviews.iloc[history_size:]
            
            # Store the history and holdout
            holdout_data['user_histories'][user_id] = {
                'history': history,
                'holdout': holdout
            }
        
        return holdout_data
    
    def calculate_ndcg(self, recommended_ids, relevant_ids, k=10):
        """Calculate Normalized Discounted Cumulative Gain"""
        if not relevant_ids or len(recommended_ids) == 0:
            return 0.0
        
        # Create binary relevance array
        relevance = np.zeros(len(recommended_ids[:k]))
        for i, item_id in enumerate(recommended_ids[:k]):
            if item_id in relevant_ids:
                relevance[i] = 1
                
        # If no relevant items are in the recommendations, return 0
        if np.sum(relevance) == 0:
            return 0.0
            
        # Calculate ideal DCG (all relevant items at the top)
        ideal_relevance = np.zeros_like(relevance)
        ideal_relevance[:min(len(relevant_ids), k)] = 1
        
        # Reshape for ndcg_score function
        relevance = relevance.reshape(1, -1)
        ideal_relevance = ideal_relevance.reshape(1, -1)
        
        return ndcg_score(ideal_relevance, relevance)
    
    def calculate_precision_recall(self, recommended_ids, relevant_ids, k=10):
        """Calculate precision and recall at k"""
        if not relevant_ids or len(recommended_ids) == 0:
            return 0.0, 0.0
            
        recommended_k = recommended_ids[:k]
        relevant_recommended = [item for item in recommended_k if item in relevant_ids]
        
        precision = len(relevant_recommended) / len(recommended_k) if recommended_k else 0
        recall = len(relevant_recommended) / len(relevant_ids) if relevant_ids else 0
        
        return precision, recall
    
    def calculate_diversity(self, recommended_listings, features=None):
        """
        Calculate recommendation diversity based on selected features
        
        Args:
            recommended_listings: DataFrame of recommended listings
            features: List of features to consider for diversity
            
        Returns:
            float: Diversity score (0-1)
        """
        if len(recommended_listings) <= 1:
            return 0.0
            
        # Default features if none provided
        if features is None:
            features = ['price', 'review_scores_location', 'review_scores_cleanliness']
            
        # Only keep features that exist in the dataframe
        features = [f for f in features if f in recommended_listings.columns]
        
        if not features:
            return 0.0
            
        # Normalize features
        normalized = pd.DataFrame()
        for feature in features:
            try:
                # Convert to numeric and handle NaN values
                feature_values = pd.to_numeric(recommended_listings[feature], errors='coerce')
                if feature_values.isna().all():
                    continue
                    
                # Skip if all values are the same
                if feature_values.nunique() <= 1:
                    continue
                    
                # Min-max normalization
                normalized[feature] = (feature_values - feature_values.min()) / (feature_values.max() - feature_values.min())
            except:
                continue
                
        if normalized.empty:
            return 0.0
            
        # Calculate pairwise distances as diversity measure
        try:
            from scipy.spatial.distance import pdist, squareform
            distances = pdist(normalized.fillna(0))
            mean_distance = np.mean(distances)
            return mean_distance  # Higher value means more diverse recommendations
        except Exception as e:
            print(f"Error calculating diversity: {e}")
            return 0.0
    
    def calculate_coverage(self, all_recommended_ids):
        """
        Calculate catalog coverage - what percentage of items gets recommended
        
        Args:
            all_recommended_ids: List of lists containing recommended IDs for all users
            
        Returns:
            float: Coverage score (0-1)
        """
        # Flatten the list of lists
        unique_recommended = set()
        for rec_list in all_recommended_ids:
            unique_recommended.update(rec_list)
            
        # Calculate coverage
        total_items = len(self.df_listings)
        coverage = len(unique_recommended) / total_items if total_items > 0 else 0
        
        return coverage
    
    def calculate_mrr(self, recommended_ids, relevant_ids, k=10):
        """
        Calculate Mean Reciprocal Rank
        
        Args:
            recommended_ids: List of recommended item IDs
            relevant_ids: List of relevant item IDs
            k: Number of recommendations to consider
            
        Returns:
            float: MRR score
        """
        if not relevant_ids or len(recommended_ids) == 0:
            return 0.0
            
        # Find the first position where a relevant item appears
        for i, item_id in enumerate(recommended_ids[:k]):
            if item_id in relevant_ids:
                return 1.0 / (i + 1)
                
        return 0.0
    
    def evaluate_user(self, user_id, history, holdout, k=10):
        """
        Evaluate recommendations for a single user
        
        Args:
            user_id: User ID
            history: DataFrame containing user's history
            holdout: DataFrame containing user's holdout items
            k: Number of recommendations to evaluate
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Get candidate listings (all listings minus those in history)
        history_listing_ids = set(history['listing_id'].unique())
        
        # Get relevant features from user's history
        avg_price = history['listing_id'].map(self.df_listings.set_index('listing_id')['price']).mean()
        price_std = history['listing_id'].map(self.df_listings.set_index('listing_id')['price']).std()
        
        # Filter candidates more intelligently
        candidate_listings = self.df_listings[
            (~self.df_listings['listing_id'].isin(history_listing_ids)) &
            # Price within 2 standard deviations of user's average
            (self.df_listings['price'].between(
                avg_price - 2 * price_std if not pd.isna(price_std) else 0,
                avg_price + 2 * price_std if not pd.isna(price_std) else float('inf')
            ))
        ].copy()
        
        if len(candidate_listings) == 0:
            return {metric: 0.0 for metric in self.metrics}, []
            
        # If too many candidates, prioritize by similarity to history
        if len(candidate_listings) > 100:
            # Calculate average ratings from user history
            hist_locations = history['listing_id'].map(
                self.df_listings.set_index('listing_id')['review_scores_location']
            ).mean()
            hist_cleanliness = history['listing_id'].map(
                self.df_listings.set_index('listing_id')['review_scores_cleanliness']
            ).mean()
            
            # Score candidates by similarity to history
            candidate_listings['similarity_score'] = (
                (candidate_listings['review_scores_location'] - hist_locations).abs() +
                (candidate_listings['review_scores_cleanliness'] - hist_cleanliness).abs()
            )
            
            # Keep top 100 most similar candidates
            candidate_listings = candidate_listings.nsmallest(100, 'similarity_score')
            
        # Get recommendations
        try:
            start_time = time.time()
            ranked_recommendations = self.ranker.retrieve_candidates(
                user_history=history,
                candidates=candidate_listings,
                interaction_data=self.df_reviews,
                top_k=k
            )
            end_time = time.time()
            
            # Extract recommended IDs
            recommended_ids = ranked_recommendations['listing_id'].tolist()
            
            # Extract relevant IDs from holdout
            relevant_ids = holdout['listing_id'].unique().tolist()
            
            # Calculate metrics
            metrics_results = {}
            
            if 'ndcg' in self.metrics:
                metrics_results['ndcg'] = self.calculate_ndcg(recommended_ids, relevant_ids, k)
                
            if 'precision' in self.metrics or 'recall' in self.metrics:
                precision, recall = self.calculate_precision_recall(recommended_ids, relevant_ids, k)
                metrics_results['precision'] = precision
                metrics_results['recall'] = recall
                
            if 'diversity' in self.metrics:
                metrics_results['diversity'] = self.calculate_diversity(ranked_recommendations[:k])
                
            if 'mrr' in self.metrics:
                metrics_results['mrr'] = self.calculate_mrr(recommended_ids, relevant_ids, k)
                
            # Add latency metric
            metrics_results['latency'] = end_time - start_time
            
            return metrics_results, recommended_ids
            
        except Exception as e:
            print(f"Error evaluating user {user_id}: {e}")
            return {metric: 0.0 for metric in self.metrics}, []
    
    def evaluate(self, holdout_data, k=10, sample_size=None):
        """
        Evaluate the recommender on the test set
        
        Args:
            holdout_data: Dictionary containing holdout data
            k: Number of recommendations to evaluate
            sample_size: Number of users to sample for evaluation (None for all)
            
        Returns:
            dict: Dictionary containing evaluation results
        """
        test_users = holdout_data['test_users']
        
        if not test_users:
            print("No test users available for evaluation")
            return {metric: 0.0 for metric in self.metrics}
        
        # Sample users if requested
        if sample_size and sample_size < len(test_users):
            sampled_users = random.sample(test_users, sample_size)
        else:
            sampled_users = test_users
            
        # Initialize results
        metrics_results = {metric: [] for metric in self.metrics}
        metrics_results['latency'] = []
        all_recommended_ids = []
        
        # Evaluate each user
        for user_id in tqdm(sampled_users, desc="Evaluating users"):
            # Get user history and holdout
            user_data = holdout_data['user_histories'].get(user_id)
            if not user_data:
                continue
                
            history = user_data['history']
            holdout = user_data['holdout']
            
            # Skip if no holdout items
            if len(holdout) == 0:
                continue
                
            # Evaluate user
            user_metrics, recommended_ids = self.evaluate_user(user_id, history, holdout, k)
            
            # Record results
            for metric in self.metrics:
                if metric in user_metrics:
                    metrics_results[metric].append(user_metrics[metric])
                    
            metrics_results['latency'].append(user_metrics.get('latency', 0))
            all_recommended_ids.append(recommended_ids)
        
        # Calculate average metrics
        avg_results = {
            metric: np.mean(values) if values else 0.0
            for metric, values in metrics_results.items()
        }
        
        # Calculate coverage
        if 'coverage' in self.metrics:
            avg_results['coverage'] = self.calculate_coverage(all_recommended_ids)
        
        # Add timestamp
        avg_results['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        avg_results['k'] = k
        avg_results['sample_size'] = len(sampled_users)
        
        return avg_results
    
    def compare_models(self, models_dict, holdout_data, k=10, sample_size=None):
        """
        Compare multiple recommendation models
        
        Args:
            models_dict: Dictionary mapping model names to model instances
            holdout_data: Dictionary containing holdout data
            k: Number of recommendations to evaluate
            sample_size: Number of users to sample (None for all)
            
        Returns:
            DataFrame: Comparison results
        """
        results = []
        
        for model_name, model in models_dict.items():
            print(f"Evaluating model: {model_name}")
            
            # Set the current model
            self.ranker = model
            
            # Evaluate the model
            model_results = self.evaluate(holdout_data, k, sample_size)
            model_results['model'] = model_name
            
            results.append(model_results)
            
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        return results_df
    
    def plot_results(self, results_df, metrics=None):
        """
        Plot comparison results
        
        Args:
            results_df: DataFrame containing comparison results
            metrics: List of metrics to plot (default: all except timestamp)
            
        Returns:
            matplotlib.figure.Figure: Plot figure
        """
        if metrics is None:
            metrics = [col for col in results_df.columns if col not in ['model', 'timestamp', 'k', 'sample_size']]
            
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 4, 5))
        
        if n_metrics == 1:
            axes = [axes]
            
        for i, metric in enumerate(metrics):
            if metric in results_df.columns:
                results_df.plot(x='model', y=metric, kind='bar', ax=axes[i], title=f'{metric.upper()} @{results_df["k"].iloc[0]}')
                axes[i].set_ylabel(metric)
                axes[i].grid(axis='y', linestyle='--', alpha=0.7)
                
        plt.tight_layout()
        return fig
    
    def save_results(self, results, filename=None):
        """
        Save evaluation results to a file
        
        Args:
            results: Dictionary or DataFrame containing results
            filename: Output filename (default: 'eval_results_TIMESTAMP.json')
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'eval_results_{timestamp}.json'
            
        if isinstance(results, pd.DataFrame):
            results_dict = results.to_dict(orient='records')
        else:
            results_dict = results
            
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
            
        print(f"Results saved to {filename}")

# Implementation of a simple baseline model
class PopularityRanker:
    """Ranks listings by popularity (number of reviews)"""
    
    def __init__(self, df_reviews):
        # Count reviews per listing
        self.popularity = df_reviews['listing_id'].value_counts()
        
    def retrieve_candidates(self, user_history, candidates, interaction_data=None, top_k=5):
        """Rank listings by popularity"""
        # Calculate popularity score for each candidate
        candidates = candidates.copy()
        candidates['popularity_score'] = candidates['listing_id'].map(
            lambda x: self.popularity.get(x, 0)
        )
        
        # Sort by popularity (descending) and return top k
        return candidates.sort_values('popularity_score', ascending=False).head(top_k)

class RandomRanker:
    """Ranks listings randomly"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def retrieve_candidates(self, user_history, candidates, interaction_data=None, top_k=5):
        """Rank listings randomly"""
        # Return random sample of candidates with fixed seed
        return candidates.sample(frac=1, random_state=self.random_state).reset_index(drop=True).head(top_k)

# Example usage script
def run_evaluation(df_listings, df_reviews, llm_model=None, embedding_model='all-MiniLM-L6-v2', sample_size=200, random_state=42):
    """
    Run a complete evaluation of recommender systems
    
    Args:
        df_listings: DataFrame of listings
        df_reviews: DataFrame of reviews
        llm_model: Language model to use for LLM ranker
        embedding_model: Sentence transformer model to use for embeddings
        sample_size: Number of users to evaluate
        random_state: Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Comparison results
    """
    print("Initializing evaluation...")
    
    # Set random seeds for reproducibility
    random.seed(random_state)
    np.random.seed(random_state)
    
    # Initialize rankers
    llm_ranker = ListingRanker(listings_df=df_listings, reviews_df=df_reviews, embedding_model=embedding_model)
    popularity_ranker = PopularityRanker(df_reviews)
    random_ranker = RandomRanker(random_state=random_state)
    
    # Initialize evaluator
    evaluator = RecommenderEvaluator(
        ranker=llm_ranker,
        df_listings=df_listings,
        df_reviews=df_reviews
    )
    
    print("Preparing holdout data...")
    holdout_data = evaluator.prepare_holdout_set(
        test_size=0.8,
        min_reviews=3
    )
    
    if not holdout_data['test_users']:
        print("Not enough data for evaluation")
        return None
        
    print(f"Prepared holdout data for {len(holdout_data['test_users'])} test users")
    
    # Compare models
    print("Comparing models...")
    models = {
        'LLM_Ranker': llm_ranker,
        'Popularity': popularity_ranker,
        'Random': random_ranker
    }
    
    comparison_results = evaluator.compare_models(
        models_dict=models,
        holdout_data=holdout_data,
        k=10,
        sample_size=sample_size
    )
    
    # Add model information
    comparison_results['llm_model'] = llm_model if llm_model else 'phi3'
    comparison_results['embedding_model'] = embedding_model
    
    print("\nModel comparison:")
    print(comparison_results[['model', 'ndcg', 'precision', 'recall', 'diversity', 'coverage', 'mrr', 'latency', 'llm_model', 'embedding_model']])
    
    # Plot comparison
    print("Generating comparison plot...")
    fig = evaluator.plot_results(comparison_results, metrics=['ndcg', 'precision', 'recall', 'diversity', 'mrr'])
    plt.savefig('model_output/model_comparison.png')
    print("Comparison plot saved to model_output/model_comparison.png")
    
    # Save results
    evaluator.save_results(comparison_results, 'model_output/recommender_comparison_results.json')
    
    return comparison_results