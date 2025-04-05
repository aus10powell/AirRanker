import ollama
import re
import itertools
import json
import pandas as pd
import numpy as np
from pprint import pprint
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, save_npz, load_npz
from scipy.sparse.linalg import norm as sparse_norm
from tqdm import tqdm
import logging
import asyncio
import aiohttp
import time
import nest_asyncio
import psutil
import os
import sys
from pathlib import Path
import pickle
from concurrent.futures import ThreadPoolExecutor
import gc

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Reduce HTTP logging verbosity
logging.getLogger("httpx").setLevel(logging.WARNING)

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)

class ListingRanker:
    """Ranks listings using semantic similarity and collaborative filtering"""
    
    def __init__(self, listings_df, reviews_df, embedding_model='all-MiniLM-L6-v2', llm_model='phi3', max_concurrent_calls=10):
        """
        Initialize the ranker with optimized parameters
        
        Args:
            listings_df: DataFrame containing listing information
            reviews_df: DataFrame containing user reviews
            embedding_model: Name of the sentence transformer model to use
            llm_model: Name of the LLM model to use for ranking decisions
            max_concurrent_calls: Maximum number of concurrent LLM API calls
        """
        self.listings_df = listings_df
        self.reviews_df = reviews_df
        self.embedding_model = SentenceTransformer(embedding_model)
        self.llm_model = llm_model
        self.max_concurrent_calls = max_concurrent_calls
        self.item_embeddings = None
        self.item_relationship_matrix = None
        self.user_item_matrix = None
        self.listing_id_to_idx = None
        self.idx_to_listing_id = None
        self.llm_call_count = 0
        self.session = None
        self.total_llm_calls = 0
        self.completed_llm_calls = 0
        self.loop = None
        self.start_time = None
        self.error_count = 0
        self.last_monitor_time = 0
        self.monitor_interval = 1.0
        self.logger = logging.getLogger(__name__)
        self.retry_count = 0
        self.max_retries = 3
        self.cache_dir = Path('cache')
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize cache paths
        self.embeddings_cache = self.cache_dir / f"embeddings_{embedding_model}.npz"
        self.relationship_cache = self.cache_dir / f"relationship_matrix_{embedding_model}.npy"
        self.user_item_cache = self.cache_dir / "user_item_matrix.npz"
        
        # Load or compute matrices
        self._load_or_compute_matrices()

    def _load_or_compute_matrices(self):
        """Load or compute the necessary matrices with caching"""
        # Load or compute embeddings
        if os.path.exists(self.embeddings_cache):
            self.logger.info("Loading embeddings from cache")
            self.item_embeddings = np.load(self.embeddings_cache)['arr_0']
        else:
            self.logger.info("Computing embeddings")
            item_prompts = self.listings_df.apply(self._construct_item_description, axis=1)
            self.item_embeddings = self._generate_item_embeddings(item_prompts)
            np.savez_compressed(self.embeddings_cache, self.item_embeddings)
        
        # Create ID mappings
        self.listing_id_to_idx = {lid: idx for idx, lid in enumerate(self.listings_df['listing_id'])}
        self.idx_to_listing_id = {idx: lid for lid, idx in self.listing_id_to_idx.items()}
        
        # Load or compute relationship matrix
        if os.path.exists(self.relationship_cache):
            self.logger.info("Loading relationship matrix from cache")
            self.item_relationship_matrix = np.load(self.relationship_cache)
        else:
            self.logger.info("Computing relationship matrix")
            self.item_relationship_matrix = cosine_similarity(self.item_embeddings)
            np.save(self.relationship_cache, self.item_relationship_matrix)
        
        # Load or compute user-item matrix
        if os.path.exists(self.user_item_cache):
            self.logger.info("Loading user-item matrix from cache")
            self.user_item_matrix = load_npz(self.user_item_cache)
        else:
            self.logger.info("Computing user-item matrix")
            self._create_user_item_matrix(self.reviews_df)
            save_npz(self.user_item_cache, self.user_item_matrix)

    def _generate_item_embeddings(self, item_prompts):
        """Generate embeddings for items in batches with improved memory management"""
        batch_size = 32  # Increased batch size
        all_embeddings = []
        
        print(f"Generating embeddings for {len(item_prompts)} items in batches of {batch_size}")
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(0, len(item_prompts), batch_size):
                batch_prompts = item_prompts[i:i + batch_size].tolist()
                futures.append(executor.submit(self.embedding_model.encode, batch_prompts, show_progress_bar=False))
            
            for future in tqdm(futures, desc="Generating embeddings"):
                batch_embeddings = future.result()
                all_embeddings.append(batch_embeddings)
                gc.collect()  # Force garbage collection after each batch
        
        print(f"Generated embeddings with shape: {np.vstack(all_embeddings).shape}")
        return np.vstack(all_embeddings)

    def _create_user_item_matrix(self, interaction_data):
        """Create user-item interaction matrix with improved memory efficiency"""
        # Create binary interactions (1 for each review)
        interactions = interaction_data.groupby(['reviewer_id', 'listing_id']).size().reset_index(name='interaction')
        
        # Get unique reviewer and listing IDs
        reviewer_ids = interactions['reviewer_id'].unique()
        listing_ids = self.listings_df['listing_id'].unique()
        
        # Create mappings
        reviewer_to_idx = {rid: idx for idx, rid in enumerate(reviewer_ids)}
        
        # Create sparse matrix
        row = [reviewer_to_idx[rid] for rid in interactions['reviewer_id']]
        col = [self.listing_id_to_idx[lid] for lid in interactions['listing_id']]
        data = interactions['interaction'].values
        
        self.user_item_matrix = csr_matrix((data, (row, col)), shape=(len(reviewer_ids), len(listing_ids)))

    async def _async_llm_call(self, prompt):
        """Make an asynchronous call to the LLM API with improved error handling"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        
        print(f"Making LLM call to {self.llm_model} with prompt: {prompt[:100]}...")
        
        for retry in range(self.max_retries):
            try:
                async with self.session.post(
                    'http://localhost:11434/api/generate',
                    json={
                        'model': self.llm_model,
                        'prompt': prompt,
                        'stream': False
                    },
                    timeout=aiohttp.ClientTimeout(total=30)  # Reduced timeout
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.completed_llm_calls += 1
                        print(f"LLM call successful: {result.get('response', '')[:50]}...")
                        return result.get('response', '')
                    else:
                        self.error_count += 1
                        print(f"LLM call failed with status {response.status}")
                        if retry < self.max_retries - 1:
                            await asyncio.sleep(1)  # Wait before retry
                            continue
                        return None
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Error in LLM call: {str(e)}")
                print(f"Error in LLM call: {str(e)}")
                if retry < self.max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                return None
        
        return None

    async def retrieve_candidates_async(self, user_history, candidates, interaction_data, top_k=10, alpha=0.5):
        """Retrieve candidates asynchronously with improved performance"""
        if self.start_time is None:
            self.start_time = time.time()
        
        print(f"\nProcessing recommendations for user with {len(user_history)} history items and {len(candidates)} candidates")
        
        # Pre-compute semantic similarities
        user_history_idx = [self.listing_id_to_idx[lid] for lid in user_history['listing_id']]
        candidate_idx = [self.listing_id_to_idx[lid] for lid in candidates['listing_id']]
        candidate_idx = np.array(candidate_idx)  # Convert to numpy array
        
        print("Computing semantic similarities...")
        semantic_scores = np.mean([
            self.item_relationship_matrix[idx][candidate_idx]
            for idx in user_history_idx
        ], axis=0)
        
        print("Computing collaborative similarities...")
        collaborative_scores = np.array([
            np.mean([
                self._compute_collaborative_similarity(idx, cidx)
                for idx in user_history_idx
            ])
            for cidx in candidate_idx
        ])
        
        # Get LLM scores for top candidates based on combined similarity
        initial_scores = alpha * semantic_scores + (1 - alpha) * collaborative_scores
        top_k_doubled = min(top_k * 2, len(candidates))  # Make sure we don't exceed the number of candidates
        top_initial_indices = np.argsort(initial_scores)[-top_k_doubled:][::-1]
        
        print(f"Getting LLM scores for top {len(top_initial_indices)} candidates...")
        llm_scores = np.zeros(len(top_initial_indices))
        
        # Get user preferences from history
        user_preferences = self._extract_user_preferences(user_history)
        
        # Try to get LLM scores, but have a fallback if LLM is not available
        llm_available = False
        try:
            # Test LLM availability with a simple prompt
            test_response = await self._async_llm_call("Respond with ONLY the number 0.5")
            llm_available = test_response is not None
        except Exception as e:
            print(f"LLM test call failed: {str(e)}")
            llm_available = False
        
        if llm_available:
            print("LLM is available, getting personalized scores...")
            # Get LLM scores for each candidate
            for i, idx in enumerate(top_initial_indices):
                try:
                    candidate = candidates.iloc[idx]
                    candidate_desc = self._construct_item_description(candidate)
                    
                    prompt = f"""Rate how well this Airbnb listing matches the user's preferences.

User preferences:
{user_preferences}

Listing to evaluate:
{candidate_desc}

IMPORTANT: Respond with ONLY a single number between 0.0 and 1.0.
Do not include any explanation or additional text.
Example response: 0.75"""
                    
                    llm_response = await self._async_llm_call(prompt)
                    try:
                        # Clean the response to extract just the number
                        cleaned_response = ''.join(c for c in llm_response if c.isdigit() or c == '.')
                        score = float(cleaned_response)
                        llm_scores[i] = max(0.0, min(1.0, score))  # Clamp between 0 and 1
                    except:
                        print(f"Failed to parse LLM score for candidate {idx}, using 0.5")
                        llm_scores[i] = 0.5  # Default score if parsing fails
                except Exception as e:
                    print(f"Error processing candidate {idx}: {str(e)}")
                    llm_scores[i] = 0.5  # Default score on error
            
            # Combine all scores with LLM
            final_scores = np.zeros(len(candidates))
            final_scores[top_initial_indices] = (
                0.4 * semantic_scores[top_initial_indices] +
                0.3 * collaborative_scores[top_initial_indices] +
                0.3 * llm_scores
            )
        else:
            print("LLM is not available, falling back to similarity-based ranking...")
            # Use only semantic and collaborative scores
            final_scores = initial_scores
        
        # Get top candidates
        top_indices = np.argsort(final_scores)[-top_k:][::-1]
        top_candidates = candidates.iloc[top_indices].copy()
        
        # Add scores to the DataFrame for evaluation
        top_candidates['score'] = final_scores[top_indices]
        
        print(f"Final recommendations: {len(top_candidates)} items with scores ranging from {top_candidates['score'].min():.3f} to {top_candidates['score'].max():.3f}")
        
        return top_candidates
        
    def _extract_user_preferences(self, user_history):
        """Extract user preferences from their history"""
        avg_price = user_history['listing_id'].map(self.listings_df.set_index('listing_id')['price']).mean()
        avg_bedrooms = user_history['listing_id'].map(self.listings_df.set_index('listing_id')['bedrooms']).mean()
        avg_accommodates = user_history['listing_id'].map(self.listings_df.set_index('listing_id')['accommodates']).mean()
        
        # Get most common room type
        room_types = user_history['listing_id'].map(self.listings_df.set_index('listing_id')['room_type'])
        most_common_room_type = room_types.mode().iloc[0] if not room_types.empty else "Any"
        
        # Get average ratings
        avg_cleanliness = user_history['listing_id'].map(self.listings_df.set_index('listing_id')['review_scores_cleanliness']).mean()
        avg_location = user_history['listing_id'].map(self.listings_df.set_index('listing_id')['review_scores_location']).mean()
        
        preferences = f"""Based on their history, this user prefers:
- Average price: ${avg_price:.2f}
- Room type: {most_common_room_type}
- Typical size: {avg_bedrooms:.1f} bedrooms, accommodates {avg_accommodates:.1f} guests
- Minimum cleanliness rating: {avg_cleanliness:.1f}
- Minimum location rating: {avg_location:.1f}"""
        
        return preferences

    def retrieve_candidates(self, user_history, candidates, interaction_data, top_k=5, alpha=0.5):
        """Synchronous wrapper for retrieve_candidates_async"""
        if self.loop is None:
            self.loop = asyncio.get_event_loop()
        return self.loop.run_until_complete(
            self.retrieve_candidates_async(user_history, candidates, interaction_data, top_k, alpha)
        )

    def _monitor_resources(self):
        """Monitor system resources and print status"""
        current_time = time.time()
        if current_time - self.last_monitor_time < self.monitor_interval:
            return
            
        self.last_monitor_time = current_time
        cpu_percent = psutil.cpu_percent()
        memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        elapsed_time = time.time() - self.start_time
        calls_per_second = self.completed_llm_calls / elapsed_time if elapsed_time > 0 else 0
        
        # Calculate error rate
        error_rate = (self.error_count / self.completed_llm_calls * 100) if self.completed_llm_calls > 0 else 0
        
        # Determine if we should adjust concurrent calls
        status = "OK"
        if cpu_percent > 90:
            status = "WARNING: High CPU usage"
        elif memory > 1000:  # More than 1GB
            status = "WARNING: High memory usage"
        elif error_rate > 5:
            status = "WARNING: High error rate"
        
        print(f"\rCPU: {cpu_percent}% | Memory: {memory:.1f}MB | "
              f"Calls/sec: {calls_per_second:.2f} | "
              f"Errors: {self.error_count} ({error_rate:.1f}%) | "
              f"Status: {status}", end="")

    def compute_semantic_relationship_matrix(self):
        """Compute semantic relationships between items using embeddings."""
        if self.item_embeddings is None:
            item_prompts = self.listings_df.apply(self._construct_item_description, axis=1)
            self.item_embeddings = self._generate_item_embeddings(item_prompts)
        
        # Store mapping between listing IDs and matrix indices
        self.listing_id_to_idx = {lid: idx for idx, lid in enumerate(self.listings_df['listing_id'])}
        self.idx_to_listing_id = {idx: lid for lid, idx in self.listing_id_to_idx.items()}
        
        # Compute cosine similarity between embeddings
        self.item_relationship_matrix = cosine_similarity(self.item_embeddings)
        return self.item_relationship_matrix

    def compute_collaborative_relationship_matrix(self, interaction_data):
        """
        Compute collaborative relationship matrix based on user interactions
        
        Args:
            interaction_data (pd.DataFrame): DataFrame of user-item interactions
        
        Returns:
            None: Matrix computation is deferred until needed
        """
        # Create user-item interaction matrix
        self.user_item_matrix = self._create_user_item_matrix(interaction_data)

    def _compute_collaborative_similarity(self, item1_idx, item2_idx):
        """Compute collaborative similarity between two items efficiently."""
        if self.user_item_matrix is None:
            return 0.0
        
        # Get the column vectors for both items
        item1_vec = self.user_item_matrix.getcol(item1_idx)
        item2_vec = self.user_item_matrix.getcol(item2_idx)
        
        # Compute cosine similarity
        dot_product = item1_vec.T.dot(item2_vec).toarray()[0, 0]
        norm1 = sparse_norm(item1_vec)
        norm2 = sparse_norm(item2_vec)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

    def _construct_item_description(self, item):
        """
        Construct a detailed prompt for item embedding
        
        Args:
            item (pd.Series): Series representing an item
        
        Returns:
            str: Formatted prompt for embedding
        """
        prompt = f"""
        Title: {item.get('name', 'N/A')}
        Description: {item.get('description', 'N/A')}
        Category: {item.get('room_type', 'N/A')}
        Price: ${item.get('price', 'N/A')}
        
        Location Information:
        - Location Score: {item.get('review_scores_location', 'N/A')}
        - Neighborhood: {item.get('neighbourhood', 'N/A')}
        - Transit: {item.get('transit', 'N/A')}
        
        Property Details:
        - Bedrooms: {item.get('bedrooms', 'N/A')}
        - Bathrooms: {item.get('bathrooms', 'N/A')}
        - Max Guests: {item.get('accommodates', 'N/A')}
        - Instant Bookable: {item.get('instant_bookable', 'N/A')}
        
        Review Scores:
        - Overall Rating: {item.get('review_scores_rating', 'N/A')}
        - Cleanliness: {item.get('review_scores_cleanliness', 'N/A')}
        - Communication: {item.get('review_scores_communication', 'N/A')}
        - Value: {item.get('review_scores_value', 'N/A')}
        - Accuracy: {item.get('review_scores_accuracy', 'N/A')}

        """
        return prompt
