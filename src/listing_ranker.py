import ollama
import re
import itertools
import json
import pandas as pd
import numpy as np
from pprint import pprint
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
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
    
    def __init__(self, listings_df, reviews_df, embedding_model='all-MiniLM-L6-v2', llm_model='phi3', max_concurrent_calls=5):
        """
        Initialize the ranker
        
        Args:
            listings_df: DataFrame containing listing information
            reviews_df: DataFrame containing user reviews
            embedding_model: Name of the sentence transformer model to use
            llm_model: Name of the LLM model to use for ranking decisions
            max_concurrent_calls: Maximum number of concurrent LLM API calls (default: 5)
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
        self.llm_call_count = 0  # Track number of LLM calls
        self.session = None  # For async HTTP requests
        self.total_llm_calls = 0  # Track total LLM calls made
        self.completed_llm_calls = 0  # Track completed LLM calls
        self.loop = None  # Store the event loop
        self.start_time = None  # Track start time for monitoring
        self.error_count = 0  # Track number of errors
        self.last_monitor_time = 0  # Track last monitoring time
        self.monitor_interval = 1.0  # Monitor every second
        self.logger = logging.getLogger(__name__)
        self.retry_count = 0  # Track retry attempts
        self.max_retries = 3  # Maximum number of retries per call

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

    def _generate_item_embeddings(self, item_prompts):
        """Generate embeddings for items in batches with memory management."""
        batch_size = 16  # Reduced batch size
        all_embeddings = []
        
        for i in range(0, len(item_prompts), batch_size):
            batch_prompts = item_prompts[i:i + batch_size].tolist()
            batch_embeddings = self.embedding_model.encode(batch_prompts, show_progress_bar=True)
            all_embeddings.append(batch_embeddings)
            
            # Clear memory after each batch
            import gc
            gc.collect()
        
        return np.vstack(all_embeddings)

    def _create_user_item_matrix(self, interaction_data):
        """Create user-item interaction matrix with memory efficiency."""
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
        
        return csr_matrix((data, (row, col)), shape=(len(reviewer_ids), len(listing_ids)))

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

    async def _async_llm_call(self, prompt):
        """
        Make an asynchronous call to the LLM API
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            str: The LLM's response
        """
        if self.session is None:
            self.logger.info("Creating new aiohttp session")
            self.session = aiohttp.ClientSession()
            
        # Implement retry logic
        for retry in range(self.max_retries):
            try:
                self.logger.debug(f"Making LLM API call to {self.llm_model} (attempt {retry+1}/{self.max_retries})")
                # Use aiohttp to make the API call
                async with self.session.post(
                    'http://localhost:11434/api/generate',
                    json={
                        'model': self.llm_model,
                        'prompt': prompt,
                        'stream': False
                    },
                    timeout=aiohttp.ClientTimeout(total=60)  # 60 second timeout
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.completed_llm_calls += 1
                        self.logger.debug(f"LLM call successful: {result.get('response', '')[:50]}...")
                        return result.get('response', '')
                    else:
                        self.logger.error(f"Error from LLM API: {response.status}")
                        # If we get a 429 (Too Many Requests), wait longer
                        if response.status == 429:
                            wait_time = 2 ** retry  # Exponential backoff
                            self.logger.info(f"Rate limited. Waiting {wait_time} seconds before retry...")
                            await asyncio.sleep(wait_time)
                            continue
                        # For other errors, increment error count
                        self.completed_llm_calls += 1
                        self.error_count += 1
                        return '0.5'  # Default neutral response
            except asyncio.TimeoutError:
                self.logger.error(f"LLM API call timed out (attempt {retry+1}/{self.max_retries})")
                # Wait before retrying
                wait_time = 2 ** retry
                self.logger.info(f"Timeout. Waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)
                continue
            except Exception as e:
                self.logger.error(f"Error in async LLM call: {e}")
                # Wait before retrying
                wait_time = 2 ** retry
                self.logger.info(f"Error. Waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)
                continue
        
        # If we've exhausted all retries, return default value
        self.logger.error("All retry attempts failed")
        self.completed_llm_calls += 1
        self.error_count += 1
        return '0.5'  # Default neutral response

    async def _get_llm_ranking_async(self, item1_desc, item2_desc):
        """
        Use LLM to determine which item is more likely to be preferred (async version)
        
        Args:
            item1_desc: Description of first item
            item2_desc: Description of second item
            
        Returns:
            float: Score indicating preference (higher means item1 is preferred)
        """
        self.llm_call_count += 1  # Increment call counter
        
        prompt = f"""Given these two Airbnb listings, which one would be more likely to be preferred by a user? 
        Consider factors like location, amenities, price, and overall quality.
        
        Listing 1:
        {item1_desc}
        
        Listing 2:
        {item2_desc}
        
        Respond with ONLY a number between 0 and 1, where:
        1.0 means Listing 1 is definitely preferred
        0.5 means they are equally good
        0.0 means Listing 2 is definitely preferred
        
        DO NOT include any explanation or text, just the number.
        """
        
        try:
            response_text = await self._async_llm_call(prompt)
            
            # Try to find a number between 0 and 1 in the response
            number_match = re.search(r'0\.\d+', response_text)
            if number_match:
                score = float(number_match.group(0))
            else:
                # If no decimal number found, try to find any number
                number_match = re.search(r'\d+', response_text)
                if number_match:
                    score = float(number_match.group(0)) / 100.0  # Assume it's a percentage
                else:
                    # Default to neutral score if no number found
                    score = 0.5
            
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"Error getting LLM ranking: {e}")
            return 0.5  # Return neutral score on error

    def _get_llm_ranking(self, item1_desc, item2_desc):
        """
        Synchronous wrapper for LLM ranking (for backward compatibility)
        
        Args:
            item1_desc: Description of first item
            item2_desc: Description of second item
            
        Returns:
            float: Score indicating preference (higher means item1 is preferred)
        """
        # Create a new event loop for this thread if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the async function in the event loop
        return loop.run_until_complete(self._get_llm_ranking_async(item1_desc, item2_desc))

    async def _process_candidate_async(self, candidate, user_history, alpha):
        """
        Process a single candidate asynchronously
        
        Args:
            candidate: The candidate listing to process
            user_history: The user's history
            alpha: Weight for semantic vs collaborative filtering
            
        Returns:
            dict: Score information for the candidate
        """
        try:
            total_score = 0
            candidate_idx = self.listing_id_to_idx[candidate['listing_id']]
            candidate_desc = self._construct_item_description(candidate)
            
            # Process history items sequentially to avoid overwhelming the API
            llm_scores = []
            
            for _, history_item in user_history.iterrows():
                history_idx = self.listing_id_to_idx[history_item['listing_id']]
                history_desc = self._construct_item_description(history_item)
                
                # Semantic relationship
                semantic_score = self.item_relationship_matrix[candidate_idx, history_idx]
                
                # Collaborative relationship
                collab_score = self._compute_collaborative_similarity(candidate_idx, history_idx)
                
                # Make LLM call
                llm_score = await self._get_llm_ranking_async(candidate_desc, history_desc)
                llm_scores.append(llm_score)
                
                # Add a small delay between calls to avoid overwhelming the API
                await asyncio.sleep(0.1)
            
            # Combine scores
            for i, llm_score in enumerate(llm_scores):
                history_idx = self.listing_id_to_idx[user_history.iloc[i]['listing_id']]
                semantic_score = self.item_relationship_matrix[candidate_idx, history_idx]
                collab_score = self._compute_collaborative_similarity(candidate_idx, history_idx)
                
                # Combine scores with weights
                total_score += (
                    alpha * semantic_score + 
                    (1 - alpha) * collab_score + 
                    llm_score
                ) / 3  # Average the three scores
            
            return {
                'listing_id': candidate['listing_id'],
                'score': total_score / len(user_history)
            }
        except Exception as e:
            self.logger.error(f"Error processing candidate {candidate.get('listing_id', 'unknown')}: {e}")
            return {
                'listing_id': candidate.get('listing_id', 'unknown'),
                'score': 0.0
            }

    async def retrieve_candidates_async(self, user_history, candidates, interaction_data, top_k=5, alpha=0.5):
        """
        Asynchronous version of retrieve_candidates
        
        Args:
            user_history (pd.DataFrame): DataFrame containing user's interaction history
            candidates (pd.DataFrame): DataFrame containing candidate items
            interaction_data (pd.DataFrame): Full interaction data for collaborative filtering
            top_k (int): Number of top recommendations to return
            alpha (float): Weight between semantic and collaborative relationships (0 to 1)
            
        Returns:
            pd.DataFrame: Top-k ranked candidate items with scores
        """
        # Compute relationship matrices if not already done
        if self.item_relationship_matrix is None:
            self.compute_semantic_relationship_matrix()
        
        if self.user_item_matrix is None:
            self.compute_collaborative_relationship_matrix(interaction_data)
        
        # Reset counters and start monitoring
        self.llm_call_count = 0
        self.completed_llm_calls = 0
        self.error_count = 0
        self.start_time = time.time()
        self.last_monitor_time = self.start_time
        
        # Calculate total number of comparisons for progress bar
        total_comparisons = len(candidates) * len(user_history)
        self.total_llm_calls = total_comparisons
        print(f"\nRanking {len(candidates)} candidates against {len(user_history)} history items")
        print(f"Total LLM calls to make: {total_comparisons}")
        print(f"Using {self.max_concurrent_calls} concurrent calls")
        print("Monitoring system resources (press Ctrl+C to stop)...")
        
        # Create progress bar for LLM calls
        llm_progress = tqdm(total=total_comparisons, desc="LLM calls", position=0, leave=True)
        
        # Process candidates in batches to control concurrency
        scores = []
        batch_size = self.max_concurrent_calls
        
        try:
            # Test a single LLM call first to verify connectivity
            self.logger.info("Testing LLM connectivity with a single call...")
            test_prompt = "Respond with the number 0.5"
            test_result = await self._async_llm_call(test_prompt)
            self.logger.info(f"Test LLM call result: {test_result}")
            
            # If the test call fails, reduce the batch size
            if test_result == '0.5' and self.error_count > 0:
                self.logger.warning("Test call failed. Reducing batch size to 1.")
                batch_size = 1
            
            for i in tqdm(range(0, len(candidates), batch_size), desc="Processing candidate batches", position=1, leave=True):
                batch = candidates.iloc[i:i+batch_size]
                
                # Create tasks for each candidate in the batch
                tasks = [self._process_candidate_async(candidate, user_history, alpha) 
                        for _, candidate in batch.iterrows()]
                
                # Wait for all tasks in the batch to complete
                self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(candidates) + batch_size - 1)//batch_size}")
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle any exceptions
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        self.logger.error(f"Error in batch {i//batch_size + 1}, item {j}: {result}")
                        batch_results[j] = {
                            'listing_id': batch.iloc[j].get('listing_id', 'unknown'),
                            'score': 0.0
                        }
                
                scores.extend(batch_results)
                
                # Update progress and monitor resources
                llm_progress.update(self.completed_llm_calls - llm_progress.n)
                self._monitor_resources()
                
                # Print more detailed progress
                if i % (batch_size * 5) == 0 or i == len(candidates) - batch_size:
                    self.logger.info(f"Processed {min(i + batch_size, len(candidates))}/{len(candidates)} candidates")
                    self.logger.info(f"Completed {self.completed_llm_calls}/{self.total_llm_calls} LLM calls")
                
                # Add a small delay between batches to avoid overwhelming the API
                if i + batch_size < len(candidates):
                    await asyncio.sleep(0.5)
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Saving partial results...")
        except Exception as e:
            self.logger.error(f"Error in retrieve_candidates_async: {e}")
        finally:
            # Close progress bars and print final stats
            llm_progress.close()
            elapsed_time = time.time() - self.start_time
            print(f"\nCompleted ranking with {self.llm_call_count} LLM calls in {elapsed_time:.1f} seconds")
            print(f"Average speed: {self.completed_llm_calls/elapsed_time:.2f} calls/second")
            print(f"Total errors: {self.error_count} ({(self.error_count/self.completed_llm_calls*100 if self.completed_llm_calls > 0 else 0):.1f}% error rate)")
            
            # Close the session if it was created
            if self.session:
                await self.session.close()
                self.session = None
        
        # Convert scores to DataFrame and sort
        scores_df = pd.DataFrame(scores)
        ranked_candidates = candidates.merge(scores_df, on='listing_id').sort_values('score', ascending=False)
        
        return ranked_candidates.head(top_k)

    def retrieve_candidates(self, user_history, candidates, interaction_data, top_k=5, alpha=0.5):
        """
        Retrieve and rank candidate items based on user history.
        
        Args:
            user_history (pd.DataFrame): DataFrame containing user's interaction history
            candidates (pd.DataFrame): DataFrame containing candidate items
            interaction_data (pd.DataFrame): Full interaction data for collaborative filtering
            top_k (int): Number of top recommendations to return
            alpha (float): Weight between semantic and collaborative relationships (0 to 1)
            
        Returns:
            pd.DataFrame: Top-k ranked candidate items with scores
        """
        # Get or create event loop
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        
        # Run the async function in the event loop
        return self.loop.run_until_complete(
            self.retrieve_candidates_async(user_history, candidates, interaction_data, top_k, alpha)
        )
