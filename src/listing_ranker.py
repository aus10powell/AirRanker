import ollama
import itertools
import json
import pandas as pd
import numpy as np
import logging
from pprint import pprint
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm as sparse_norm
from tqdm import tqdm
import torch

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

class ListingRanker:
    """Ranks listings using semantic similarity and collaborative filtering with optional pair-wise ranking"""
    
    def __init__(self, listings_df, reviews_df, embedding_model='all-MiniLM-L6-v2', llm_model='phi3.5'):
        """
        Initialize the ranker
        
        Args:
            listings_df: DataFrame containing listing information
            reviews_df: DataFrame containing user reviews
            embedding_model: Name of the sentence transformer model to use
            llm_model: Name of the LLM model to use for pairwise ranking
        """
        op = "ListingRanker.__init__"
        self.listings_df = listings_df
        self.reviews_df = reviews_df
        self.llm_model = llm_model
        
        # Check if MPS is available
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"op={op} Using device: {self.device}")
        
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        if self.device.type == "mps":
            self.embedding_model = self.embedding_model.to(self.device)
            
        self.item_embeddings = None
        self.item_relationship_matrix = None
        self.user_item_matrix = None
        self.listing_id_to_idx = None
        self.idx_to_listing_id = None

    def _generate_item_embeddings(self, item_prompts):
        """Generate embeddings for items using the sentence transformer model."""
        op = "ListingRanker._generate_item_embeddings"
        # Convert prompts to list if they're not already
        if isinstance(item_prompts, pd.Series):
            item_prompts = item_prompts.tolist()
            
        # Generate embeddings
        with torch.no_grad():
            embeddings = self.embedding_model.encode(item_prompts, convert_to_tensor=True, device=self.device)
            
        # Move embeddings to CPU for numpy operations if using MPS
        if self.device.type == "mps":
            embeddings = embeddings.cpu()
            
        # Convert to float32 before returning
        return embeddings.numpy().astype(np.float32)

    def compute_semantic_relationship_matrix(self):
        """Compute semantic relationships between items using embeddings."""
        op = "ListingRanker.compute_semantic_relationship_matrix"
        logger.info(f"op={op} Computing semantic relationship matrix...")
        
        if self.item_embeddings is None:
            logger.info(f"op={op} Generating item embeddings...")
            item_prompts = self.listings_df.apply(self._construct_item_description, axis=1)
            self.item_embeddings = self._generate_item_embeddings(item_prompts)
        
        # Store mapping between listing IDs and matrix indices
        self.listing_id_to_idx = {lid: idx for idx, lid in enumerate(self.listings_df['listing_id'])}
        self.idx_to_listing_id = {idx: lid for lid, idx in self.listing_id_to_idx.items()}
        
        # Initialize the relationship matrix with float32
        n_items = len(self.listings_df)
        self.item_relationship_matrix = np.zeros((n_items, n_items), dtype=np.float32)
        
        # Process in batches to avoid memory issues
        batch_size = 100  # Adjust based on available memory
        
        if self.device.type == "mps":
            logger.info(f"op={op} Using MPS for computation")
            # Convert embeddings to tensor for faster computation
            embeddings_tensor = torch.tensor(self.item_embeddings, dtype=torch.float32, device=self.device)
            
            # Process in batches
            for i in tqdm(range(0, n_items, batch_size), desc="Computing similarity matrix", position=0, leave=False):
                end_idx = min(i + batch_size, n_items)
                batch_size_actual = end_idx - i
                
                # Get batch of embeddings
                batch_embeddings = embeddings_tensor[i:end_idx]
                
                # Compute similarity for this batch against all items
                # Use a smaller sub-batch size for the second dimension to further reduce memory usage
                sub_batch_size = 50
                for j in range(0, n_items, sub_batch_size):
                    sub_end_idx = min(j + sub_batch_size, n_items)
                    sub_batch_embeddings = embeddings_tensor[j:sub_end_idx]
                    
                    # Compute cosine similarity on GPU
                    similarity_batch = torch.nn.functional.cosine_similarity(
                        batch_embeddings.unsqueeze(1),
                        sub_batch_embeddings.unsqueeze(0),
                        dim=2
                    )
                    
                    # Move back to CPU and store in the result matrix
                    self.item_relationship_matrix[i:end_idx, j:sub_end_idx] = similarity_batch.cpu().numpy()
                    
                    # Clear GPU memory
                    del similarity_batch
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Clear GPU memory after each batch
                del batch_embeddings
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        else:
            logger.info(f"op={op} Using CPU for computation")
            # Compute cosine similarity on CPU in batches
            for i in tqdm(range(0, n_items, batch_size), desc="Computing similarity matrix"):
                end_idx = min(i + batch_size, n_items)
                batch_embeddings = self.item_embeddings[i:end_idx]
                
                # Compute similarity for this batch against all items
                self.item_relationship_matrix[i:end_idx] = cosine_similarity(batch_embeddings, self.item_embeddings)
        
        logger.info(f"op={op} Semantic relationship matrix computation complete")
        return self.item_relationship_matrix

    def compute_collaborative_relationship_matrix(self, interaction_data):
        """
        Compute collaborative relationship matrix based on user interactions
        
        Args:
            interaction_data (pd.DataFrame): DataFrame of user-item interactions
        
        Returns:
            None: Matrix computation is deferred until needed
        """
        op = "ListingRanker.compute_collaborative_relationship_matrix"
        logger.info(f"op={op} Computing collaborative relationship matrix...")
        
        # Create user-item interaction matrix
        self.user_item_matrix = self._create_user_item_matrix(interaction_data)
        logger.info(f"op={op} Collaborative relationship matrix computation complete")

    def _create_user_item_matrix(self, interaction_data):
        """Create user-item interaction matrix with memory efficiency."""
        op = "ListingRanker._create_user_item_matrix"
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
        
        logger.info(f"op={op} Created user-item matrix with {len(reviewer_ids)} users and {len(listing_ids)} listings")
        return csr_matrix((data, (row, col)), shape=(len(reviewer_ids), len(listing_ids)))

    def _compute_collaborative_similarity(self, item1_idx, item2_idx):
        """Compute collaborative similarity between two items efficiently."""
        if self.user_item_matrix is None:
            return 0.0
        
        # Get the column vectors for both items
        item1_vec = self.user_item_matrix.getcol(item1_idx)
        item2_vec = self.user_item_matrix.getcol(item2_idx)
        
        if self.device.type == "mps":
            # Convert sparse vectors to dense tensors and move to MPS
            # Use a more memory-efficient approach by converting to dense only when needed
            item1_dense = item1_vec.toarray().flatten()
            item2_dense = item2_vec.toarray().flatten()
            
            # Check if vectors are non-zero before converting to tensor
            if np.any(item1_dense) and np.any(item2_dense):
                # Convert to float32 before creating tensors
                item1_tensor = torch.tensor(item1_dense.astype(np.float32), device=self.device)
                item2_tensor = torch.tensor(item2_dense.astype(np.float32), device=self.device)
                
                # Compute cosine similarity on GPU
                dot_product = torch.dot(item1_tensor, item2_tensor)
                norm1 = torch.norm(item1_tensor)
                norm2 = torch.norm(item2_tensor)
                
                # Clear tensors to free memory
                del item1_tensor, item2_tensor
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                    
                return (dot_product / (norm1 * norm2)).item()
            else:
                return 0.0
        else:
            # Compute cosine similarity on CPU
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

    def _compare_pair_with_llm(self, item1, item2, user_history):
        """Compare two items using LLM to determine which is more relevant."""
        op = "ListingRanker._compare_pair_with_llm"
        # Convert listing IDs to strings before joining
        user_history_str = [str(uid) for uid in user_history]
        
        prompt = f"""Given a user's purchase history and two items, determine which item is more relevant to recommend next.
User's purchase history: {', '.join(user_history_str)}
Item 1: {item1['name']} - {item1['description']}
Item 2: {item2['name']} - {item2['description']}

Which item is more relevant to recommend next? Answer with just '1' or '2'."""

        logger.debug(f"op={op} Sending prompt to LLM: {prompt[:100]}...")
        response = ollama.chat(model=f'{self.llm_model}:latest', messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ])
        
        # Parse response to determine if items should be swapped
        answer = response['message']['content'].strip().lower()
        logger.debug(f"op={op} LLM response: {answer}")
        return answer == '2'  # Return True if items should be swapped

    def _get_initial_scores(self, user_history, candidates, alpha=0.5):
        """
        Calculate initial scores for candidates based on user history.
        
        Args:
            user_history (pd.DataFrame): DataFrame containing user's interaction history
            candidates (pd.DataFrame): DataFrame containing candidate items
            alpha (float): Weight between semantic and collaborative relationships (0 to 1)
            
        Returns:
            pd.DataFrame: Candidates with their scores
        """
        op = "ListingRanker._get_initial_scores"
        logger.info(f"op={op} Calculating initial scores for {len(candidates)} candidates with alpha={alpha}")
        
        scores = []
        
        # Get history indices once
        history_indices = [self.listing_id_to_idx[history_item['listing_id']] for _, history_item in user_history.iterrows()]
        
        if self.device.type == "mps":
            logger.info(f"op={op} Using MPS for score calculation")
            # Convert relationship matrix to float32 before creating tensor
            relationship_matrix_float32 = self.item_relationship_matrix.astype(np.float32)
            relationship_matrix = torch.tensor(relationship_matrix_float32, device=self.device)
            
            # Process candidates in smaller batches for memory efficiency
            batch_size = 32
            for i in range(0, len(candidates), batch_size):
                batch_candidates = candidates.iloc[i:i+batch_size]
                batch_scores = []
                
                # Get candidate indices for this batch
                candidate_indices = [self.listing_id_to_idx[cid] for cid in batch_candidates['listing_id']]
                
                # Get semantic scores for all candidates in batch at once
                semantic_scores = relationship_matrix[candidate_indices][:, history_indices]
                
                # Process collaborative scores in smaller sub-batches
                sub_batch_size = 10
                collab_scores_list = []
                
                for j in range(0, len(candidate_indices), sub_batch_size):
                    sub_candidate_indices = candidate_indices[j:j+sub_batch_size]
                    
                    # Compute collaborative scores for this sub-batch
                    sub_collab_scores = []
                    for candidate_idx in sub_candidate_indices:
                        sub_collab_scores.append([
                            self._compute_collaborative_similarity(candidate_idx, history_idx)
                            for history_idx in history_indices
                        ])
                    
                    # Convert to tensor and move to device
                    sub_collab_tensor = torch.tensor(sub_collab_scores, dtype=torch.float32, device=self.device)
                    collab_scores_list.append(sub_collab_tensor)
                    
                    # Clear memory
                    del sub_collab_scores
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Concatenate all collaborative score tensors
                collab_scores = torch.cat(collab_scores_list, dim=0)
                
                # Combine scores
                combined_scores = alpha * semantic_scores + (1 - alpha) * collab_scores
                total_scores = torch.mean(combined_scores, dim=1)
                
                # Create score entries
                for j, candidate_idx in enumerate(candidate_indices):
                    batch_scores.append({
                        'listing_id': self.idx_to_listing_id[candidate_idx],
                        'score': total_scores[j].item()
                    })
                
                scores.extend(batch_scores)
                
                # Clear memory
                del semantic_scores, collab_scores, combined_scores, total_scores
                del collab_scores_list
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        else:
            logger.info(f"op={op} Using CPU for score calculation")
            # Original CPU implementation
            for _, candidate in candidates.iterrows():
                total_score = 0
                candidate_idx = self.listing_id_to_idx[candidate['listing_id']]
                
                for _, history_item in user_history.iterrows():
                    history_idx = self.listing_id_to_idx[history_item['listing_id']]
                    
                    # Semantic relationship
                    semantic_score = self.item_relationship_matrix[candidate_idx, history_idx]
                    
                    # Collaborative relationship
                    collab_score = self._compute_collaborative_similarity(candidate_idx, history_idx)
                    
                    # Combine scores
                    total_score += alpha * semantic_score + (1 - alpha) * collab_score
                
                scores.append({
                    'listing_id': candidate['listing_id'],
                    'score': total_score / len(user_history)
                })
        
        # Convert scores to DataFrame and merge with candidates
        scores_df = pd.DataFrame(scores)
        result = candidates.merge(scores_df, on='listing_id').sort_values('score', ascending=False)
        logger.info(f"op={op} Initial score calculation complete")
        return result

    def retrieve_candidates(self, user_history, candidates, interaction_data, top_k=5, alpha=0.5, use_pairwise=False, top_k_multiplier=3):
        """
        Retrieve and rank candidate items based on user history.
        
        Args:
            user_history (pd.DataFrame): DataFrame containing user's interaction history
            candidates (pd.DataFrame): DataFrame containing candidate items
            interaction_data (pd.DataFrame): Full interaction data for collaborative filtering
            top_k (int): Number of top recommendations to return
            alpha (float): Weight between semantic and collaborative relationships (0 to 1)
            use_pairwise (bool): Whether to use pair-wise ranking with LLM
            top_k_multiplier (int): Multiplier for top_k when using pairwise ranking to limit LLM calls
            
        Returns:
            pd.DataFrame: Top-k ranked candidate items with scores
        """
        op = "ListingRanker.retrieve_candidates"
        logger.info(f"op={op} Retrieving candidates with top_k={top_k}, alpha={alpha}, use_pairwise={use_pairwise}")
        
        # Compute relationship matrices if not already done
        if self.item_relationship_matrix is None:
            self.compute_semantic_relationship_matrix()
        
        if self.user_item_matrix is None:
            self.compute_collaborative_relationship_matrix(interaction_data)
        
        # Get initial scores
        ranked_candidates = self._get_initial_scores(user_history, candidates, alpha)
        logger.info(f"op={op} First 3 initial scores (listing_id, score): {ranked_candidates[['listing_id', 'score']].head(3)}")
        
        # Apply pair-wise ranking if requested
        if use_pairwise:
            logger.info(f"op={op} Using LLM model '{self.llm_model}' for pair-wise ranking...")

            logger.info(f"op={op}. retrieving top {top_k*top_k_multiplier} candidates for pairwise ranking from {len(ranked_candidates)} candidates")   
            reduced_ranked_candidates = ranked_candidates.head(top_k*top_k_multiplier)
            logger.info(f"op={op} num reduced_ranked_candidates={len(reduced_ranked_candidates)}")

            # Use sliding window of size 2 to compare adjacent pairs
            for i in tqdm(range(len(reduced_ranked_candidates) - 1), desc="Pair-wise ranking", position=0, leave=True):
                item1 = reduced_ranked_candidates.iloc[i]
                item2 = reduced_ranked_candidates.iloc[i + 1]
                
                # Compare pair and swap if needed
                if self._compare_pair_with_llm(item1, item2, user_history['listing_id'].tolist()):
                    reduced_ranked_candidates.iloc[i], reduced_ranked_candidates.iloc[i + 1] = reduced_ranked_candidates.iloc[i + 1], reduced_ranked_candidates.iloc[i]
            
            ranked_candidates = reduced_ranked_candidates.head(top_k)
            logger.info(f"op={op} Pair-wise ranking complete")
        
        logger.info(f"op={op} Returning top {top_k} candidates")
        return ranked_candidates.head(top_k)
        
    def _compare_pair_with_query(self, item1, item2, query_text):
        """Compare two items using LLM to determine which is more relevant to a query."""
        op = "ListingRanker._compare_pair_with_query"
        
        prompt = f"""Given a user's query and two items, determine which item is more relevant to the query.
User's query: {query_text}
Item 1: {item1['name']} - {item1['description']}
Item 2: {item2['name']} - {item2['description']}

Which item is more relevant to the query? Answer with just '1' or '2'."""

        logger.debug(f"op={op} Sending prompt to LLM: {prompt[:100]}...")
        response = ollama.chat(model=f'{self.llm_model}:latest', messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ])
        
        # Parse response to determine if items should be swapped
        answer = response['message']['content'].strip().lower()
        logger.debug(f"op={op} LLM response: {answer}")
        return answer == '2'  # Return True if items should be swapped

    def retrieve_by_query(self, query_text, candidates, interaction_data, top_k=5, alpha=0.5, use_pairwise=False, top_k_multiplier=3):
        """
        Retrieve and rank candidate items based on a direct query text.
        
        Args:
            query_text (str): The query text to use for ranking
            candidates (pd.DataFrame): DataFrame containing candidate items
            interaction_data (pd.DataFrame): Full interaction data for collaborative filtering
            top_k (int): Number of top recommendations to return
            alpha (float): Weight between semantic and collaborative relationships (0 to 1)
            use_pairwise (bool): Whether to use pair-wise ranking with LLM
            top_k_multiplier (int): Multiplier for top_k when using pairwise ranking to limit LLM calls
            
        Returns:
            pd.DataFrame: Top-k ranked candidate items with scores
        """
        op = "ListingRanker.retrieve_by_query"
        logger.info(f"op={op} Retrieving candidates with query text, top_k={top_k}, alpha={alpha}, use_pairwise={use_pairwise}")
        
        # Compute relationship matrices if not already done
        if self.item_relationship_matrix is None:
            self.compute_semantic_relationship_matrix()
        
        if self.user_item_matrix is None:
            self.compute_collaborative_relationship_matrix(interaction_data)
        
        # For query-based retrieval, we'll use a different approach than user history
        # We'll directly compute semantic similarity between the query and each candidate
        
        # Generate embedding for the query
        query_embedding = self._generate_item_embeddings([query_text])[0]
        
        # Generate embeddings for all candidates
        candidate_prompts = candidates.apply(self._construct_item_description, axis=1)
        candidate_embeddings = self._generate_item_embeddings(candidate_prompts)
        
        # Compute cosine similarity between query and all candidates
        similarities = []
        for i, candidate_embedding in enumerate(candidate_embeddings):
            similarity = np.dot(query_embedding, candidate_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(candidate_embedding)
            )
            similarities.append(similarity)
        
        # Create a DataFrame with scores
        scores_df = pd.DataFrame({
            'listing_id': candidates['listing_id'],
            'score': similarities
        })
        
        # Sort by score
        ranked_candidates = candidates.merge(scores_df, on='listing_id').sort_values('score', ascending=False)
        
        # Apply pair-wise ranking if requested
        if use_pairwise:
            logger.info(f"op={op} Using LLM model '{self.llm_model}' for pair-wise ranking...")
            
            # Take top candidates for pairwise ranking
            logger.info(f"op={op}. retrieving top {top_k*top_k_multiplier} candidates for pairwise ranking from {len(ranked_candidates)} candidates")
            top_candidates = ranked_candidates.head(top_k*top_k_multiplier)
            
            # Use sliding window of size 2 to compare adjacent pairs
            for i in range(len(top_candidates) - 1):
                item1 = top_candidates.iloc[i]
                item2 = top_candidates.iloc[i + 1]
                
                # Compare pair and swap if needed
                if self._compare_pair_with_query(item1, item2, query_text):
                    top_candidates.iloc[i], top_candidates.iloc[i + 1] = top_candidates.iloc[i + 1], top_candidates.iloc[i]
            
            ranked_candidates = top_candidates.head(top_k)
        
        logger.info(f"op={op} Returning top {top_k} candidates")
        return ranked_candidates.head(top_k)