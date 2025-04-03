import ollama
import itertools
import json
import pandas as pd
import numpy as np
from pprint import pprint
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm as sparse_norm

class ListingRanker:
    """Ranks listings using semantic similarity and collaborative filtering"""
    
    def __init__(self, listings_df, reviews_df, embedding_model='all-MiniLM-L6-v2'):
        """
        Initialize the ranker
        
        Args:
            listings_df: DataFrame containing listing information
            reviews_df: DataFrame containing user reviews
            embedding_model: Name of the sentence transformer model to use
        """
        self.listings_df = listings_df
        self.reviews_df = reviews_df
        self.embedding_model = SentenceTransformer(embedding_model)
        self.item_embeddings = None
        self.item_relationship_matrix = None
        self.user_item_matrix = None
        self.listing_id_to_idx = None
        self.idx_to_listing_id = None

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
        # Compute relationship matrices if not already done
        if self.item_relationship_matrix is None:
            self.compute_semantic_relationship_matrix()
        
        if self.user_item_matrix is None:
            self.compute_collaborative_relationship_matrix(interaction_data)
        
        # Calculate scores for each candidate
        scores = []
        for _, candidate in candidates.iterrows():
            total_score = 0
            candidate_idx = self.listing_id_to_idx[candidate['listing_id']]
            
            for _, history_item in user_history.iterrows():
                history_idx = self.listing_id_to_idx[history_item['listing_id']]
                
                # Semantic relationship
                semantic_score = self.item_relationship_matrix[candidate_idx, history_idx]
                
                # Collaborative relationship - compute on demand
                collab_score = self._compute_collaborative_similarity(candidate_idx, history_idx)
                
                # Combine scores
                total_score += alpha * semantic_score + (1 - alpha) * collab_score
            
            scores.append({
                'listing_id': candidate['listing_id'],
                'score': total_score / len(user_history)
            })
        
        # Convert scores to DataFrame and sort
        scores_df = pd.DataFrame(scores)
        ranked_candidates = candidates.merge(scores_df, on='listing_id').sort_values('score', ascending=False)
        
        return ranked_candidates.head(top_k)