"""
Service for generating embeddings using OpenAI.
"""
import os
import logging
import time
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating embeddings using OpenAI."""
    
    def __init__(self):
        """Initialize the embedding service."""
        # Get OpenAI API key
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        # Initialize OpenAI client with only the required parameters
        # Removed any potential proxies parameter that might be causing issues
        self.client = OpenAI(
            api_key=self.api_key,
        )
        
        # Set embedding model name
        self.model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        logger.info(f"Using embedding model: {self.model_name}")
        
        # Set vector dimension
        self.vector_dimension = int(os.getenv("VECTOR_DIMENSION", "1536"))
        
        # Rate limiting parameters
        self.request_count = 0
        self.last_request_time = time.time()
        self.max_requests_per_minute = 60  # OpenAI limit (adjust based on tier)
        
    def _handle_rate_limit(self):
        """Handle rate limiting to avoid 429 errors."""
        current_time = time.time()
        time_diff = current_time - self.last_request_time
        
        # Reset counter if a minute has passed
        if time_diff > 60:
            self.request_count = 0
            self.last_request_time = current_time
            return
        
        # If we're approaching the rate limit, add delay
        if self.request_count >= self.max_requests_per_minute:
            # Sleep until the minute is up plus a small buffer
            sleep_time = 60 - time_diff + 1.0
            logger.info(f"Rate limit approached, sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
            self.request_count = 0
            self.last_request_time = time.time()
            
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floating-point numbers representing the embedding
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding generation")
            # Return zero vector with correct dimension
            return [0.0] * self.vector_dimension
        
        # Apply rate limiting
        self._handle_rate_limit()
        
        try:
            # Call OpenAI API to generate embedding
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text,
                encoding_format="float"
            )
            
            # Increment request count for rate limiting
            self.request_count += 1
            
            # Extract embedding from response
            embedding = response.data[0].embedding
            
            logger.info(f"Generated embedding with {len(embedding)} dimensions")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
            
    def batch_generate_embeddings(self, texts: List[str], batch_size: int = 8) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to generate embeddings for
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
            
        results = []
        
        # Process in batches to avoid rate limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Apply rate limiting
            self._handle_rate_limit()
            
            try:
                # Call OpenAI API to generate embeddings for batch
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                    encoding_format="float"
                )
                
                # Increment request count for rate limiting
                self.request_count += 1
                
                # Extract embeddings from response
                batch_embeddings = [data.embedding for data in response.data]
                results.extend(batch_embeddings)
                
                logger.info(f"Generated {len(batch_embeddings)} embeddings in batch")
                
                # Add a small delay between batches
                if i + batch_size < len(texts):
                    time.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Error generating batch embeddings: {str(e)}")
                # For failed batches, add zero vectors
                for _ in range(len(batch)):
                    results.append([0.0] * self.vector_dimension)
                    
        return results
        
    def similarity_score(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        if not embedding1 or not embedding2:
            return 0.0
            
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2) 