import os
import logging
import numpy as np
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import time
import random
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# IMPORTANT: Explicitly load environment variables from the correct path
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
logger.info(f"Loading environment variables in embedding service from: {env_path}")
load_dotenv(env_path)

# Get API key with logging
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.error("GEMINI_API_KEY not found in environment variables for embedding service!")
else:
    logger.info(f"GEMINI_API_KEY found for embedding service with length: {len(api_key)}")

# Configure the Gemini API with explicit API key
genai.configure(api_key=api_key)

class EmbeddingService:
    def __init__(self):
        # Store the API key rather than loading it again
        self.api_key = api_key
        # Correct format for the specified model
        self.model = "models/gemini-embedding-exp-03-07"
        # Rate limiting parameters
        self.request_count = 0
        self.last_request_time = time.time()
        self.max_requests_per_minute = 10  # Adjust based on your quota
        
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
            sleep_time = 60 - time_diff + random.uniform(0.1, 1.0)
            logger.info(f"Rate limit approached, sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
            self.request_count = 0
            self.last_request_time = time.time()
        
    def generate_embedding(self, text: str, task_type: str = "SEMANTIC_SIMILARITY") -> Optional[List[float]]:
        """
        Generate an embedding for a text using Gemini API.
        
        Args:
            text: Text to generate embedding for
            task_type: The type of task for which the embedding will be used
                       Options: RETRIEVAL_QUERY, SEMANTIC_SIMILARITY, etc.
                       
        Returns:
            List of floating point numbers representing the embedding or None if error
        """
        try:
            # Apply rate limiting
            self._handle_rate_limit()
            
            logger.info(f"Generating embedding for text: '{text[:50]}...' with task_type: {task_type}")
            
            # Call the Gemini API to generate embedding - explicitly verify API key is set
            if not self.api_key:
                logger.error("API key not available for embedding generation")
                return None
                
            # Always reconfigure the API key to be safe
            genai.configure(api_key=self.api_key)
            
            # Call embed_content with the configured API key
            response = genai.embed_content(
                model=self.model,
                content=text,
                task_type=task_type
            )
            
            # Increment request counter
            self.request_count += 1
            
            # Extract the embedding values
            embedding = response["embedding"]
            logger.info(f"Successfully generated embedding with {len(embedding)} dimensions")
            
            # Return small sample of embedding for debugging
            if len(embedding) > 5:
                logger.info(f"Sample embedding values: {embedding[:5]}...")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}", exc_info=True)
            # If we hit rate limit, wait and retry once
            if "429" in str(e) or "Resource has been exhausted" in str(e):
                logger.warning("Rate limit hit, waiting 10 seconds and retrying once")
                time.sleep(10)
                try:
                    # Explicitly reconfigure API key for retry
                    genai.configure(api_key=self.api_key)
                    response = genai.embed_content(
                        model=self.model,
                        content=text,
                        task_type=task_type
                    )
                    embedding = response["embedding"]
                    logger.info(f"Retry successful, generated embedding with {len(embedding)} dimensions")
                    return embedding
                except Exception as retry_e:
                    logger.error(f"Retry failed: {str(retry_e)}")
            return None
            
    def batch_generate_embeddings(self, texts: List[str], task_type: str = "SEMANTIC_SIMILARITY") -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to generate embeddings for
            task_type: The task type for embeddings
            
        Returns:
            List of embeddings (or None for failed items)
        """
        results = []
        for i, text in enumerate(texts):
            logger.info(f"Generating embedding for text {i+1}/{len(texts)}")
            embedding = self.generate_embedding(text, task_type)
            results.append(embedding)
            # Add small delay between batch requests
            if i < len(texts) - 1:
                time.sleep(0.5)
        return results 