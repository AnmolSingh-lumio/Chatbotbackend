import os
import logging
import numpy as np
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import time
import random
import json
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

# Setup cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

class EmbeddingService:
    def __init__(self):
        # Store the API key rather than loading it again
        self.api_key = api_key
        # Correct format for the specified model
        self.model = "models/gemini-embedding-exp-03-07"
        # Rate limiting parameters - set to Google's documented limits
        self.request_count = 0
        self.last_request_time = time.time()
        self.max_requests_per_minute = 15  # Google's limit is 15 RPM/TPM
        
        # In-memory embedding cache
        self._embedding_cache = {}
        
        # Load embeddings from disk cache if available
        self._load_embedding_cache()
        
    def _load_embedding_cache(self):
        """Load embedding cache from disk."""
        cache_path = os.path.join(CACHE_DIR, "document_embeddings.json")
        template_cache_path = os.path.join(CACHE_DIR, "template_embeddings.json")
        
        # Try to load document embeddings
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    self._embedding_cache = json.load(f)
                logger.info(f"Loaded {len(self._embedding_cache)} cached document embeddings")
            except Exception as e:
                logger.error(f"Error loading document embedding cache: {str(e)}")
                self._embedding_cache = {}
                
        # Try to load template embeddings into the same cache
        if os.path.exists(template_cache_path):
            try:
                with open(template_cache_path, 'r') as f:
                    template_cache = json.load(f)
                    # Add template cache keys to main cache
                    for key, value in template_cache.items():
                        template_key = f"template_{key}"
                        self._embedding_cache[template_key] = value
                logger.info(f"Loaded {len(template_cache)} cached template embeddings")
            except Exception as e:
                logger.error(f"Error loading template embedding cache: {str(e)}")
    
    def _save_embedding_cache(self):
        """Save embedding cache to disk."""
        # Only save if we have a reasonable number of embeddings to avoid disk overhead
        if len(self._embedding_cache) > 0:
            cache_path = os.path.join(CACHE_DIR, "document_embeddings.json")
            try:
                # Limit cache size to avoid excessive disk usage
                if len(self._embedding_cache) > 500:
                    # Keep only the most recent 500 entries
                    sorted_keys = sorted(self._embedding_cache.keys())
                    for key in sorted_keys[:-500]:
                        del self._embedding_cache[key]
                        
                with open(cache_path, 'w') as f:
                    json.dump(self._embedding_cache, f)
                logger.info(f"Saved {len(self._embedding_cache)} embeddings to cache")
            except Exception as e:
                logger.error(f"Error saving embedding cache: {str(e)}")
    
    def save_template_embedding(self, template_id, embedding, force_save=False):
        """
        Save a template embedding to the cache file.
        
        Args:
            template_id: ID of the template
            embedding: The embedding vector
            force_save: Whether to force writing to file immediately
        """
        if not self.use_cache:
            return
        
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        template_cache_path = os.path.join(cache_dir, "template_embeddings.json")
        
        # Load existing cache if it exists
        cached_embeddings = {}
        if os.path.exists(template_cache_path):
            try:
                with open(template_cache_path, 'r') as f:
                    cached_embeddings = json.load(f)
            except Exception as e:
                logger.error(f"Error loading template embedding cache: {str(e)}")
        
        # Add/update the new embedding
        cached_embeddings[str(template_id)] = embedding
        
        # Always save to the cache file right away for templates
        # This ensures we don't lose progress between restarts
        try:
            with open(template_cache_path, 'w') as f:
                json.dump(cached_embeddings, f)
            logger.info(f"Saved template embedding for template {template_id} to cache")
            
            # Log a sample of the embedding for debugging
            sample_values = str(embedding[:5]) if embedding else "None"
            logger.info(f"Sample embedding values: {sample_values}...")
        except Exception as e:
            logger.error(f"Error saving template embedding cache: {str(e)}")
        
        return embedding
    
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
            # Sleep until the minute is up plus a small random buffer
            sleep_time = 60 - time_diff + random.uniform(1.0, 5.0)
            logger.info(f"Rate limit approached, sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
            self.request_count = 0
            self.last_request_time = time.time()
        
    def generate_embedding(self, text: str, task_type: str = "SEMANTIC_SIMILARITY", is_template: bool = False, template_id: str = None) -> Optional[List[float]]:
        """
        Generate an embedding for a text using Gemini API.
        
        Args:
            text: Text to generate embedding for
            task_type: The type of task for which the embedding will be used
            is_template: Whether this is a template embedding
            template_id: ID of the template if is_template is True
                       
        Returns:
            List of floating point numbers representing the embedding or None if error
        """
        # Create a cache key from the text and task type
        cache_key = f"{hash(text)}-{task_type}"
        if is_template and template_id:
            cache_key = f"template_{template_id}"
        
        # Check if we have a cached embedding
        if cache_key in self._embedding_cache:
            logger.info(f"Using cached embedding for text: '{text[:30]}...'")
            return self._embedding_cache[cache_key]
            
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
            
            # Cache the embedding
            self._embedding_cache[cache_key] = embedding
            
            # If it's a template, save to the dedicated template cache
            if is_template and template_id:
                self.save_template_embedding(template_id, embedding)
            # Otherwise, periodically save document embeddings
            elif len(self._embedding_cache) % 5 == 0:
                self._save_embedding_cache()
            
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
                    
                    # Cache the successful retry
                    self._embedding_cache[cache_key] = embedding
                    
                    # If it's a template, save to the dedicated template cache after retry
                    if is_template and template_id:
                        self.save_template_embedding(template_id, embedding)
                    
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
            # Add substantial delay between batch requests to avoid rate limits
            if i < len(texts) - 1:
                time.sleep(5.0)  # Increased delay to be safer
        
        # Save cache after batch processing
        self._save_embedding_cache()
        
        return results 