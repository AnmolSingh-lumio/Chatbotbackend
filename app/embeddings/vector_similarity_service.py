import logging
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class VectorSimilarityService:
    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            v1: First vector
            v2: Second vector
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            v1_array = np.array(v1)
            v2_array = np.array(v2)
            dot_product = np.dot(v1_array, v2_array)
            norm_v1 = np.linalg.norm(v1_array)
            norm_v2 = np.linalg.norm(v2_array)
            
            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0
                
            return dot_product / (norm_v1 * norm_v2)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}", exc_info=True)
            return 0.0
    
    @staticmethod
    def find_similar_questions(
        query_embedding: List[float],
        template_data: List[Dict[str, Any]],
        threshold: float = 0.7,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find the most similar questions to the query.
        
        Args:
            query_embedding: Embedding of the query
            template_data: List of dictionaries with template data including embeddings
            threshold: Minimum similarity score to consider
            top_k: Number of top matches to return
            
        Returns:
            List of dictionaries with template data and similarity scores
        """
        try:
            results = []
            logger.info(f"Finding similar questions among {len(template_data)} templates")
            
            for template in template_data:
                template_embedding = template.get("embedding")
                if not template_embedding:
                    continue
                    
                similarity = VectorSimilarityService.cosine_similarity(
                    query_embedding, template_embedding
                )
                
                logger.debug(f"Similarity with '{template['question_text'][:50]}...': {similarity:.4f}")
                
                if similarity >= threshold:
                    # Create a copy of the template data with the similarity score
                    result = template.copy()
                    result["similarity"] = similarity
                    results.append(result)
            
            # Sort by similarity in descending order
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Log top matches
            if results:
                logger.info(f"Found {len(results)} matches above threshold {threshold}")
                for i, match in enumerate(results[:top_k]):
                    logger.info(f"Match #{i+1}: '{match['question_text']}' with similarity {match['similarity']:.4f}")
            else:
                logger.info(f"No matches found above threshold {threshold}")
            
            # Return top k results
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar questions: {str(e)}", exc_info=True)
            return [] 