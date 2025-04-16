import os
import google.generativeai as genai
from dotenv import load_dotenv
import logging
from typing import Dict, List, Any, Optional
from app.models.qa_request import QARequest
from app.embeddings.embedding_service import EmbeddingService
from app.embeddings.vector_similarity_service import VectorSimilarityService
from app.services.template_store import TemplateStore
import markdown
import time
from google.api_core.exceptions import ResourceExhausted

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
logger.info("Loading environment variables for Chatbot Service")
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
if os.path.exists(dotenv_path):
    logger.info(f"Found .env file at {dotenv_path}")
    load_dotenv(dotenv_path)
else:
    logger.warning(f".env file not found at {dotenv_path}")
    load_dotenv()  # Try default locations

# Get API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    logger.info(f"GEMINI_API_KEY found with length: {len(GEMINI_API_KEY)}")
else:
    logger.error("GEMINI_API_KEY not found in environment variables")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Configure the model
generation_config = {
    "temperature": 0.1,  # Low temperature for more focused answers
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 2048,
}

# Initialize the model after API key is configured
model = genai.GenerativeModel("gemini-2.0-flash-001")

# Initialize embedding service and template store
embedding_service = EmbeddingService()
is_embedding_initialized = False

class ChatbotService:
    """Service for handling chatbot interactions."""
    
    # Initialize model at module level
    _model = genai.GenerativeModel('gemini-1.5-pro') if GEMINI_API_KEY else None
    
    # Track when rate limits were hit to avoid excessive requests
    _rate_limit_hit = False
    _rate_limit_time = 0
    _max_retries = 1
    
    @classmethod
    async def answer_question(cls, request):
        """
        Process a question about a markdown document and generate an answer.
        
        Args:
            request: QARequest object containing the question and document content
        
        Returns:
            Dict containing the answer and metadata
        """
        try:
            # Initial validation
            if not request.question:
                return {
                    "success": False,
                    "message": "Question is required",
                    "data": None
                }
                
            if not request.markdown_content:
                return {
                    "success": False,
                    "message": "Markdown content is required",
                    "data": None
                }
            
            # Model validation
            if not cls._model:
                if not GEMINI_API_KEY:
                    return {
                        "success": False,
                        "message": "GEMINI_API_KEY not configured",
                        "data": None
                    }
                else:
                    # Attempt to initialize model
                    try:
                        logger.info("Initializing Gemini model")
                        cls._model = genai.GenerativeModel('gemini-1.5-pro')
                    except Exception as e:
                        logger.error(f"Failed to initialize Gemini model: {str(e)}")
                        return {
                            "success": False,
                            "message": f"Failed to initialize Gemini model: {str(e)}",
                            "data": None
                        }
            
            # Check if we recently hit a rate limit and should wait
            if cls._rate_limit_hit and time.time() - cls._rate_limit_time < 60:
                wait_time = int(60 - (time.time() - cls._rate_limit_time))
                logger.warning(f"Rate limit cooling period active, {wait_time} seconds remaining")
                return {
                    "success": False,
                    "message": f"API rate limit reached. Please try again in {wait_time} seconds.",
                    "data": {
                        "rate_limited": True,
                        "wait_time": wait_time
                    }
                }
            
            # Create the prompt
            prompt = f"""
            You are an AI assistant tasked with answering questions about the following document. 
            Only answer questions based on information found in the document.
            If the information cannot be found in the document, please state that clearly.
            
            Document:
            {request.markdown_content}
            
            Question: {request.question}
            """
            
            logger.info(f"Processing question: {request.question[:50]}...")
            
            # Generate response with retry logic
            for attempt in range(cls._max_retries + 1):
                try:
                    # Generate response
                    response = cls._model.generate_content(prompt)
                    
                    # Reset rate limit flag if successful
                    cls._rate_limit_hit = False
                    
                    logger.info("Successfully generated response")
                    
                    # Format and return the response
                    return {
                        "success": True,
                        "message": "Answer generated successfully",
                        "data": {
                            "question": request.question,
                            "answer": response.text,
                            "source_document": request.source_document if hasattr(request, 'source_document') else None,
                        }
                    }
                
                except ResourceExhausted as e:
                    logger.warning(f"Rate limit hit (attempt {attempt+1}/{cls._max_retries+1}): {str(e)}")
                    cls._rate_limit_hit = True
                    cls._rate_limit_time = time.time()
                    
                    # If we have retries left, wait and try again
                    if attempt < cls._max_retries:
                        retry_delay = 5
                        logger.info(f"Waiting {retry_delay} seconds before retrying...")
                        time.sleep(retry_delay)
                    else:
                        # Return a helpful error message
                        return {
                            "success": False,
                            "message": "API rate limit exceeded. Please try again in a minute.",
                            "data": {
                                "rate_limited": True,
                                "error": str(e)
                            }
                        }
                
                except Exception as e:
                    logger.error(f"Error generating response (attempt {attempt+1}/{cls._max_retries+1}): {str(e)}")
                    
                    # If we have retries left, try again
                    if attempt < cls._max_retries:
                        time.sleep(2)
                    else:
                        raise
            
        except Exception as e:
            logger.error(f"Error in answer_question: {str(e)}", exc_info=True)
            return {
                "success": False,
                "message": f"Error generating answer: {str(e)}",
                "data": None
            }

    @staticmethod
    async def answer_standard_question(content: str, question: str) -> Dict:
        """
        Answers questions about a document using the Gemini model.
        
        Args:
            content: The markdown content of the document
            question: The question to ask about the document
            
        Returns:
            Dict containing the answer and metadata
        """
        try:
            logger.info("Processing standard question")
            
            # Format the prompt with the content and question
            prompt = f"""
You are a document analysis assistant. Your task is to answer questions about the provided document content accurately and concisely.

Document Content:
{content}

Question: {question}

Instructions:
1. Read the document content carefully
2. Answer the question based ONLY on the information present in the document
3. If the information is not in the document, say "This information is not available in the document"
4. Be precise and direct in your answers
5. Include relevant quotes or sections from the document when appropriate
6. If the question is unclear or ambiguous, ask for clarification

Answer:
"""
            
            # Generate response
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Extract the answer
            answer = response.text.strip()
            
            return {
                "success": True,
                "message": "Question answered successfully",
                "data": {
                    "answer": answer,
                    "confidence": 0.95,  # Placeholder for now
                    "source_documents": []  # Placeholder for now
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to process question: {str(e)}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed to process question: {str(e)}",
                "data": None
            }
            
    @staticmethod
    async def answer_question_with_custom_prompt(prompt: str) -> Dict:
        """
        Answer a question using a custom prompt.
        
        Args:
            prompt: Custom prompt with instructions, context, and question
            
        Returns:
            Dict containing the answer and metadata
        """
        try:
            logger.info("Processing question with custom prompt")
            
            # Generate response using the provided custom prompt
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Extract the answer
            answer = response.text.strip()
            
            return {
                "success": True,
                "message": "Question answered successfully",
                "data": {
                    "answer": answer,
                    "confidence": 0.95,  # Placeholder
                    "source_documents": []  # Placeholder
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to process question with custom prompt: {str(e)}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed to process question: {str(e)}",
                "data": None
            } 