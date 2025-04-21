"""
Chatbot service for answering questions about documents using OpenAI and function calling.
"""
import os
import logging
import time
import json
import asyncio
from typing import List, Dict, Any, Optional
from openai import OpenAI
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.services.embedding_service import EmbeddingService
from app.services.vector_repository import VectorRepository
from app.services.chunking_service import ChunkingService
from app.core.database import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Global initialization variables
is_initialized = False
initialization_time = time.time()

class ChatbotService:
    """Service for answering questions about documents using OpenAI."""
    
    def __init__(self):
        """Initialize the chatbot service."""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            raise ValueError("OPENAI_API_KEY not found")
            
        # Initialize client
        self.client = OpenAI(api_key=self.openai_api_key)
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
        logger.info(f"Using OpenAI model: {self.model_name}")
        
        # Initialize supporting services
        self.chunking_service = ChunkingService()
        self.embedding_service = EmbeddingService()
        self.vector_repository = VectorRepository()
        
        # Track service initialization state
        self.is_initialized = False
        self.is_initializing = False
        self.service_start_time = time.time()
        
    async def initialize(self):
        """Initialize the chatbot service during application startup."""
        if self.is_initialized or self.is_initializing:
            logger.info("ChatbotService initialization already in progress or completed")
            return
        
        try:
            self.is_initializing = True
            logger.info("Initializing ChatbotService...")
            
            # Test connection to OpenAI API
            test_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Respond with OK to test the connection."}
                ],
                max_tokens=10
            )
            
            logger.info(f"OpenAI API connection test successful: {test_response.choices[0].message.content}")
            
            # Initialize template store with embeddings if needed
            # from app.services.template_store import TemplateStore
            # TemplateStore.generate_embeddings(self.embedding_service)
            
            self.is_initialized = True
            logger.info("ChatbotService initialization complete")
            
        except Exception as e:
            logger.error(f"Error initializing ChatbotService: {str(e)}", exc_info=True)
            self.is_initializing = False
            raise
        finally:
            self.is_initializing = False
        
    async def process_upload(
        self, 
        db: AsyncSession, 
        filename: str, 
        content: str, 
        description: Optional[str] = None,
        reindex: bool = False
    ) -> Dict[str, Any]:
        """
        Process an uploaded markdown document by chunking and generating embeddings.
        
        Args:
            db: Database session
            filename: Filename of the document
            content: Text content of the document
            description: Optional description of the document
            reindex: Whether this is a reindexing operation rather than a new upload
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing upload for document: {filename}")
        
        try:
            # Check if document already exists (except during reindexing)
            if not reindex:
                existing_doc = await self.vector_repository.get_document_by_filename(db, filename)
                if existing_doc:
                    # Delete existing document first if it exists
                    logger.info(f"Document {filename} already exists, deleting it first")
                    await self.vector_repository.delete_document(db, filename)
            else:
                # For reindexing, always delete the existing document
                logger.info(f"Reindexing document {filename}, removing old version first")
                await self.vector_repository.delete_document(db, filename)
            
            # Split content into chunks
            chunks = self.chunking_service.chunk_markdown(content)
            logger.info(f"Document chunked into {len(chunks)} pieces")
            
            # Store document and chunks in vector database
            document_id = await self.vector_repository.store_document(
                db=db,
                filename=filename,
                content=content,
                chunks=chunks,
                description=description
            )
            
            if reindex:
                logger.info(f"Successfully reindexed document {filename} with ID {document_id}")
            else:
                logger.info(f"Successfully processed document {filename} with ID {document_id}")
            
            # Add a processing delay to let vector store operations complete before querying
            processing_delay = 1.0
            await asyncio.sleep(processing_delay)
            
            return {
                "success": True,
                "message": "Document processed successfully",
                "data": {
                    "filename": filename,
                    "chunks": len(chunks),
                    "document_id": document_id
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}", exc_info=True)
            return {
                "success": False,
                "message": f"Error processing document: {str(e)}",
                "data": None
            }
    
    async def answer_question(
        self, 
        db: AsyncSession, 
        question: str,
        document_id: Optional[int] = None,
        filename: Optional[str] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Answer a question about a document using RAG with function calling.
        
        Args:
            db: Database session
            question: The question to answer
            document_id: Optional ID of the document to query
            filename: Optional filename of the document to query
            chat_history: Optional list of previous chat messages
            
        Returns:
            Dictionary with the answer and related information
        """
        start_time = time.time()
        logger.info(f"Processing question: {question}")
        logger.info(f"Document context - ID: {document_id}, Filename: {filename}")
        
        # Check if we're still initializing
        global is_initialized, initialization_time
        uptime = time.time() - initialization_time
        in_progress = uptime < 60 and not is_initialized
        
        # Log the current state
        logger.info(f"Request received. Initialized: {is_initialized}, In progress: {in_progress}, Uptime: {uptime:.1f}s")
        
        # If we've been running for over 60 seconds, consider it initialized
        if uptime > 60 and not is_initialized:
            logger.info(f"Initialization taking too long ({uptime:.1f}s). Proceeding with request.")
            is_initialized = True
            self.is_initialized = True
        
        # Add tracking for document processing delay
        processing_delay = 0
        
        # Convert document_id to int if it's a string
        if document_id and isinstance(document_id, str) and document_id.isdigit():
            document_id = int(document_id)
            
        # If a filename is provided, get the document ID
        if filename and not document_id:
            doc = await self.vector_repository.get_document_by_filename(db, filename)
            if doc:
                document_id = doc["id"]
            else:
                return {
                    "success": False,
                    "message": f"Document not found: {filename}",
                    "data": None
                }
        
        # Get document info for context
        document_info = None
        if document_id:
            document_info = await self.vector_repository.get_document_by_id(db, document_id)
        elif filename:
            document_info = await self.vector_repository.get_document_by_filename(db, filename)
            
        # Log document length for context
        if filename:
            doc = await self.vector_repository.get_document_by_filename(db, filename)
            if doc:
                logger.info(f"Processing question: {question}... (document length: {len(doc['content'])} chars)")
                # For new documents, add a processing delay if the system just started
                if uptime < 60 and in_progress:
                    logger.info(f"New document detected (ID: {doc['id']}), adding processing delay")
                    processing_delay = 2
                    await asyncio.sleep(processing_delay)
                    
                # Log content size
                logger.info(f"Using full document content as we're on higher tier with increased quota limits")
        
        try:
            # Prepare chat messages with system prompt and history
            messages = self._prepare_chat_messages(
                question=question, 
                chat_history=chat_history,
                document_info=document_info
            )
            
            # Define tools for function calling
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "retrieve_document_chunks",
                        "description": "Retrieve chunks from documents that are relevant to the query",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The query to search for in the document"
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "Maximum number of chunks to retrieve"
                                }
                            },
                            "required": ["query"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_document_metadata",
                        "description": "Get metadata about a document",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "document_id": {
                                    "type": "integer",
                                    "description": "ID of the document"
                                }
                            },
                            "required": ["document_id"]
                        }
                    }
                }
            ]
            
            # Initialize response tracking
            tool_calls_remaining = 5  # Limit number of tool calls to prevent infinite loops
            all_used_contexts = []
            
            while tool_calls_remaining > 0:
                # Call OpenAI API
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.3,
                    tools=tools,
                    tool_choice="auto",
                )
                
                # Get assistant message
                assistant_message = response.choices[0].message
                
                # Add assistant message to messages list
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content or "",
                    "tool_calls": assistant_message.tool_calls
                })
                
                # If no tool calls, we have the final answer
                if not assistant_message.tool_calls:
                    logger.info("Final answer generated without additional tool calls")
                    break
                
                # Process tool calls
                for tool_call in assistant_message.tool_calls:
                    # Parse tool call
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    logger.info(f"Processing tool call: {function_name} with args: {function_args}")
                    
                    # Execute the appropriate function
                    if function_name == "retrieve_document_chunks":
                        # Get similar chunks for the query
                        query = function_args.get("query", question)
                        
                        # Keep search general without query-specific enhancements
                        chunks = await self.vector_repository.search_similar_chunks(
                            db=db,
                            query=query,
                            limit=function_args.get("limit", 5),
                            document_id=document_id
                        )
                        result = chunks
                        
                        # Log which document we're searching in
                        if document_id:
                            logger.info(f"Retrieving chunks specifically for document ID: {document_id}")
                        
                        all_used_contexts.extend([chunk["content"] for chunk in chunks])
                        
                    elif function_name == "get_document_metadata":
                        # Get document metadata
                        doc_id = function_args.get("document_id", document_id)
                        if doc_id:
                            # Find document by ID
                            query_result = await db.execute(
                                select(Document).where(Document.id == doc_id)
                            )
                            document = query_result.scalars().first()
                            
                            if document:
                                result = {
                                    "id": document.id,
                                    "filename": document.filename,
                                    "description": document.description,
                                    "created_at": document.created_at.isoformat(),
                                    "updated_at": document.updated_at.isoformat()
                                }
                            else:
                                result = {"error": "Document not found"}
                        else:
                            result = {"error": "No document ID provided"}
                    else:
                        result = f"Unknown function: {function_name}"
                    
                    # Add function result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps(result)
                    })
                
                # Decrement remaining tool calls
                tool_calls_remaining -= 1
            
            # Get the final answer
            final_answer = ""
            for msg in reversed(messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    final_answer = msg["content"]
                    break
            
            if not final_answer and tool_calls_remaining == 0:
                # If we ran out of tool calls, generate a final answer
                final_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.3,
                )
                final_answer = final_response.choices[0].message.content
            
            logger.info("Successfully generated response")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Set initialization flag after successful response
            is_initialized = True
            self.is_initialized = True
            
            return {
                "success": True,
                "message": "Question answered successfully",
                "data": {
                    "answer": final_answer,
                    "processing_time": processing_time,
                    "contexts_used": len(all_used_contexts),
                    "document_id": document_id,
                    "filename": filename
                }
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}", exc_info=True)
            return {
                "success": False,
                "message": f"Error answering question: {str(e)}",
                "data": None
            }
    
    def _prepare_chat_messages(
        self, 
        question: str, 
        chat_history: Optional[List[Dict[str, Any]]] = None,
        document_info: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Prepare chat messages with system prompt and history.
        
        Args:
            question: The current question
            chat_history: List of previous chat messages
            document_info: Information about the current document being queried
            
        Returns:
            List of messages for OpenAI API
        """
        # Create document context string
        document_context = ""
        if document_info:
            document_context = f"""
            CURRENT DOCUMENT CONTEXT:
            Filename: {document_info.get('filename', 'Unknown')}
            Document ID: {document_info.get('id', 'Unknown')}
            
            You are analyzing this specific document. All answers should be based on this document's content.
            """
        
        # System prompt for the chatbot
        system_prompt = f"""
        You are a Contract Analysis Chatbot specialized in analyzing legal documents and contracts.
        
        {document_context}
        
        Your task is to:
        1. Only provide information that is explicitly found in the CURRENT document being analyzed
        2. NEVER mix information between different contracts - each contract is a separate document
        3. Each question refers to a specific contract indicated in the interface - maintain strict document context
        4. If the current document doesn't contain the specific information requested, clearly state this
        5. When using content from documents, be precise and cite sections/clauses where possible
        
        Guidelines:
        - Your answers should be specific to only the current document being queried
        - Do not apply information from one contract to another, even if they appear similar
        - Be precise: if dates, parties, or specifics differ between contracts, reflect those differences
        - If you can't find relevant information in the current document chunks, say so directly
        - When providing answers, include the document name/ID to clarify which document you're referencing
        """
        
        # Initialize messages with system prompt
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add chat history if provided
        if chat_history:
            for message in chat_history:
                # Skip system messages in history
                if message.get("role") == "system":
                    continue
                    
                # Add user and assistant messages
                if message.get("role") in ["user", "assistant"]:
                    messages.append({
                        "role": message["role"],
                        "content": message["content"]
                    })
        
        # Add the current question
        messages.append({"role": "user", "content": question})
        
        return messages 