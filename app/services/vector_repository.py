"""
Vector repository service using LangChain PGVector.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.core.database import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document as LangchainDocument
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class VectorRepository:
    """Repository for managing document vectors using LangChain PGVector."""
    
    def __init__(self):
        """Initialize the vector repository."""
        # Initialize OpenAI Embeddings
        self.embeddings = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Get database URL and handle Render's requirements
        database_url = os.getenv("DATABASE_URL")
        if database_url is None:
            logger.warning("DATABASE_URL not found in environment. Using default local PostgreSQL instance.")
            database_url = "postgresql://postgres:password@localhost:5432/chatbot"
        elif "render.com" in database_url and "sslmode" not in database_url:
            database_url += "?sslmode=require"
        
        # Initialize PGVector with correct settings
        self.vector_store = PGVector(
            collection_name="document_chunks",
            connection=database_url,
            embeddings=self.embeddings,
            use_jsonb=True,
            connection_args={
                "ssl": "require" if "render.com" in database_url else None
            }
        )
        
        logger.info("Vector repository initialized")
    
    async def store_document(
        self, 
        db: AsyncSession, 
        filename: str, 
        content: str, 
        chunks: List[Dict[str, Any]],
        description: Optional[str] = None
    ) -> int:
        """Store a document and its chunks with embeddings."""
        logger.info(f"Storing document: {filename} with {len(chunks)} chunks")
        
        # Create document record first using SQLAlchemy
        document = Document(
            filename=filename,
            description=description,
            content=content
        )
        db.add(document)
        await db.flush()
        document_id = document.id
        
        # Convert chunks to LangChain Documents
        langchain_docs = []
        for i, chunk in enumerate(chunks):
            # Create metadata with all the information we need
            # When creating metadata in store_document
            # When creating the document metadata:
            metadata = {
                "document_id": str(document_id),  # Convert to string
                "chunk_index": i,
                "filename": filename,
                "section": chunk["metadata"].get("section", "")
            }
            # Create LangChain document
            langchain_docs.append(
                LangchainDocument(
                    page_content=chunk["content"],
                    metadata=metadata
                )
            )
        
        # Store documents in vector store
        self.vector_store.add_documents(langchain_docs)
        
        # Commit the document record
        await db.commit()
        
        logger.info(f"Successfully stored document with ID: {document_id}")
        return document_id
    
    async def search_similar_chunks(
        self, 
        db: AsyncSession, 
        query: str, 
        limit: int = 5,
        document_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search for chunks similar to the query."""
        logger.info(f"Searching for chunks similar to: {query}")
        
        try:
            # Build filter criteria - don't convert to string!
            filter_criteria = {}
            if document_id is not None:
                logger.info(f"Filtering for document_id: {document_id} (type: {type(document_id)})")
                filter_criteria["document_id"] = document_id  # Keep as integer, don't convert to string
                
            # Use LangChain's similarity search - try without MMR first for simplicity
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=limit,
                filter=filter_criteria if filter_criteria else None
            )
            
            # Log raw results
            logger.info(f"Raw search returned {len(results)} results")
            
            # Format results
            chunks = []
            for doc, score in results:
                # For debugging add this log
                logger.info(f"Found document with metadata: {doc.metadata}")
                
                chunk = {
                    "id": doc.metadata.get("id", 0),
                    "content": doc.page_content,
                    "metadata": {
                        "section": doc.metadata.get("section", ""),
                        "chunk_index": doc.metadata.get("chunk_index", 0)
                    },
                    "filename": doc.metadata.get("filename", ""),
                    "document_id": doc.metadata.get("document_id"),
                    "similarity": 1.0 - score
                }
                chunks.append(chunk)
            
            logger.info(f"Found {len(chunks)} similar chunks")
            return chunks
                
        except Exception as e:
            logger.error(f"Error in search_similar_chunks: {str(e)}", exc_info=True)
            # If filtering fails, try again without filtering
            if document_id is not None:
                logger.info("Trying search without document filter as fallback")
                try:
                    # Try search without any filter
                    results = self.vector_store.similarity_search_with_score(
                        query=query,
                        k=limit
                    )
                    
                    # Filter results after search
                    chunks = []
                    for doc, score in results:
                        # Only include results matching the document_id
                        if str(doc.metadata.get("document_id")) == str(document_id):
                            chunk = {
                                "id": doc.metadata.get("id", 0),
                                "content": doc.page_content,
                                "metadata": {
                                    "section": doc.metadata.get("section", ""),
                                },
                                "filename": doc.metadata.get("filename", ""),
                                "document_id": doc.metadata.get("document_id"),
                                "similarity": 1.0 - score
                            }
                            chunks.append(chunk)
                    
                    logger.info(f"Fallback search found {len(chunks)} matching chunks")
                    return chunks
                except Exception as inner_e:
                    logger.error(f"Fallback search failed: {str(inner_e)}")
                    
            return []
        
    async def get_document_by_filename(
        self, 
        db: AsyncSession, 
        filename: str
    ) -> Optional[Dict[str, Any]]:
        """Get a document by filename."""
        logger.info(f"Getting document: {filename}")
        
        # Use SQLAlchemy to get the document
        result = await db.execute(
            select(Document).where(Document.filename == filename)
        )
        document = result.scalars().first()
        
        if not document:
            logger.warning(f"Document not found: {filename}")
            return None
        
        # Convert to dictionary
        document_dict = {
            "id": document.id,
            "filename": document.filename,
            "description": document.description,
            "content": document.content,
            "created_at": document.created_at,
            "updated_at": document.updated_at
        }
        
        return document_dict
    
    async def list_documents(
        self, 
        db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """List all documents."""
        logger.info("Listing all documents")
        
        # Use SQLAlchemy to get all documents
        result = await db.execute(select(Document))
        documents = result.scalars().all()
        
        # Convert to list of dictionaries
        document_list = []
        for document in documents:
            document_dict = {
                "id": document.id,
                "filename": document.filename,
                "description": document.description,
                "created_at": document.created_at,
                "updated_at": document.updated_at
            }
            document_list.append(document_dict)
        
        return document_list
    
    async def delete_document(
        self, 
        db: AsyncSession, 
        filename: str
    ) -> bool:
        """Delete a document and all its chunks."""
        logger.info(f"Deleting document: {filename}")
        
        # Find document by filename
        result = await db.execute(
            select(Document).where(Document.filename == filename)
        )
        document = result.scalars().first()
        
        if not document:
            logger.warning(f"Document not found: {filename}")
            return False
        
        # Delete from vector store by filter
        try:
            # Delete all chunks for this document
            self.vector_store.delete(
                filter={"document_id": document.id}
            )
        except Exception as e:
            logger.error(f"Error deleting from vector store: {str(e)}")
        
        # Delete document from database
        await db.delete(document)
        await db.commit()
        
        logger.info(f"Successfully deleted document: {filename}")
        return True