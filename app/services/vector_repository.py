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
        
        try:
            # Initialize PGVector with correct settings for the current version
            # Removing connection_args parameter which is not supported
            self.vector_store = PGVector(
                collection_name="document_chunks",
                connection=database_url,
                embeddings=self.embeddings,
                use_jsonb=True
            )
            logger.info("PGVector initialized successfully")
        except TypeError as e:
            # If there's a parameter error, try with fewer parameters
            logger.error(f"Error initializing PGVector: {str(e)}")
            logger.info("Trying with minimal parameters")
            self.vector_store = PGVector(
                collection_name="document_chunks",
                connection=database_url,
                embeddings=self.embeddings
            )
            logger.info("PGVector initialized with minimal parameters")
        
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
        """
        Search for chunks similar to the query.
        
        Args:
            db: Database session
            query: The search query
            limit: Maximum number of results to return
            document_id: Optional document ID to filter by
            
        Returns:
            List of chunks similar to the query
        """
        try:
            logger.info(f"Searching for chunks similar to: {query}")
            
            # If document_id is provided, filter by it
            if document_id is not None:
                logger.info(f"Filtering for document_id: {document_id} (type: {type(document_id)})")
                
                # Get document to confirm it exists
                doc_info = await self.get_document_by_id(db, document_id)
                if not doc_info:
                    logger.warning(f"Document with ID {document_id} not found")
                    return []
                
                logger.info(f"Searching in document: {doc_info.get('filename')}")
                
                # HYBRID SEARCH IMPLEMENTATION
                # 1. Perform vector similarity search
                vector_results = []
                try:
                    vector_results = self.vector_store.similarity_search_with_score(
                        query=query,
                        k=limit,
                        filter={"document_id": document_id}
                    )
                except Exception as e:
                    logger.warning(f"Vector search failed, falling back to keyword search: {str(e)}")
                
                # 2. Perform keyword search on the full document content
                keywords = self._extract_keywords(query)
                keyword_chunks = []
                
                if keywords:
                    logger.info(f"Performing keyword search with terms: {keywords}")
                    document_content = doc_info.get('content', '')
                    
                    # Create chunks from sections matching keywords
                    sections = self._split_by_headers(document_content)
                    for section_title, section_content in sections:
                        # Check if any keyword is in this section
                        if any(keyword.lower() in section_content.lower() for keyword in keywords):
                            keyword_chunks.append({
                                "content": section_content,
                                "metadata": {
                                    "section": section_title,
                                    "document_id": document_id,
                                    "filename": doc_info.get('filename'),
                                    "source": "keyword_search"
                                },
                                "score": 0.8  # Assign a default score for keyword matches
                            })
                
                # 3. Combine and deduplicate results
                results = vector_results
                logger.info(f"Raw search returned {len(results)} results")
                
                # If vector search found nothing, use keyword results
                if not results and keyword_chunks:
                    logger.info(f"Using {len(keyword_chunks)} results from keyword search")
                    # Convert keyword chunks to the same format as vector search results
                    for chunk in keyword_chunks[:limit]:
                        results.append((
                            LangchainDocument(  # Use LangchainDocument instead of Document
                                page_content=chunk["content"],
                                metadata={
                                    "section": chunk["metadata"]["section"],
                                    "document_id": document_id,
                                    "filename": doc_info.get('filename')
                                }
                            ),
                            0.2  # Lower score means higher similarity in this API
                        ))
            else:
                # If no document_id, just use vector search across all documents
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=limit
                )
            
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
            
            # If no results and we have a document ID, try to get at least the document info
            if not chunks and document_id:
                doc_info = await self.get_document_by_id(db, document_id)
                if doc_info:
                    logger.warning(f"No chunks found, but document exists: {doc_info.get('filename')}")
                    # Create a fallback chunk with the document title as a last resort
                    chunks.append({
                        "id": 0,
                        "content": f"Document: {doc_info.get('filename')}. Please try reformulating your query.",
                        "metadata": {
                            "section": "Document",
                            "chunk_index": 0
                        },
                        "filename": doc_info.get('filename'),
                        "document_id": document_id,
                        "similarity": 0.5
                    })
            
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
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query"""
        # Remove common words and keep only significant terms
        common_words = {"the", "a", "an", "in", "on", "at", "is", "are", "what", "who", "where", "when", "why", "how", "and", "or", "but", "of", "for", "with", "does", "do", "has", "have", "had"}
        words = query.lower().split()
        
        # Keep only significant terms (non-common words with at least 3 characters)
        keywords = [word for word in words if word not in common_words and len(word) > 2]
        
        # Remove any remaining punctuation marks from keywords
        keywords = [word.strip(".,;:?!()\"'") for word in keywords]
        
        # Remove empty strings after cleaning
        keywords = [word for word in keywords if word]
        
        # If we end up with no keywords, use the most significant words from the original query
        if not keywords and words:
            # Sort words by length (longer words tend to be more significant)
            sorted_words = sorted(words, key=len, reverse=True)
            # Take up to 3 of the longest words
            keywords = [word for word in sorted_words[:3] if len(word) > 2]
        
        logger.info(f"Extracted keywords from query: {keywords}")
        return keywords
    
    def _split_by_headers(self, content: str) -> List[tuple]:
        """Split content by headers for keyword search"""
        import re
        
        # Find all headers (# Header)
        header_pattern = r'^(#+)\s+(.+?)$'
        
        # Split content by headers
        lines = content.split('\n')
        sections = []
        current_header = "Document"
        current_content = []
        
        for line in lines:
            # Check if line is a header
            match = re.match(header_pattern, line, re.MULTILINE)
            if match:
                # Save previous section if there's content
                if current_content:
                    sections.append((current_header, '\n'.join(current_content)))
                    current_content = []
                
                # Start new section
                current_header = match.group(2)
                current_content = [line]  # Include the header in the content
            else:
                current_content.append(line)
        
        # Add the last section
        if current_content:
            sections.append((current_header, '\n'.join(current_content)))
        
        # If no headers were found, return the entire document as one section
        if len(sections) == 0:
            sections.append(("Document", content))
            
        return sections
    
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
    
    async def get_document_by_id(
        self, 
        db: AsyncSession, 
        document_id: int
    ) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        logger.info(f"Getting document by ID: {document_id}")
        
        # Use SQLAlchemy to get the document
        result = await db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalars().first()
        
        if not document:
            logger.warning(f"Document not found with ID: {document_id}")
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