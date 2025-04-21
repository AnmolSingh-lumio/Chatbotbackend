"""Main FastAPI application entry point."""

# Importing required FastAPI components
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from dotenv import load_dotenv
import asyncio
from contextlib import asynccontextmanager

# Initialize Logger
logger = logging.getLogger(__name__)

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

# Import Controllers (Routers) and services
from app.api.chatbot_controller import router as chatbot_router
from app.core.database import init_db
from app.services.chatbot_service import ChatbotService

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI application"""
    # Initialize database
    logger.info("Initializing database...")
    try:
        init_db()  # Remove await since init_db is not async
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}", exc_info=True)
    
    # Initialize chatbot service
    try:
        # Warm up the chatbot service
        chatbot_service = ChatbotService()
        await chatbot_service.initialize()
        logger.info("ChatbotService initialized successfully")
        
        # Re-index all existing documents to apply improved chunking and metadata
        await reindex_documents()
    except Exception as e:
        logger.error(f"Error during application startup: {str(e)}", exc_info=True)
    
    # Yield control back to FastAPI
    yield
    
    # Cleanup when application shuts down
    logger.info("Application shutting down")

async def reindex_documents():
    """Reindex all documents to apply new chunking and metadata extraction."""
    from sqlalchemy.ext.asyncio import AsyncSession
    from app.core.database import get_async_db, Document
    from sqlalchemy import select
    from app.services.vector_repository import VectorRepository
    
    logger.info("Starting document reindexing...")
    vector_repository = VectorRepository()
    
    try:
        async for db in get_async_db():
            try:
                # Get all documents with their data explicitly loaded
                query = select(
                    Document.id,
                    Document.filename, 
                    Document.content,
                    Document.description
                )
                result = await db.execute(query)
                documents_data = result.all()
                
                total_docs = len(documents_data)
                logger.info(f"Found {total_docs} documents for reindexing")
                
                # Process each document using the explicitly loaded data
                for i, (doc_id, filename, content, description) in enumerate(documents_data, 1):
                    logger.info(f"Reindexing document {i}/{total_docs}: {filename}")
                    
                    # Process the document with improved chunking
                    try:
                        chatbot_service = ChatbotService()
                        await chatbot_service.process_upload(
                            db=db,
                            filename=filename,
                            content=content,
                            description=description,
                            reindex=True  # Flag to indicate this is a reindex operation
                        )
                        logger.info(f"Successfully reindexed document: {filename}")
                    except Exception as doc_error:
                        logger.error(f"Error reindexing document {filename}: {str(doc_error)}")
                
                logger.info("Document reindexing complete")
                break  # We only need to process one session
            except Exception as e:
                logger.error(f"Error during document reindexing: {str(e)}", exc_info=True)
    except Exception as e:
        logger.error(f"Failed to get database session: {str(e)}", exc_info=True)

def create_app() -> FastAPI:
    """Create and configure FastAPI app instance."""
    logger.info("Creating FastAPI application instance")
    
    # Create the upload directory if it doesn't exist
    upload_dir = os.path.join(os.getcwd(), "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    # Create FastAPI App Instance
    app = FastAPI(
        title="Contract Chatbot API",
        description="API for querying contract documents using advanced RAG techniques",
        version="1.0.0",
        openapi_url="/api/openapi.json",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        lifespan=lifespan
    )
    
    # Register Routers
    app.include_router(chatbot_router, prefix="/api", tags=["chatbot"])
    
    # Add simple health check endpoint
    @app.get("/health", tags=["health"])
    async def health_check():
        return {"status": "healthy", "version": "1.0.0"}
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app

# Create FastAPI App Instance
app = create_app()

# For local development only
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)