"""Main FastAPI application entry point."""

# Importing required FastAPI components
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from dotenv import load_dotenv
import sys

# Initialize Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

# Import Controllers (Routers)
from app.api.chatbot_controller import router as chatbot_router
from app.core.database import init_db

def create_app() -> FastAPI:
    """Creates and configures the FastAPI application."""
    logger.debug("Creating Chatbot Backend API...")

    # Initialize FastAPI app with increased timeout for large documents
    app = FastAPI(
        title="Chatbot Q&A API",
        description="API for chatbot Q&A functionality with markdown documents",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        timeout=300
    )

    # CORS Middleware - Configure based on environment
    origins = ["*"]  # Default to all origins in development
    if os.getenv("ENVIRONMENT") == "production":
        # In production, you might want to restrict this to your frontend domain
        origins = [
            "https://your-frontend-domain.com",  # Replace with your actual frontend domain
            "https://your-app.onrender.com",     # Your Render frontend domain
            "*"  # Keep this for now, remove in strict production
        ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # Health Check Route
    @app.get("/api/health", tags=["Health Check"])
    async def health_check():
        """Checks the health of the Chatbot API."""
        logger.debug("Health check called")
        return {
            "service": "chatbot",
            "status": "healthy",
            "version": "1.0.0",
            "environment": os.getenv("ENVIRONMENT", "development"),
            "database_url": "configured" if os.getenv("DATABASE_URL") else "not configured",
            "openai_api": "configured" if os.getenv("OPENAI_API_KEY") else "not configured"
        }

    # Register Routers
    app.include_router(chatbot_router, prefix="/api", tags=["Chatbot Q&A"])

    # Log Registered Routes
    logger.debug("Chatbot API Routes:")
    for route in app.routes:
        logger.debug(f"Registered route: {route.methods} {route.path}")
        
    # Initialize database
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        if os.getenv("ENVIRONMENT") == "production":
            # In production, we want to fail fast if DB init fails
            raise
        else:
            logger.warning("Continuing despite database initialization error in development")

    return app

# Create FastAPI App Instance
app = create_app()

# For local development only
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)