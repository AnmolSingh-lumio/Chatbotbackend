from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from dotenv import load_dotenv

# Initialize Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Load environment variables
load_dotenv()

# Import Controllers (Routers)
from app.api.chatbot_controller import router as chatbot_router

def create_app() -> FastAPI:
    """Creates and configures the FastAPI application."""
    logger.debug("Creating Chatbot Backend API...")

    app = FastAPI(
        title="Chatbot Q&A API",
        description="API for chatbot Q&A functionality with markdown documents",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    # CORS Middleware - Allow all origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins
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
            "cors": {
                "allowed_origins": ["*"]
            }
        }

    # Register Routers
    app.include_router(chatbot_router, prefix="/api", tags=["Chatbot Q&A"])

    # Log Registered Routes
    logger.debug("Chatbot API Routes:")
    for route in app.routes:
        logger.debug(f"Registered route: {route.methods} {route.path}")

    return app

# Create FastAPI App Instance
app = create_app()

# For local development only
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Use PORT env var if provided by Render
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)