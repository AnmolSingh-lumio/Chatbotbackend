import logging
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form, Response, Depends
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
from app.models.qa_request import QARequest
from app.models.qa_response import QAResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_async_db
import os
from typing import List, Optional
import aiofiles
import asyncio
import json
from datetime import datetime
import time

# Services
from app.services.chatbot_service import ChatbotService
from app.services.vector_repository import VectorRepository
from app.services.chunking_service import ChunkingService

router = APIRouter()
logger = logging.getLogger(__name__)

# Directory to store uploaded markdown files
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Create services
chatbot_service = ChatbotService()
vector_repository = VectorRepository()
chunking_service = ChunkingService()

# Check if we're in production environment
IS_PRODUCTION = os.getenv("ENVIRONMENT") == "production"

# Add an OPTIONS handler for CORS preflight requests
@router.options("/{rest_of_path:path}")
async def options_route(rest_of_path: str):
    """Handle OPTIONS requests for CORS preflight."""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        },
    )

@router.post("/qa", 
            summary="Ask questions about a markdown document", 
            response_model=QAResponse)
async def ask_question(
    request: QARequest,
    db: AsyncSession = Depends(get_async_db)
):
    """
    Endpoint to ask questions about a markdown document.
    
    This endpoint:
    1. Validates the request
    2. Processes the question using RAG with function calling
    3. Returns the answer with relevant metadata
    """
    try:
        logger.info(f"Received Q&A request with question: {request.question}")
        
        document_id = request.document_id
        filename = request.filename
        
        # If filename is provided but no document_id, fetch document_id first
        if filename and not document_id:
            logger.info(f"Looking up document ID for filename: {filename}")
            doc = await vector_repository.get_document_by_filename(db, filename)
            if doc:
                document_id = doc["id"]
                logger.info(f"Found document ID: {document_id} for filename: {filename} - WILL USE THIS DOCUMENT CONTEXT")
            else:
                logger.warning(f"Filename {filename} not found in database")
                
        # Use the chatbot service to answer the question
        response = await chatbot_service.answer_question(
            db=db, 
            question=request.question,
            filename=filename,
            document_id=document_id,
            chat_history=request.chat_history
        )
        
        if not response["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=response["message"]
            )
            
        return QAResponse(
            success=response["success"],
            message=response["message"],
            data=response["data"]
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to process Q&A request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.get("/files",
            summary="List uploaded markdown files",
            description="Get a list of all uploaded markdown files")
async def list_files(db: AsyncSession = Depends(get_async_db)):
    """
    Endpoint to list all uploaded markdown files.
    """
    try:
        # Get files from vector repository
        files = await vector_repository.list_documents(db)
        
        # Format response
        formatted_files = []
        for file in files:
            formatted_files.append({
                "filename": file["filename"],
                "description": file["description"],
                "created_at": file["created_at"],
                "updated_at": file["updated_at"]
            })
        
        return {
            "success": True,
            "message": f"Found {len(formatted_files)} markdown files",
            "data": {
                "files": formatted_files
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to list files: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list files"
        )

@router.post("/upload",
            summary="Upload a markdown document",
            description="Upload a markdown document to be used for Q&A")
async def upload_markdown(
    file: UploadFile = File(...),
    description: str = Form(None),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Endpoint to upload a markdown document.
    
    This endpoint:
    1. Validates the uploaded file is a markdown file
    2. Saves the file to the local file system
    3. Processes the file for embeddings and vector storage
    4. Returns the file information
    """
    try:
        logger.info(f"Received file upload: {file.filename}")
        
        # Validate file type
        if not file.filename.endswith(('.md', '.markdown')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only markdown files (.md, .markdown) are accepted"
            )
        
        # Create a safe filename
        safe_filename = os.path.basename(file.filename)
        file_path = os.path.join(UPLOAD_DIR, safe_filename)
        
        # Read file content
        content = await file.read()
        
        # Save file to disk
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        
        # Convert content to string
        content_str = content.decode('utf-8')
        
        # Process the document
        result = await chatbot_service.process_upload(
            db=db,
            filename=safe_filename,
            content=content_str,
            description=description
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["message"]
            )
        
        return {
            "success": True,
            "message": f"File {safe_filename} uploaded and processed successfully",
            "data": {
                "filename": safe_filename,
                "size": len(content),
                "chunks": result["data"]["chunks"],
                "document_id": result["data"]["document_id"]
            }
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to upload file: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}"
        )

@router.get("/file/{filename}",
            summary="Get content of a markdown file",
            description="Get the content of a specific markdown file")
async def get_file_content(
    filename: str,
    db: AsyncSession = Depends(get_async_db)
):
    """
    Endpoint to get the content of a markdown file.
    """
    try:
        # Get document from vector repository
        document = await vector_repository.get_document_by_filename(db, filename)
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File {filename} not found"
            )
        
        return {
            "success": True,
            "message": f"File {filename} found",
            "data": {
                "filename": document["filename"],
                "content": document["content"],
                "description": document["description"]
            }
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to get file content: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get file content: {str(e)}"
        )

@router.delete("/files/{filename}",
            summary="Delete a markdown file",
            description="Delete a specific markdown file")
async def delete_file(
    filename: str,
    db: AsyncSession = Depends(get_async_db)
):
    """
    Endpoint to delete a markdown file.
    """
    try:
        # Delete document from vector repository
        success = await vector_repository.delete_document(db, filename)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File {filename} not found"
            )
        
        # Also delete the file from the file system
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return {
            "success": True,
            "message": f"File {filename} deleted successfully",
            "data": None
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to delete file: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete file: {str(e)}"
        )

@router.get("/status", 
           summary="API status endpoint", 
           description="Check status of API including database initialization")
async def api_status(db: AsyncSession = Depends(get_async_db)):
    """
    Endpoint to check API status.
    """
    try:
        # Get uptime
        uptime = time.time() - chatbot_service.initialization_time
        
        # Get initialization status
        initialized = chatbot_service.is_initialized
        
        # Calculate estimated wait time
        estimated_wait_time = max(0, 60 - uptime) if not initialized else 0
        
        # Check if database is available by listing documents
        documents = await vector_repository.list_documents(db)
        
        return {
            "success": True,
            "message": "API status",
            "data": {
                "status": "healthy",
                "ready_for_queries": initialized or uptime > 60,
                "uptime": f"{uptime:.1f}s",
                "initialized": initialized,
                "documents_count": len(documents),
                "estimated_wait_time": int(estimated_wait_time)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get API status: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": f"API status check failed: {str(e)}",
            "data": {
                "status": "unhealthy",
                "error": str(e)
            }
        }

@router.get("/debug", 
           summary="Debug endpoint to check configuration", 
           description="Displays information about the current API configuration")
async def debug_info():
    """
    Debug endpoint to check environment variables and API key configuration.
    """
    try:
        # Get the API key securely (mask it for display)
        api_key = os.getenv("GEMINI_API_KEY", "")
        masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "Not set"
        
        # Check if Gemini API key is configured
        client_configured = bool(api_key)
        
        # Get environment variable debug info
        env_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
        env_file_exists = os.path.exists(env_file_path)
        
        # Try to get the model info to verify API key works
        api_working = False
        error_message = None
        try:
            # Simple test call to check if API is working
            genai.configure(api_key=api_key)
            models = genai.list_models()
            api_working = len(models) > 0
        except Exception as e:
            error_message = str(e)
        
        # Check uploads directory
        uploads_dir_exists = os.path.exists(UPLOAD_DIR)
        uploads_dir_writable = os.access(UPLOAD_DIR, os.W_OK) if uploads_dir_exists else False
        
        return {
            "success": True,
            "environment": os.getenv("ENVIRONMENT", "not set"),
            "is_production": IS_PRODUCTION,
            "api_key_exists": bool(api_key),
            "api_key_masked": masked_key,
            "api_key_length": len(api_key),
            "gemini_client_configured": client_configured,
            "gemini_api_working": api_working,
            "api_error_message": error_message,
            "uploads_dir": {
                "path": UPLOAD_DIR,
                "exists": uploads_dir_exists,
                "writable": uploads_dir_writable
            },
            "env_file_exists": env_file_exists,
            "env_file_path": env_file_path,
            "cwd": os.getcwd()
        }
    except Exception as e:
        logger.error(f"Debug endpoint error: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        } 