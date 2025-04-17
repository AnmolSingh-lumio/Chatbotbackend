import logging
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Response
from app.models.qa_request import QARequest
from app.models.qa_response import QAResponse
from app.services.chatbot_service import ChatbotService
import os
from typing import List, Optional
import aiofiles
import google.generativeai as genai
import asyncio
import json

router = APIRouter()
logger = logging.getLogger(__name__)

# Directory to store uploaded markdown files
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

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
    request: QARequest
):
    """
    Endpoint to ask questions about a markdown document.
    
    This endpoint:
    1. Validates the request
    2. Processes the question using RAG (with optional embeddings)
    3. Returns the answer with relevant metadata
    """
    try:
        logger.info(f"Received Q&A request with question: {request.question}")
        
        # Use the chatbot service to answer the question
        response = await ChatbotService.answer_question(request)
        
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

@router.get("/files",
            summary="List uploaded markdown files",
            description="Get a list of all uploaded markdown files")
async def list_files():
    """
    Endpoint to list all uploaded markdown files.
    """
    try:
        # Use local file system storage for both development and production
        files = []
        if os.path.exists(UPLOAD_DIR):
            for filename in os.listdir(UPLOAD_DIR):
                if filename.endswith(('.md', '.markdown')):
                    file_path = os.path.join(UPLOAD_DIR, filename)
                    files.append({
                        "filename": filename,
                        "size": os.path.getsize(file_path),
                        "last_modified": os.path.getmtime(file_path),
                        "path": file_path
                    })
        
        return {
            "success": True,
            "message": f"Found {len(files)} markdown files",
            "data": {
                "files": files
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
    description: str = Form(None)
):
    """
    Endpoint to upload a markdown document.
    
    This endpoint:
    1. Validates the uploaded file is a markdown file
    2. Saves the file to the local file system
    3. Returns the file information
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
        
        # Read file content
        content = await file.read()
        
        # Save to file system (both in development and production)
        file_path = os.path.join(UPLOAD_DIR, safe_filename)
        
        # Save the file locally
        async with aiofiles.open(file_path, 'wb') as out_file:
            await out_file.write(content)
        
        logger.info(f"Successfully saved file to path: {file_path}")
        
        # Add a 2-second artificial delay to prevent immediate embedding generation 
        # that might trigger rate limits when the frontend makes immediate requests
        await asyncio.sleep(2)
        
        return {
            "success": True,
            "message": "File uploaded successfully",
            "data": {
                "filename": safe_filename,
                "description": description,
                "size": len(content),
                "path": file_path,
                "processing": True  # Signal to the frontend that processing is happening
            }
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to upload file: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload file"
        )

@router.get("/file/{filename}",
            summary="Get content of a markdown file",
            description="Get the content of a specific markdown file")
async def get_file_content(filename: str):
    """
    Endpoint to get the content of a markdown file.
    """
    try:
        # Use local file system for both development and production
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
            content = await file.read()
        
        return {
            "success": True,
            "message": "File content retrieved successfully",
            "data": {
                "filename": filename,
                "content": content,
                "path": file_path
            }
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to get file content: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get file content"
        )

@router.delete("/files/{filename}",
            summary="Delete a markdown file",
            description="Delete a specific markdown file")
async def delete_file(filename: str):
    """
    Endpoint to delete a markdown file.
    """
    try:
        logger.info(f"Attempting to delete file: {filename}")
        
        # Use local file system for both development and production
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        if not os.path.exists(file_path):
            logger.warning(f"File not found for deletion: {file_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        # Delete the file
        os.remove(file_path)
        logger.info(f"Successfully deleted file: {file_path}")
        
        return {
            "success": True,
            "message": f"File {filename} deleted successfully",
            "data": {
                "filename": filename
            }
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to delete file: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete file"
        )

@router.get("/status", 
           summary="API status endpoint", 
           description="Check status of API including embedding initialization")
async def api_status():
    """
    Status endpoint to check if the API is ready for use.
    """
    from app.services.chatbot_service import is_embedding_initialized, is_template_initialization_in_progress, initialization_start_time, MAX_INITIALIZATION_TIME
    
    try:
        # Calculate how long initialization has been running
        initialization_time_elapsed = 0
        initialization_timeout = False
        
        if initialization_start_time > 0:
            import time
            current_time = time.time()
            initialization_time_elapsed = int(current_time - initialization_start_time)
            
            # Check if initialization has timed out
            if initialization_time_elapsed > MAX_INITIALIZATION_TIME:
                initialization_timeout = True
        
        # Get embedding initialization status
        embedding_status = {
            "initialized": is_embedding_initialized,
            "initialization_in_progress": is_template_initialization_in_progress,
            "initialization_time_elapsed": initialization_time_elapsed,
            "initialization_timeout": initialization_timeout,
            "max_initialization_time": MAX_INITIALIZATION_TIME
        }
        
        # Check if template embedding cache exists
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "cache")
        template_cache_path = os.path.join(cache_dir, "template_embeddings.json")
        cache_exists = os.path.exists(template_cache_path)
        
        if cache_exists:
            try:
                with open(template_cache_path, 'r') as f:
                    cached_data = json.load(f)
                cache_info = {
                    "exists": True,
                    "template_count": len(cached_data),
                    "last_modified": os.path.getmtime(template_cache_path)
                }
            except Exception as e:
                cache_info = {
                    "exists": True,
                    "error": str(e),
                    "last_modified": os.path.getmtime(template_cache_path)
                }
        else:
            cache_info = {
                "exists": False
            }
        
        # System is ready if embeddings are initialized OR initialization timeout occurred
        system_ready = is_embedding_initialized or initialization_timeout
        
        return {
            "success": True,
            "message": "API status check completed",
            "data": {
                "api_online": True,
                "embedding_service": embedding_status,
                "template_cache": cache_info,
                "ready_for_queries": system_ready,
                "estimated_wait_time": 0 if system_ready else min(30, MAX_INITIALIZATION_TIME - initialization_time_elapsed)
            }
        }
    except Exception as e:
        logger.error(f"Status endpoint error: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        } 