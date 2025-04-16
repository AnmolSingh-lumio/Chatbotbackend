import logging
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Response
from app.models.qa_request import QARequest
from app.models.qa_response import QAResponse
from app.services.chatbot_service import ChatbotService
from app.services.storage_service import StorageService
import os
from typing import List, Optional
import aiofiles
import google.generativeai as genai

router = APIRouter()
logger = logging.getLogger(__name__)

# Directory to store uploaded markdown files (used for local development only)
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
        
        # Check Supabase configuration
        supabase_url = os.getenv("SUPABASE_URL", "")
        supabase_key = os.getenv("SUPABASE_KEY", "")
        supabase_configured = bool(supabase_url) and bool(supabase_key)
        masked_supabase_key = f"{supabase_key[:4]}...{supabase_key[-4:]}" if len(supabase_key) > 8 else "Not set"
        
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
            "supabase_configured": supabase_configured,
            "supabase_url": supabase_url,
            "supabase_key_masked": masked_supabase_key,
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
    2. Saves the file to the storage (local or Supabase)
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
        
        # Store the file based on environment
        if IS_PRODUCTION:
            # Use Supabase in production
            try:
                file_info = await StorageService.upload_file(content, safe_filename)
                logger.info(f"Successfully uploaded file to Supabase: {file_info}")
                
                return {
                    "success": True,
                    "message": "File uploaded successfully to cloud storage",
                    "data": {
                        "filename": safe_filename,
                        "description": description,
                        "size": file_info["size"],
                        "url": file_info["url"]
                    }
                }
            except Exception as e:
                logger.error(f"Failed to upload to Supabase: {str(e)}")
                
                # Fallback to memory-only operation - let the frontend know this won't persist
                return {
                    "success": True,
                    "message": "File processed but not stored (storage error)",
                    "data": {
                        "filename": safe_filename,
                        "description": description,
                        "size": len(content),
                        "content": content.decode('utf-8'),
                        "temporary": True,
                        "error": str(e)
                    }
                }
        else:
            # Use local storage in development
            file_path = os.path.join(UPLOAD_DIR, safe_filename)
            
            # Save the file locally
            async with aiofiles.open(file_path, 'wb') as out_file:
                await out_file.write(content)
            
            logger.info(f"Successfully saved file to local path: {file_path}")
            
            return {
                "success": True,
                "message": "File uploaded successfully to local storage",
                "data": {
                    "filename": safe_filename,
                    "description": description,
                    "size": len(content),
                    "path": file_path
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

@router.get("/files",
            summary="List uploaded markdown files",
            description="Get a list of all uploaded markdown files")
async def list_files():
    """
    Endpoint to list all uploaded markdown files.
    """
    try:
        if IS_PRODUCTION:
            # Use Supabase in production
            try:
                files = await StorageService.list_files()
                return {
                    "success": True,
                    "message": f"Found {len(files)} markdown files",
                    "data": {
                        "files": files
                    }
                }
            except Exception as e:
                logger.error(f"Failed to list files from Supabase: {str(e)}")
                # Fallback to empty list if Supabase fails
                return {
                    "success": True,
                    "message": "No files available (storage error)",
                    "data": {
                        "files": [],
                        "error": str(e)
                    }
                }
        else:
            # Use local storage in development
            files = []
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
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to list files: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list files"
        )

@router.get("/file/{filename}",
            summary="Get content of a markdown file",
            description="Get the content of a specific markdown file")
async def get_file_content(filename: str):
    """
    Endpoint to get the content of a markdown file.
    """
    try:
        if IS_PRODUCTION:
            # Use Supabase in production
            try:
                content, file_url = await StorageService.get_file_content(filename)
                return {
                    "success": True,
                    "message": "File content retrieved successfully",
                    "data": {
                        "filename": filename,
                        "content": content.decode('utf-8'),
                        "url": file_url
                    }
                }
            except Exception as e:
                logger.error(f"Failed to get file from Supabase: {str(e)}")
                # Return a friendly error
                return {
                    "success": False,
                    "message": "File not found or could not be retrieved",
                    "error": str(e)
                }
        else:
            # Use local storage in development
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