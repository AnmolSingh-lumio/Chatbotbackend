import os
import logging
from supabase import create_client, Client
from dotenv import load_dotenv
import io
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET_NAME = "markdown-documents"  # The bucket name to use in Supabase

class StorageService:
    """Service for handling file storage using Supabase."""
    
    _supabase: Optional[Client] = None
    
    @classmethod
    def get_client(cls) -> Client:
        """Get or initialize the Supabase client."""
        if cls._supabase is None:
            if not SUPABASE_URL or not SUPABASE_KEY:
                logger.error("Supabase credentials not found. Please set SUPABASE_URL and SUPABASE_KEY environment variables.")
                raise ValueError("Supabase credentials not found")
            
            logger.info(f"Initializing Supabase client with URL: {SUPABASE_URL}")
            cls._supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            
            # Ensure the bucket exists
            try:
                cls._supabase.storage.get_bucket(BUCKET_NAME)
                logger.info(f"Bucket '{BUCKET_NAME}' already exists")
            except Exception as e:
                if "Bucket not found" in str(e):
                    logger.info(f"Creating bucket '{BUCKET_NAME}'")
                    cls._supabase.storage.create_bucket(BUCKET_NAME)
                else:
                    logger.error(f"Error checking/creating bucket: {str(e)}")
                    raise
        
        return cls._supabase
    
    @classmethod
    async def upload_file(cls, file_content: bytes, filename: str) -> Dict:
        """
        Upload a file to Supabase Storage.
        
        Args:
            file_content: The file content as bytes
            filename: The name of the file
            
        Returns:
            Dict with file information
        """
        try:
            client = cls.get_client()
            
            # Upload the file
            path = f"{filename}"
            response = client.storage.from_(BUCKET_NAME).upload(
                path,
                file_content,
                {"content-type": "text/markdown"}
            )
            
            # Get the public URL
            file_url = client.storage.from_(BUCKET_NAME).get_public_url(path)
            
            logger.info(f"Successfully uploaded file {filename} to Supabase Storage")
            
            return {
                "filename": filename,
                "size": len(file_content),
                "url": file_url
            }
            
        except Exception as e:
            logger.error(f"Failed to upload file to Supabase: {str(e)}")
            raise
    
    @classmethod
    async def list_files(cls) -> List[Dict]:
        """
        List all files in the storage bucket.
        
        Returns:
            List of file information dictionaries
        """
        try:
            client = cls.get_client()
            
            # List files in the bucket
            response = client.storage.from_(BUCKET_NAME).list()
            
            files = []
            for item in response:
                if item["name"].endswith(('.md', '.markdown')):
                    file_url = client.storage.from_(BUCKET_NAME).get_public_url(item["name"])
                    files.append({
                        "filename": item["name"],
                        "size": item["metadata"]["size"],
                        "last_modified": item["metadata"]["lastModified"],
                        "url": file_url
                    })
            
            logger.info(f"Listed {len(files)} markdown files from Supabase Storage")
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files from Supabase: {str(e)}")
            raise
    
    @classmethod
    async def get_file_content(cls, filename: str) -> Tuple[bytes, str]:
        """
        Get the content of a file from storage.
        
        Args:
            filename: The name of the file
            
        Returns:
            Tuple of (file_content_bytes, file_url)
        """
        try:
            client = cls.get_client()
            
            # Download the file
            response = client.storage.from_(BUCKET_NAME).download(filename)
            
            # Get the public URL
            file_url = client.storage.from_(BUCKET_NAME).get_public_url(filename)
            
            logger.info(f"Successfully downloaded file {filename} from Supabase Storage")
            
            return response, file_url
            
        except Exception as e:
            logger.error(f"Failed to get file from Supabase: {str(e)}")
            raise
    
    @classmethod
    async def delete_file(cls, filename: str) -> bool:
        """
        Delete a file from storage.
        
        Args:
            filename: The name of the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            client = cls.get_client()
            
            # Delete the file
            client.storage.from_(BUCKET_NAME).remove([filename])
            
            logger.info(f"Successfully deleted file {filename} from Supabase Storage")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete file from Supabase: {str(e)}")
            raise 