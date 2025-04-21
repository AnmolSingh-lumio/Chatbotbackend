from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ChatMessage(BaseModel):
    role: str
    content: str

class QARequest(BaseModel):
    """Request model for document Q&A"""
    question: str = Field(..., description="The question to ask about the document")
    filename: Optional[str] = Field(None, description="The filename of the document to query")
    document_id: Optional[int] = Field(None, description="The ID of the document to query")
    chat_history: Optional[List[ChatMessage]] = Field(None, description="Previous chat messages for context")
    options: Optional[Dict[str, Any]] = Field(None, description="Additional options for the chatbot")
    
    class Config:
        json_schema_extra = {  # Changed from schema_extra to json_schema_extra
            "example": {
                "markdown_content": "# Sample Document\n\nThis is a sample markdown document.\n\n## Features\n\n- Feature 1\n- Feature 2\n- Feature 3",
                "question": "What features are listed in the document?"
            }
        }