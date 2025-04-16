from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Any

class QAResponse(BaseModel):
    """Response model for document Q&A"""
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="A message describing the result")
    data: Optional[Dict[str, Any]] = Field(None, description="The response data containing the answer and metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Question answered successfully",
                "data": {
                    "answer": "The document lists three features: Feature 1, Feature 2, and Feature 3.",
                    "confidence": 0.95,
                    "source_documents": [],
                    "matched_template": {
                        "id": 2,
                        "question": "What are the key points of this document?",
                        "similarity": 0.87,
                        "category": "General"
                    }
                }
            }
        } 