from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class QAResponseData(BaseModel):
    """Response data model for document Q&A"""
    answer: str = Field(..., description="The answer to the question")
    processing_time: Optional[float] = Field(None, description="Time taken to process the request in seconds")
    contexts_used: Optional[int] = Field(None, description="Number of document contexts used to generate the answer")
    document_id: Optional[int] = Field(None, description="The ID of the document queried")
    filename: Optional[str] = Field(None, description="The filename of the document queried")

class QAResponse(BaseModel):
    """Response model for document Q&A"""
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="A message describing the result")
    data: Optional[QAResponseData] = Field(None, description="The response data")
    
    class Config:
        json_schema_extra = {  # Changed from schema_extra to json_schema_extra
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