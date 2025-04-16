from pydantic import BaseModel, Field

class QARequest(BaseModel):
    """Request model for document Q&A"""
    markdown_content: str = Field(..., description="The markdown content to query")
    question: str = Field(..., min_length=1, max_length=1000, description="The question to ask about the document")
    
    class Config:
        schema_extra = {
            "example": {
                "markdown_content": "# Sample Document\n\nThis is a sample markdown document.\n\n## Features\n\n- Feature 1\n- Feature 2\n- Feature 3",
                "question": "What features are listed in the document?"
            }
        } 