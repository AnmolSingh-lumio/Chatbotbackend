import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# In-memory storage for question templates
class TemplateStore:
    # Initialize with pre-defined templates
    templates = [
        {
            "id": 1,
            "question_text": "What is the document about?",
            "category": "General",
            "embedding": None,  # Will be populated by the embedding service
            "answer_strategy": """
To answer questions about what a document is about:
1. Look for the title, headings, or introduction that summarize the document's purpose
2. Identify the main topics or themes covered
3. Note any explicit statements about the document's purpose or scope
4. Summarize in 1-3 sentences what the document is primarily focused on
5. Mention the type of document (guide, specification, report, etc.)
"""
        },
        {
            "id": 2,
            "question_text": "What are the key points of this document?",
            "category": "General",
            "embedding": None,
            "answer_strategy": """
To answer questions about key points in a document:
1. Identify the main sections or headings in the document
2. Look for executive summaries, conclusions, or bullet points highlighting important information
3. Note any recurring themes or emphasized information
4. Identify points marked as important, critical, or essential
5. List 3-5 of the most important takeaways from the document
"""
        },
        {
            "id": 3,
            "question_text": "How do I use this feature?",
            "category": "Usage",
            "embedding": None,
            "answer_strategy": """
To answer questions about how to use a feature:
1. Look for sections labeled "How to", "Usage", "Instructions", or "Getting Started"
2. Find step-by-step instructions or numbered lists describing the process
3. Note any prerequisites or requirements before using the feature
4. Identify any screenshots, diagrams, or examples showing feature usage
5. Look for tips, best practices, or common pitfalls related to the feature
"""
        },
        {
            "id": 4,
            "question_text": "What are the requirements for this?",
            "category": "Requirements",
            "embedding": None,
            "answer_strategy": """
To answer questions about requirements:
1. Look for sections labeled "Requirements", "Prerequisites", "Dependencies", or "System Requirements"
2. Identify any hardware specifications mentioned
3. Note software dependencies and version requirements
4. Check for account permissions, API keys, or authentication requirements
5. Look for environmental requirements (OS, browser compatibility, etc.)
"""
        },
        {
            "id": 5,
            "question_text": "How do I troubleshoot issues with this?",
            "category": "Troubleshooting",
            "embedding": None,
            "answer_strategy": """
To answer questions about troubleshooting:
1. Look for sections labeled "Troubleshooting", "Common Issues", "FAQs", or "Known Issues"
2. Identify specific error messages or symptoms mentioned
3. Find step-by-step debugging processes
4. Note any recommended tools or logs to check
5. Look for contact information for support or further assistance
"""
        },
        {
            "id": 6,
            "question_text": "What are the best practices for this?",
            "category": "Best Practices",
            "embedding": None,
            "answer_strategy": """
To answer questions about best practices:
1. Look for sections labeled "Best Practices", "Recommendations", or "Guidelines"
2. Note any warnings, cautions, or important notices
3. Identify optimization tips or performance suggestions
4. Find security recommendations or data handling practices
5. Look for patterns that are explicitly encouraged or discouraged
"""
        },
        {
            "id": 7,
            "question_text": "What versions are supported?",
            "category": "Compatibility",
            "embedding": None,
            "answer_strategy": """
To answer questions about supported versions:
1. Look for sections about "Compatibility", "Supported Versions", or "System Requirements"
2. Identify specific version numbers mentioned
3. Note any deprecated or upcoming version information
4. Check for backward compatibility statements
5. Find any version-specific features or limitations
"""
        },
        {
            "id": 8,
            "question_text": "How do I install this?",
            "category": "Installation",
            "embedding": None,
            "answer_strategy": """
To answer questions about installation:
1. Look for sections labeled "Installation", "Setup", or "Getting Started"
2. Find step-by-step installation instructions
3. Note any installation prerequisites or requirements
4. Identify different installation methods if multiple are offered
5. Look for post-installation verification steps or troubleshooting
"""
        },
        {
            "id": 9,
            "question_text": "What are the limitations of this?",
            "category": "Limitations",
            "embedding": None,
            "answer_strategy": """
To answer questions about limitations:
1. Look for sections about "Limitations", "Constraints", "Known Issues", or "Restrictions"
2. Note any explicit statements about what the system cannot do
3. Identify performance limitations, quotas, or thresholds
4. Check for compatibility restrictions or unsupported features
5. Find any disclaimers or warnings about edge cases
"""
        },
        {
            "id": 10,
            "question_text": "How does this compare to other solutions?",
            "category": "Comparison",
            "embedding": None,
            "answer_strategy": """
To answer questions about comparisons:
1. Look for sections labeled "Comparison", "Alternatives", or "vs. Competitors"
2. Find tables or charts comparing features across solutions
3. Note advantages and disadvantages mentioned
4. Identify unique selling points or differentiators
5. Check for use case recommendations that suggest when to use this solution vs others
"""
        }
    ]
    
    @classmethod
    def generate_embeddings(cls, embedding_service):
        """Generate embeddings for all templates."""
        logger.info("Generating embeddings for question templates")
        
        for template in cls.templates:
            if template["embedding"] is None:
                # Use SEMANTIC_SIMILARITY for better matching of similar questions
                embedding = embedding_service.generate_embedding(
                    template["question_text"],
                    task_type="SEMANTIC_SIMILARITY"
                )
                template["embedding"] = embedding
        
        logger.info(f"Generated embeddings for {len(cls.templates)} templates")
    
    @classmethod
    def get_templates_with_embeddings(cls) -> List[Dict[str, Any]]:
        """
        Get all templates with embeddings.
        
        Returns:
            List of templates with embeddings
        """
        result = []
        
        for template in cls.templates:
            # Skip templates without embeddings
            if template["embedding"] is None:
                continue
                
            # Include template in results
            result.append(template)
            
        return result 