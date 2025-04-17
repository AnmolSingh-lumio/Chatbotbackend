import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# In-memory storage for question templates
class TemplateStore:
    # Initialize with contract-specific templates
    templates = [
        {
            "id": 1,
            "question_text": "What is the contract start and end date?",
            "category": "General Contract Terms",
            "embedding": None,  # Will be populated by the embedding service
            "answer_strategy": """
To answer questions about contract start and end dates:
1. Look for sections labeled "Term", "Duration", "Contract Period", or "Effective Date"
2. Identify specific dates mentioned for contract commencement
3. Find specific dates mentioned for contract termination or expiration
4. Note any statements about the contract length (e.g., "shall remain in effect for 24 months")
5. If exact dates aren't specified, look for relative terms (e.g., "effective upon signing" or "expires 12 months after effective date")
"""
        },
        {
            "id": 2,
            "question_text": "Is there an automatic renewal clause in the contract?",
            "category": "General Contract Terms",
            "embedding": None,
            "answer_strategy": """
To answer questions about automatic renewal clauses:
1. Look for sections labeled "Renewal", "Extension", "Term", or "Termination"
2. Search for phrases like "automatically renew", "auto-renewal", or "evergreen clause"
3. Identify the conditions for automatic renewal (if any)
4. Note the length of the renewal periods (e.g., "renews for successive one-year terms")
5. Look for any notification requirements to prevent automatic renewal
"""
        },
        {
            "id": 3,
            "question_text": "What is the termination notice period required?",
            "category": "General Contract Terms",
            "embedding": None,
            "answer_strategy": """
To answer questions about termination notice periods:
1. Look for sections labeled "Termination", "Cancellation", or "Notice Requirements"
2. Identify specific timeframes for providing notice (e.g., "30 days written notice")
3. Note any differences in notice periods for different termination scenarios
4. Check if notice must be delivered in a specific format (e.g., certified mail, email)
5. Look for any special conditions related to notice periods at different contract stages
"""
        },
        {
            "id": 4,
            "question_text": "Are there any penalties for early termination?",
            "category": "General Contract Terms",
            "embedding": None,
            "answer_strategy": """
To answer questions about early termination penalties:
1. Look for sections labeled "Early Termination", "Termination Fees", or "Liquidated Damages"
2. Identify any financial penalties mentioned for ending the contract before its term
3. Note if penalties vary based on when termination occurs during the contract period
4. Check if there are exceptions where early termination is allowed without penalty
5. Look for formulas used to calculate termination fees (e.g., percentage of remaining contract value)
"""
        },
        {
            "id": 5,
            "question_text": "Which accounts are eligible for contract benefits?",
            "category": "General Contract Terms",
            "embedding": None,
            "answer_strategy": """
To answer questions about account eligibility for contract benefits:
1. Look for sections about "Eligibility", "Scope", "Coverage", or "Participating Accounts"
2. Identify any account qualifications or requirements mentioned
3. Note any excluded accounts or services specifically mentioned
4. Check for language about affiliates, subsidiaries, or related entities
5. Look for any volume or spending thresholds that determine eligibility
"""
        },
        {
            "id": 6,
            "question_text": "Who are the parties involved in the contract?",
            "category": "General Contract Terms",
            "embedding": None,
            "answer_strategy": """
To answer questions about parties involved in the contract:
1. Look at the beginning of the contract for a section defining the parties
2. Check for terms like "Party A", "Party B", "Client", "Provider", "Vendor", "Customer", etc.
3. Note the full legal names of all entities mentioned as contracting parties
4. Look for definitions of parties' affiliates or subsidiaries that may be covered
5. Identify any third parties referenced that have rights or obligations under the contract
"""
        },
        {
            "id": 7,
            "question_text": "Are there any exclusivity clauses restricting me from using other carriers?",
            "category": "General Contract Terms",
            "embedding": None,
            "answer_strategy": """
To answer questions about exclusivity clauses:
1. Look for sections labeled "Exclusivity", "Non-compete", or "Preferred Provider"
2. Search for terms like "exclusive", "sole provider", or "minimum commitment"
3. Identify any restrictions on working with competitors
4. Note any volume or percentage commitments required
5. Look for exceptions to exclusivity requirements or carve-outs for specific situations
"""
        },
        {
            "id": 8,
            "question_text": "Does the contract specify a governing law and jurisdiction for disputes?",
            "category": "General Contract Terms",
            "embedding": None,
            "answer_strategy": """
To answer questions about governing law and jurisdiction:
1. Look for sections labeled "Governing Law", "Jurisdiction", "Venue", or "Dispute Resolution"
2. Identify the specific state, province, or country whose laws govern the contract
3. Note any specified courts or venues where disputes must be filed
4. Check for alternative dispute resolution mechanisms (arbitration, mediation)
5. Look for choice of law provisions that may apply different laws to different aspects of the contract
"""
        },
        {
            "id": 9,
            "question_text": "What are the key performance indicators (KPIs) outlined in the contract?",
            "category": "General Contract Terms",
            "embedding": None,
            "answer_strategy": """
To answer questions about key performance indicators:
1. Look for sections labeled "Performance Metrics", "SLAs", "KPIs", or "Service Levels"
2. Identify specific measurable targets or thresholds mentioned
3. Note any penalties or rewards tied to performance levels
4. Check for reporting requirements related to performance monitoring
5. Look for provisions about remediation processes if KPIs aren't met
"""
        },
        {
            "id": 10,
            "question_text": "Are there any blackout dates or service restrictions during holidays?",
            "category": "General Contract Terms",
            "embedding": None,
            "answer_strategy": """
To answer questions about blackout dates or holiday restrictions:
1. Look for sections about "Service Availability", "Blackout Periods", or "Holiday Schedule"
2. Identify any specific dates or time periods when services may be limited
3. Note any seasonal restrictions or capacity limitations mentioned
4. Check for modified service levels during specific periods
5. Look for any provisions about advance notice for service interruptions
"""
        },
        {
            "id": 11,
            "question_text": "What is the notification period for rate changes?",
            "category": "General Contract Terms",
            "embedding": None,
            "answer_strategy": """
To answer questions about notification for rate changes:
1. Look for sections about "Pricing", "Rate Changes", "Price Adjustments", or "Fees"
2. Identify any specific timeframes for providing notice of rate changes
3. Note any limits on frequency or percentage of rate increases
4. Check for client rights upon rate changes (e.g., right to terminate)
5. Look for any exceptions where rates can change without notice
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