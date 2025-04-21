"""
Service for chunking markdown documents.
"""
import os
import re
import logging
from typing import List, Dict, Any
from math import ceil

logger = logging.getLogger(__name__)

class ChunkingService:
    """Service for chunking documents into manageable pieces."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize the chunking service.
        
        Args:
            chunk_size: Size of chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "5000"))
        self.chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "100"))
        logger.info(f"Initialized chunking service with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")
    
    def chunk_markdown(self, content: str) -> List[Dict[str, Any]]:
        """
        Split markdown content into chunks with metadata.
        
        Args:
            content: The markdown content to chunk
            
        Returns:
            List of dictionaries with chunk content and metadata
        """
        # First, try to split by sections (using markdown headers)
        sections = self._split_by_headers(content)
        chunks = []
        
        # Process each section
        for section_title, section_content in sections:
            # If section is smaller than chunk size, use it as is
            if len(section_content) <= self.chunk_size:
                chunks.append({
                    "content": section_content,
                    "metadata": {
                        "section": section_title,
                        "start_char": 0,
                        "end_char": len(section_content)
                    }
                })
            else:
                # If section is larger than chunk size, split by paragraphs
                section_chunks = self._chunk_text(section_content)
                for i, chunk in enumerate(section_chunks):
                    chunks.append({
                        "content": chunk,
                        "metadata": {
                            "section": section_title,
                            "chunk_index": i,
                            "total_chunks": len(section_chunks)
                        }
                    })
        
        logger.info(f"Split document into {len(chunks)} chunks")
        return chunks
    
    def _split_by_headers(self, content: str) -> List[tuple]:
        """
        Split markdown content by headers.
        
        Args:
            content: The markdown content
            
        Returns:
            List of (header, content) tuples
        """
        # Find all headers (# Header)
        header_pattern = r'^(#+)\s+(.+?)$'
        
        # Split content by headers
        lines = content.split('\n')
        sections = []
        current_header = "Document"
        current_content = []
        
        for line in lines:
            # Check if line is a header
            match = re.match(header_pattern, line, re.MULTILINE)
            if match:
                # Save previous section if there's content
                if current_content:
                    sections.append((current_header, '\n'.join(current_content)))
                    current_content = []
                
                # Start new section
                current_header = match.group(2)
                current_content = [line]  # Include the header in the content
            else:
                current_content.append(line)
        
        # Add the last section
        if current_content:
            sections.append((current_header, '\n'.join(current_content)))
        
        # If no headers were found, return the entire document as one section
        if len(sections) == 0:
            sections.append(("Document", content))
            
        return sections
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: The text to chunk
            
        Returns:
            List of text chunks
        """
        # If text is empty or smaller than chunk size, return as is
        if not text or len(text) <= self.chunk_size:
            return [text] if text else []
        
        # Split by paragraphs first (prefer to keep paragraphs intact)
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # If paragraph fits in current chunk
            if current_size + len(paragraph) + 1 <= self.chunk_size:
                current_chunk.append(paragraph)
                current_size += len(paragraph) + 1  # +1 for the newline
            else:
                # If current chunk is not empty, add it to chunks
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                
                # If paragraph is larger than chunk size, split it by sentences
                if len(paragraph) > self.chunk_size:
                    # Split by sentences
                    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                    current_chunk = []
                    current_size = 0
                    
                    for sentence in sentences:
                        if current_size + len(sentence) + 1 <= self.chunk_size:
                            current_chunk.append(sentence)
                            current_size += len(sentence) + 1
                        else:
                            if current_chunk:
                                chunks.append(" ".join(current_chunk))
                            
                            # If sentence is larger than chunk size, split it by words
                            if len(sentence) > self.chunk_size:
                                sentence_chunks = self._chunk_large_sentence(sentence)
                                chunks.extend(sentence_chunks)
                                current_chunk = []
                                current_size = 0
                            else:
                                current_chunk = [sentence]
                                current_size = len(sentence)
                    
                    # Add any remaining content in current chunk
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = []
                        current_size = 0
                else:
                    # Start a new chunk with the current paragraph
                    current_chunk = [paragraph]
                    current_size = len(paragraph)
        
        # Add final chunk if not empty
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        # Apply overlap between chunks if needed
        if self.chunk_overlap > 0 and len(chunks) > 1:
            return self._apply_overlap(chunks)
        
        return chunks
    
    def _chunk_large_sentence(self, sentence: str) -> List[str]:
        """
        Split a large sentence into chunks by characters.
        
        Args:
            sentence: The sentence to chunk
            
        Returns:
            List of chunks
        """
        chunk_size = self.chunk_size
        chunks = []
        
        for i in range(0, len(sentence), chunk_size):
            chunk = sentence[i:i+chunk_size]
            chunks.append(chunk)
            
        return chunks
    
    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """
        Apply overlap between chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of chunks with overlap
        """
        result = []
        for i in range(len(chunks)):
            if i == 0:
                result.append(chunks[i])
            else:
                # Add overlap from previous chunk
                previous_chunk = chunks[i-1]
                overlap_text = previous_chunk[-self.chunk_overlap:] if len(previous_chunk) > self.chunk_overlap else previous_chunk
                result.append(overlap_text + chunks[i])
                
        return result 