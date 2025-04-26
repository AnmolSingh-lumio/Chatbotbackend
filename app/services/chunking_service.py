"""
Service for chunking markdown documents with advanced contract-specific handling.
"""
import os
import re
import logging
import markdown
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Any, Tuple, Optional
from math import ceil

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

class ChunkingService:
    """Service for chunking documents into manageable pieces with contract-specific enhancements."""
    
    def __init__(self, 
                 min_chunk_size: int = None,
                 max_chunk_size: int = None, 
                 overlap_size: int = None,
                 preserve_tables: bool = True):
        """
        Initialize the chunking service.
        
        Args:
            min_chunk_size: Minimum size of chunks in characters (default from env or 300)
            max_chunk_size: Maximum size of chunks in characters (default from env or 1500)
            overlap_size: Overlap between chunks in characters (default from env or 100)
            preserve_tables: Whether to keep tables intact as single chunks
        """
        # Read configuration from environment variables with defaults
        self.min_chunk_size = min_chunk_size or int(os.getenv("MIN_CHUNK_SIZE", "300"))
        self.max_chunk_size = max_chunk_size or int(os.getenv("MAX_CHUNK_SIZE", "1500"))
        self.overlap_size = overlap_size or int(os.getenv("CHUNK_OVERLAP", "100"))
        self.preserve_tables = preserve_tables
        
        logger.info(f"Initialized chunking service with min_size={self.min_chunk_size}, "
                   f"max_size={self.max_chunk_size}, overlap={self.overlap_size}, "
                   f"preserve_tables={self.preserve_tables}")
        
        # Try to download NLTK data if not already present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                logger.info("Downloading NLTK punkt tokenizer data...")
                nltk.download('punkt', quiet=True)
                logger.info("NLTK punkt tokenizer data downloaded successfully")
            except Exception as e:
                logger.warning(f"Could not download NLTK data: {e}. Falling back to simple sentence splitting.")
    
    def chunk_markdown(self, content: str) -> List[Dict[str, Any]]:
        """
        Split markdown content into semantically meaningful chunks with metadata.
        
        This method maintains backward compatibility with the original interface
        while providing advanced chunking capabilities.
        
        Args:
            content: The markdown content to chunk
            
        Returns:
            List of dictionaries with chunk content and metadata
        """
        # Extract document title and metadata
        title = self._extract_title(content)
        metadata = {
            'document_title': title
        }
        
        # Extract contract value if present in title
        contract_value = self._extract_contract_value(title)
        if contract_value:
            metadata['contract_value'] = contract_value
        
        # Process the document
        chunks = self._process_document(content, metadata)
        
        logger.info(f"Split document into {len(chunks)} chunks")
        return chunks
    
    def _process_document(self, document: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Process a full contract document into chunks.
        
        Args:
            document: The markdown document text
            metadata: Additional metadata about the document
            
        Returns:
            A list of chunks, each with content and metadata
        """
        if metadata is None:
            metadata = {}
        
        # Parse document structure to identify sections
        sections = self._parse_document_structure(document)
        
        # Process chunks
        chunks = []
        for section in sections:
            section_chunks = self._chunk_section(section)
            for chunk in section_chunks:
                # Enhance chunk metadata with section info
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'section': section['title'],
                    'section_level': section['level'],
                    'section_path': section['path']
                })
                
                # Extract contract-specific features
                features = self._extract_contract_features(chunk)
                if features:
                    chunk_metadata.update(features)
                
                chunks.append({
                    'content': chunk,
                    'metadata': chunk_metadata
                })
        
        # Post-process chunks to ensure they're suitable for embedding
        processed_chunks = self._post_process_chunks(chunks)
        
        # Add final metadata
        for i, chunk in enumerate(processed_chunks):
            chunk['metadata']['chunk_index'] = i
            chunk['metadata']['total_chunks'] = len(processed_chunks)
        
        return processed_chunks
    
    def _extract_title(self, document: str) -> str:
        """Extract the document title from the markdown."""
        # Look for a level 1 or 2 heading at the start of the document
        match = re.search(r'^#\s+(.+?)$|^##\s+(.+?)$', document, re.MULTILINE)
        if match:
            return match.group(1) or match.group(2)
        return "Untitled Document"
    
    def _extract_contract_value(self, title: str) -> Optional[str]:
        """Extract the contract value if present in the title."""
        match = re.search(r'\$(\d+(?:\.\d+)?[KMB]?)', title)
        if match:
            return match.group(1)
        return None
    
    def _parse_document_structure(self, document: str) -> List[Dict[str, Any]]:
        """
        Parse the document structure into sections based on markdown headings.
        
        Returns:
            A list of sections, each with title, content, level, and hierarchical path
        """
        try:
            # Convert markdown to HTML for easier parsing
            html = markdown.markdown(document)
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find all headings to determine structure
            headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            
            # If no headings, treat the entire document as one section
            if not headings:
                return [{
                    'title': 'Main Content',
                    'content': document,
                    'level': 0,
                    'path': 'Main Content'
                }]
                
            # Parse document into sections based on headings
            document_lines = document.split('\n')
            sections = []
            heading_positions = self._find_heading_positions(document_lines)
            
            if not heading_positions:
                return [{
                    'title': 'Main Content',
                    'content': document,
                    'level': 0,
                    'path': 'Main Content'
                }]
            
            # Track hierarchical path
            current_levels = [0, 0, 0, 0, 0, 0]  # For h1-h6
            current_titles = ['', '', '', '', '', '']
            
            # Process each heading and section
            for i, (level, title, line_num) in enumerate(heading_positions):
                # Determine section content
                if i < len(heading_positions) - 1:
                    next_heading_pos = heading_positions[i+1][2]
                    section_content = '\n'.join(document_lines[line_num:next_heading_pos])
                else:
                    section_content = '\n'.join(document_lines[line_num:])
                
                # Update hierarchical tracking
                level_idx = level - 1  # Convert 1-based to 0-based index
                current_levels[level_idx] += 1
                current_titles[level_idx] = title
                
                # Reset lower levels
                for j in range(level_idx + 1, 6):
                    current_levels[j] = 0
                    current_titles[j] = ''
                
                # Create path from hierarchy (non-empty titles)
                path_parts = [t for t in current_titles[:level_idx+1] if t]
                section_path = ' > '.join(path_parts)
                
                sections.append({
                    'title': title,
                    'content': section_content,
                    'level': level,
                    'path': section_path
                })
            
            return sections
            
        except Exception as e:
            logger.warning(f"Error parsing document structure with BS4: {e}. Falling back to simple parsing.")
            return self._parse_document_simple(document)
    
    def _find_heading_positions(self, document_lines: List[str]) -> List[Tuple[int, str, int]]:
        """
        Find all markdown headings with their levels, titles, and line positions.
        
        Returns list of tuples: (level, title, line_number)
        """
        heading_positions = []
        for i, line in enumerate(document_lines):
            match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                heading_positions.append((level, title, i))
        return heading_positions
    
    def _parse_document_simple(self, document: str) -> List[Dict[str, Any]]:
        """
        Parse document with a simple regex-based approach as fallback.
        """
        # Find all headers (# Header)
        header_pattern = r'^(#+)\s+(.+?)$'
        
        # Split content by headers
        lines = document.split('\n')
        sections = []
        current_section = {
            'title': 'Main Content',
            'content': [],
            'level': 0,
            'path': 'Main Content'
        }
        
        for line in lines:
            # Check if line is a header
            match = re.match(header_pattern, line, re.MULTILINE)
            if match:
                # Save previous section if there's content
                if current_section['content']:
                    current_section['content'] = '\n'.join(current_section['content'])
                    sections.append(current_section)
                
                # Start new section
                level = len(match.group(1))
                title = match.group(2)
                current_section = {
                    'title': title,
                    'content': [line],  # Include the header in the content
                    'level': level,
                    'path': title
                }
            else:
                current_section['content'].append(line)
        
        # Add the last section
        if current_section['content']:
            current_section['content'] = '\n'.join(current_section['content'])
            sections.append(current_section)
            
        # If no sections with headers were found, return the entire document
        if len(sections) == 1 and sections[0]['title'] == 'Main Content':
            sections[0]['content'] = document
            
        return sections
    
    def _chunk_section(self, section: Dict[str, Any]) -> List[str]:
        """
        Chunk a section's content, preserving tables and logical structures.
        
        Args:
            section: Section dictionary with content to chunk
            
        Returns:
            List of content chunks
        """
        content = section['content']
        
        # Detect tables in the content
        tables = self._extract_tables(content)
        
        # If we need to preserve tables, replace them with placeholders
        if self.preserve_tables and tables:
            content_with_placeholders, table_map = self._replace_tables_with_placeholders(content, tables)
        else:
            content_with_placeholders = content
            table_map = {}
        
        # Chunk the content
        chunks = self._create_chunks(content_with_placeholders)
        
        # Restore tables if needed
        if table_map:
            chunks = [self._restore_tables(chunk, table_map) for chunk in chunks]
            
            # Add standalone tables if they exceed max size
            for placeholder, table in table_map.items():
                # Check if this table is already fully included in any chunk
                if not any(placeholder in chunk for chunk in chunks):
                    # If table is too large, make it a standalone chunk
                    if len(table) > self.max_chunk_size:
                        chunks.append(table)
        
        return chunks
    
    def _extract_tables(self, content: str) -> List[str]:
        """
        Extract markdown tables from content.
        
        Args:
            content: Text content to extract tables from
            
        Returns:
            List of extracted tables as strings
        """
        # Regular expression to find markdown tables
        table_pattern = r'(\|[^\n]+\|\n\|[-:| ]+\|\n(?:\|[^\n]+\|\n)+)'
        tables = re.findall(table_pattern, content)
        
        # Also detect ASCII-art tables (simplified detection)
        ascii_table_pattern = r'(\+[-+]+\+\n(?:\|[^+]+\|\n)+\+[-+]+\+)'
        ascii_tables = re.findall(ascii_table_pattern, content)
        
        return tables + ascii_tables
    
    def _replace_tables_with_placeholders(self, content: str, tables: List[str]) -> Tuple[str, Dict[str, str]]:
        """
        Replace tables with placeholders to prevent breaking them during chunking.
        
        Args:
            content: Original content
            tables: List of tables to replace
            
        Returns:
            Tuple of (modified content, mapping of placeholders to original tables)
        """
        table_map = {}
        modified_content = content
        
        for i, table in enumerate(tables):
            placeholder = f"[[TABLE_PLACEHOLDER_{i}]]"
            table_map[placeholder] = table
            modified_content = modified_content.replace(table, placeholder)
            
        return modified_content, table_map
    
    def _restore_tables(self, chunk: str, table_map: Dict[str, str]) -> str:
        """
        Restore table placeholders with actual tables.
        
        Args:
            chunk: Chunk potentially containing table placeholders
            table_map: Mapping of placeholders to tables
            
        Returns:
            Chunk with tables restored
        """
        restored_chunk = chunk
        
        for placeholder, table in table_map.items():
            if placeholder in chunk:
                # If restoring the table would make the chunk too large, 
                # keep it as a separate chunk
                if len(chunk.replace(placeholder, table)) > self.max_chunk_size * 1.5:
                    # If the chunk only contains the placeholder, replace it
                    if chunk.strip() == placeholder:
                        restored_chunk = table
                    # Otherwise, keep the placeholder for handling in post-processing
                else:
                    restored_chunk = chunk.replace(placeholder, table)
                    
        return restored_chunk
    
    def _create_chunks(self, content: str) -> List[str]:
        """
        Create chunks from content trying to maintain sentence/paragraph boundaries.
        
        Args:
            content: Text content to chunk
            
        Returns:
            List of content chunks
        """
        # If content is small enough, return as single chunk
        if len(content) <= self.max_chunk_size:
            return [content]
        
        chunks = []
        
        # Try to use NLTK for better sentence tokenization
        try:
            sentences = sent_tokenize(content)
        except Exception as e:
            logger.warning(f"Error using NLTK tokenizer: {e}. Falling back to regex-based splitting.")
            sentences = re.split(r'(?<=[.!?])\s+', content)
        
        current_chunk = ""
        
        for sentence in sentences:
            # If this sentence alone exceeds max size, split it further
            if len(sentence) > self.max_chunk_size:
                # Add current chunk if not empty
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # Split long sentence
                sentence_chunks = self._split_long_sentence(sentence)
                chunks.extend(sentence_chunks)
                continue
            
            # Check if adding this sentence would exceed max size
            if len(current_chunk) + len(sentence) > self.max_chunk_size:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """
        Split a long sentence into smaller parts.
        
        Args:
            sentence: Long sentence to split
            
        Returns:
            List of smaller chunks
        """
        # Try to split on natural breaks first - semicolons, commas, etc.
        parts = re.split(r'[;:,]', sentence)
        
        chunks = []
        current_chunk = ""
        
        for part in parts:
            if len(part) > self.max_chunk_size:
                # If the part is still too long, force split it
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # Brute force split by character count with overlap
                for i in range(0, len(part), self.max_chunk_size - self.overlap_size):
                    end_idx = min(i + self.max_chunk_size, len(part))
                    chunks.append(part[i:end_idx])
            else:
                if len(current_chunk) + len(part) > self.max_chunk_size:
                    chunks.append(current_chunk)
                    current_chunk = part
                else:
                    current_chunk += part
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def _post_process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Perform post-processing on chunks to ensure they're suitable for embedding.
        
        Args:
            chunks: List of original chunks
            
        Returns:
            List of processed chunks
        """
        processed_chunks = []
        standalone_tables = []
        
        # First pass: identify standalone tables and small chunks
        for chunk in chunks:
            content = chunk['content']
            
            # Check if chunk is just a table
            if content.strip().startswith('|') and content.strip().endswith('|'):
                # Check if it's a full table
                if content.count('\n') >= 2 and '|-' in content:
                    standalone_tables.append(chunk)
                    continue
            
            # If the chunk is too small, flag it for merging
            if len(content) < self.min_chunk_size:
                chunk['metadata']['small_chunk'] = True
            
            processed_chunks.append(chunk)
        
        # Second pass: merge small chunks if possible
        if any(chunk.get('metadata', {}).get('small_chunk', False) for chunk in processed_chunks):
            processed_chunks = self._merge_small_chunks(processed_chunks)
        
        # Add standalone tables back
        processed_chunks.extend(standalone_tables)
        
        return processed_chunks
    
    def _merge_small_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge small chunks with neighbors while respecting section boundaries.
        
        Args:
            chunks: List of chunks with some marked as small
            
        Returns:
            List of merged chunks
        """
        result = []
        current_merged = None
        
        for chunk in chunks:
            # Skip small chunks that are just whitespace
            if chunk.get('metadata', {}).get('small_chunk', False) and len(chunk['content'].strip()) < 10:
                continue
                
            # If we don't have a current merged chunk, start with this one
            if current_merged is None:
                current_merged = chunk.copy()
                continue
            
            # Check if this chunk is small and should be merged
            if chunk.get('metadata', {}).get('small_chunk', False):
                # Check if sections are compatible for merging
                curr_section = current_merged['metadata'].get('section_path', '')
                next_section = chunk['metadata'].get('section_path', '')
                
                # Only merge if in same section path
                if curr_section == next_section or curr_section.startswith(next_section) or next_section.startswith(curr_section):
                    # Merge the chunks
                    combined_content = current_merged['content'] + "\n\n" + chunk['content']
                    
                    # If combined content is reasonable size, merge them
                    if len(combined_content) <= self.max_chunk_size * 1.2:
                        current_merged['content'] = combined_content
                        current_merged['metadata']['contains_merged_chunks'] = True
                        continue
            
            # If we reach here, we can't merge with current chunk
            result.append(current_merged)
            current_merged = chunk.copy()
        
        # Add the last merged chunk if it exists
        if current_merged:
            result.append(current_merged)
            
        return result
    
    def _extract_contract_features(self, content: str) -> Dict[str, Any]:
        """
        Extract contract-specific features from chunk content.
        
        Args:
            content: Chunk content
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Extract key-value pairs
        key_values = self._extract_key_values(content)
        if key_values:
            features['extracted_fields'] = key_values
        
        # Extract contract entities
        entities = self._extract_contract_entities(content)
        if any(v for v in entities.values()):
            features['entities'] = entities
        
        # Classify chunk type
        chunk_type = self._classify_chunk_type(content)
        if chunk_type != 'general_content':
            features['chunk_type'] = chunk_type
        
        # Extract table metadata if it's a table
        if '|' in content and '\n|' in content and '|-' in content:
            table_metadata = self._extract_table_metadata(content)
            if table_metadata:
                features['table_metadata'] = table_metadata
        
        return features
    
    def _extract_key_values(self, content: str) -> Dict[str, str]:
        """Extract key-value pairs from the content."""
        key_values = {}
        
        # Match patterns like "Term: Value" or "Key - Value"
        pattern = r'(?:^|\n)(?:\*\*)?([A-Za-z\s]+)(?:\*\*)?[\s]*[:-]\s*([^,\n]+)(?:,|\n|$)'
        matches = re.findall(pattern, content)
        
        for key, value in matches:
            key = key.strip().lower().replace(' ', '_')
            value = value.strip()
            key_values[key] = value
        
        return key_values
    
    def _extract_contract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract named entities like companies, locations, dates, and monetary values."""
        entities = {
            'companies': [],
            'locations': [],
            'dates': [],
            'monetary_values': []
        }
        
        # Find company names (simple heuristics)
        company_pattern = r'([A-Z][A-Za-z]+(?:,? Inc\.?|,? LLC|,? Corp\.?|,? Corporation|,? Co\.))'
        companies = re.findall(company_pattern, content)
        if companies:
            entities['companies'] = list(set(companies))
        
        # Find locations (simple pattern for addresses)
        location_pattern = r'([A-Za-z ]+, [A-Z]{2},? \d{5}(?:-\d{4})?)'
        locations = re.findall(location_pattern, content)
        if locations:
            entities['locations'] = list(set(locations))
        
        # Find dates
        date_pattern = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|[A-Z][a-z]+ \d{1,2},? \d{4})'
        dates = re.findall(date_pattern, content)
        if dates:
            entities['dates'] = list(set(dates))
        
        # Find monetary values
        money_pattern = r'(\$\d+(?:,\d+)*(?:\.\d+)?(?:\s*[KMB])?)'
        money_values = re.findall(money_pattern, content)
        if money_values:
            entities['monetary_values'] = list(set(money_values))
        
        # Remove empty lists
        return {k: v for k, v in entities.items() if v}
    
    def _classify_chunk_type(self, content: str) -> str:
        """Classify the type of contract chunk."""
        content_lower = content.lower()
        
        # Check for specific section indicators
        if 'pricing' in content_lower or 'rate' in content_lower or 'discount' in content_lower:
            if '|' in content and '\n|' in content:  # Likely a table
                return 'pricing_table'
            return 'pricing_terms'
            
        if 'service' in content_lower and ('level' in content_lower or 'commitment' in content_lower):
            return 'service_levels'
            
        if ('term' in content_lower and 'condition' in content_lower) or 'agree' in content_lower:
            return 'terms_conditions'
            
        if 'account' in content_lower and 'number' in content_lower:
            return 'account_details'
            
        if 'zone' in content_lower and ('definition' in content_lower or 'weight' in content_lower):
            return 'zone_definitions'
            
        if 'addendum' in content_lower or 'appendix' in content_lower:
            return 'addendum'
            
        return 'general_content'
    
    def _extract_table_metadata(self, table_content: str) -> Dict[str, Any]:
        """Extract metadata from a table."""
        metadata = {
            'table_rows': table_content.count('\n') + 1,
            'table_columns': len(table_content.split('\n')[0].split('|')) - 2 if '|' in table_content else 0,
        }
        
        # Attempt to determine table type
        lower_text = table_content.lower()
        
        if 'price' in lower_text or 'rate' in lower_text or 'cost' in lower_text:
            metadata['table_type'] = 'pricing'
        elif 'discount' in lower_text:
            metadata['table_type'] = 'discount'
        elif 'surcharge' in lower_text or 'fee' in lower_text:
            metadata['table_type'] = 'fees'
        elif 'weight' in lower_text and ('zone' in lower_text or 'tier' in lower_text):
            metadata['table_type'] = 'weight_zones'
        elif 'account' in lower_text or 'customer' in lower_text:
            metadata['table_type'] = 'account_info'
        elif 'term' in lower_text or 'condition' in lower_text:
            metadata['table_type'] = 'terms'
        else:
            metadata['table_type'] = 'unknown'
        
        # Try to parse table into dataframe and extract data type info
        if PANDAS_AVAILABLE:
            try:
                df = self._markdown_to_dataframe(table_content)
                # Save column headers
                metadata['columns'] = df.columns.tolist()
                # Try to determine if any column has numeric data
                numeric_cols = 0
                for col in df.columns:
                    try:
                        # Try to convert to numeric
                        df[col] = pd.to_numeric(df[col].str.replace('$', '').str.replace(',', ''))
                        numeric_cols += 1
                    except:
                        pass
                metadata['has_numeric_data'] = numeric_cols > 0
            except Exception as e:
                logger.debug(f"Could not parse table as dataframe: {e}")
        
        return metadata
    
    def _markdown_to_dataframe(self, markdown_table: str) -> 'pd.DataFrame':
        """Convert a markdown table to a pandas DataFrame."""
        if not PANDAS_AVAILABLE:
            return None
            
        try:
            # Split the table into lines
            lines = markdown_table.strip().split('\n')
            
            # Extract the header row
            header = lines[0].strip('|').split('|')
            header = [h.strip() for h in header]
            
            # Skip the separator row
            data_rows = []
            for line in lines[2:]:  # Skip header and separator
                if '|' in line:  # Make sure it's a table row
                    row = line.strip('|').split('|')
                    row = [cell.strip() for cell in row]
                    data_rows.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(data_rows, columns=header)
            return df
        except Exception as e:
            logger.debug(f"Failed to parse markdown table: {e}")
            # Return an empty DataFrame as fallback
            return pd.DataFrame()