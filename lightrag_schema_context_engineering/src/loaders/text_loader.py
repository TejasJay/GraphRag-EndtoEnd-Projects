"""
Text loader for handling plain text files.
Creates paragraph-level chunks with simple metadata.
"""

import re
from typing import List, Dict, Any
from .base_loader import BaseLoader, Document


class TextLoader(BaseLoader):
    """Loader for plain text files."""
    
    def __init__(self, file_path: str, chunk_by_paragraph: bool = True, max_chunk_size: int = 1000):
        super().__init__(file_path)
        self.chunk_by_paragraph = chunk_by_paragraph
        self.max_chunk_size = max_chunk_size
    
    def load(self) -> List[Document]:
        """Load text file and create chunks."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if self.chunk_by_paragraph:
            return self._chunk_by_paragraph(content)
        else:
            return self._chunk_by_size(content)
    
    def _chunk_by_paragraph(self, content: str) -> List[Document]:
        """Create chunks by paragraph."""
        documents = []
        
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', content)
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if paragraph:
                doc = self._create_paragraph_document(paragraph, i)
                documents.append(doc)
        
        return documents
    
    def _chunk_by_size(self, content: str) -> List[Document]:
        """Create chunks by size with overlap."""
        documents = []
        
        # Simple character-based chunking
        chunk_size = self.max_chunk_size
        overlap = chunk_size // 3
        
        start = 0
        chunk_id = 0
        
        while start < len(content):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(content):
                # Look for sentence endings
                sentence_end = content.rfind('.', start, end)
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
                else:
                    # Look for paragraph breaks
                    para_end = content.rfind('\n\n', start, end)
                    if para_end > start + chunk_size // 2:
                        end = para_end + 2
            
            chunk_content = content[start:end].strip()
            if chunk_content:
                doc = self._create_size_document(chunk_content, chunk_id, start, end)
                documents.append(doc)
                chunk_id += 1
            
            start = end - overlap
            if start >= len(content):
                break
        
        return documents
    
    def _create_paragraph_document(self, paragraph: str, paragraph_id: int) -> Document:
        """Create a document for a paragraph."""
        metadata = {
            'chunk_type': 'paragraph',
            'paragraph_id': paragraph_id,
            'source_file': str(self.file_path),
            'format': 'text',
            'word_count': len(paragraph.split())
        }
        
        chunk_id = f"txt_para_{paragraph_id}"
        
        return Document(
            content=paragraph,
            metadata=metadata,
            source=str(self.file_path),
            chunk_id=chunk_id
        )
    
    def _create_size_document(self, content: str, chunk_id: int, start: int, end: int) -> Document:
        """Create a document for a size-based chunk."""
        metadata = {
            'chunk_type': 'size_based',
            'chunk_id': chunk_id,
            'start_position': start,
            'end_position': end,
            'source_file': str(self.file_path),
            'format': 'text',
            'word_count': len(content.split())
        }
        
        chunk_id_str = f"txt_size_{chunk_id}"
        
        return Document(
            content=content,
            metadata=metadata,
            source=str(self.file_path),
            chunk_id=chunk_id_str
        )
    
    def get_file_info(self) -> Dict[str, Any]:
        """Get information about the text file."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count paragraphs
        paragraphs = re.split(r'\n\s*\n', content)
        paragraph_count = len([p for p in paragraphs if p.strip()])
        
        # Count words
        word_count = len(content.split())
        
        # Count sentences (simple heuristic)
        sentences = re.split(r'[.!?]+', content)
        sentence_count = len([s for s in sentences if s.strip()])
        
        return {
            'file_path': str(self.file_path),
            'file_size': self.file_path.stat().st_size,
            'format': 'text',
            'paragraph_count': paragraph_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'chunk_strategy': 'paragraph' if self.chunk_by_paragraph else 'size',
            'max_chunk_size': self.max_chunk_size
        } 