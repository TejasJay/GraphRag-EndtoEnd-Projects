"""
Fixed window chunker for splitting text into fixed token windows with overlap.
"""

import re
from typing import List, Dict, Any
from .base_chunker import BaseChunker, Chunk
from src.loaders.base_loader import Document


class FixedWindowChunker(BaseChunker):
    """Chunker that splits text into fixed token windows with overlap."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk documents into fixed token windows."""
        chunks = []
        
        for doc in documents:
            doc_chunks = self._chunk_document(doc)
            chunks.extend(doc_chunks)
        
        self.update_stats(chunks)
        return chunks
    
    def _chunk_document(self, document: Document) -> List[Chunk]:
        """Chunk a single document."""
        chunks = []
        
        # Split content into tokens (simple word-based tokenization)
        tokens = document.content.split()
        
        if len(tokens) <= self.chunk_size:
            # Document fits in one chunk
            chunk = self._create_chunk(
                ' '.join(tokens),
                document,
                0,
                len(tokens)
            )
            chunks.append(chunk)
        else:
            # Need multiple chunks
            start = 0
            chunk_id = 0
            
            while start < len(tokens):
                end = min(start + self.chunk_size, len(tokens))
                
                # Create chunk
                chunk_tokens = tokens[start:end]
                chunk = self._create_chunk(
                    ' '.join(chunk_tokens),
                    document,
                    chunk_id,
                    len(chunk_tokens)
                )
                chunks.append(chunk)
                
                # Move to next chunk with overlap
                start = end - self.overlap
                chunk_id += 1
                
                # Ensure we don't get stuck in infinite loop
                if start >= len(tokens) - 1:
                    break
        
        return chunks
    
    def _create_chunk(self, content: str, document: Document, chunk_id: int, token_count: int) -> Chunk:
        """Create a chunk from document content."""
        # Merge metadata
        metadata = document.metadata.copy()
        metadata.update({
            'chunking_strategy': 'fixed_window',
            'chunk_id': chunk_id,
            'chunk_size': self.chunk_size,
            'overlap': self.overlap,
            'original_chunk_id': document.chunk_id
        })
        
        chunk_id_str = f"{document.chunk_id}_fixed_{chunk_id}" if document.chunk_id else f"fixed_{chunk_id}"
        
        return Chunk(
            content=content,
            metadata=metadata,
            chunk_id=chunk_id_str,
            token_count=token_count,
            source_documents=[document.source]
        )
    
    def get_chunking_info(self) -> Dict[str, Any]:
        """Get information about the chunking strategy."""
        return {
            'strategy': 'fixed_window',
            'chunk_size': self.chunk_size,
            'overlap': self.overlap,
            'overlap_ratio': self.overlap / self.chunk_size if self.chunk_size > 0 else 0,
            'stats': self.get_stats()
        } 