"""
Base chunker class defining the interface for all chunking strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..loaders.base_loader import Document


@dataclass
class Chunk:
    """Chunk object with metadata."""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    token_count: Optional[int] = None
    source_documents: Optional[List[str]] = None


class BaseChunker(ABC):
    """Abstract base class for chunking strategies."""
    
    def __init__(self, **kwargs):
        self.config = kwargs
        self.stats = {
            'total_chunks': 0,
            'avg_tokens': 0,
            'std_dev_tokens': 0,
            'min_tokens': float('inf'),
            'max_tokens': 0
        }
    
    @abstractmethod
    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk a list of documents into smaller pieces."""
        pass
    
    @abstractmethod
    def get_chunking_info(self) -> Dict[str, Any]:
        """Get information about the chunking strategy and parameters."""
        pass
    
    def update_stats(self, chunks: List[Chunk]):
        """Update chunking statistics."""
        if not chunks:
            return
        
        token_counts = [chunk.token_count or len(chunk.content.split()) for chunk in chunks]
        
        self.stats['total_chunks'] = len(chunks)
        self.stats['avg_tokens'] = sum(token_counts) / len(token_counts)
        self.stats['min_tokens'] = min(token_counts)
        self.stats['max_tokens'] = max(token_counts)
        
        # Calculate standard deviation
        if len(token_counts) > 1:
            variance = sum((x - self.stats['avg_tokens']) ** 2 for x in token_counts) / (len(token_counts) - 1)
            self.stats['std_dev_tokens'] = variance ** 0.5
        else:
            self.stats['std_dev_tokens'] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current chunking statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset chunking statistics."""
        self.stats = {
            'total_chunks': 0,
            'avg_tokens': 0,
            'std_dev_tokens': 0,
            'min_tokens': float('inf'),
            'max_tokens': 0
        } 