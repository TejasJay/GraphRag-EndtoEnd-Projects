"""
Graph unit chunker that creates one chunk per node or relationship.
Optimized for schema-heavy JSON data.
"""

from typing import List, Dict, Any
from .base_chunker import BaseChunker, Chunk
from src.loaders.base_loader import Document


class GraphUnitChunker(BaseChunker):
    """Chunker that creates one chunk per graph unit (node/relationship)."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk documents into graph units (one chunk per node/relationship)."""
        chunks = []
        
        for doc in documents:
            # For graph unit chunking, we typically don't split documents further
            # Each document should already represent a single graph unit
            chunk = self._create_graph_unit_chunk(doc)
            chunks.append(chunk)
        
        self.update_stats(chunks)
        return chunks
    
    def _create_graph_unit_chunk(self, document: Document) -> Chunk:
        """Create a chunk for a graph unit (node or relationship)."""
        # Merge metadata
        metadata = document.metadata.copy()
        metadata.update({
            'chunking_strategy': 'graph_unit',
            'graph_unit_type': document.metadata.get('chunk_type', 'unknown'),
            'original_chunk_id': document.chunk_id
        })
        
        # Use original chunk_id if available, otherwise create one
        chunk_id = document.chunk_id or f"graph_unit_{hash(document.content)}"
        
        # Estimate token count
        token_count = len(document.content.split())
        
        return Chunk(
            content=document.content,
            metadata=metadata,
            chunk_id=chunk_id,
            token_count=token_count,
            source_documents=[document.source]
        )
    
    def get_chunking_info(self) -> Dict[str, Any]:
        """Get information about the chunking strategy."""
        return {
            'strategy': 'graph_unit',
            'description': 'One chunk per graph unit (node/relationship)',
            'stats': self.get_stats()
        } 