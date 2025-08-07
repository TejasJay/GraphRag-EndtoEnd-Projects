"""
Sliding window chunker with contextual compression for summarizing overlong context.
"""

from typing import List, Dict, Any
from .base_chunker import BaseChunker, Chunk
from src.loaders.base_loader import Document


class SlidingWindowChunker(BaseChunker):
    """Chunker that uses sliding window with contextual compression."""
    
    def __init__(self, window_size: int = 3, compression_ratio: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.compression_ratio = compression_ratio
    
    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk documents using sliding window with compression."""
        chunks = []
        
        # First, create initial chunks using a simple approach
        initial_chunks = []
        for doc in documents:
            # Split document into smaller pieces
            doc_chunks = self._split_document(doc)
            initial_chunks.extend(doc_chunks)
        
        # Apply sliding window compression
        compressed_chunks = self._apply_sliding_window_compression(initial_chunks)
        chunks.extend(compressed_chunks)
        
        self.update_stats(chunks)
        return chunks
    
    def _split_document(self, document: Document) -> List[Chunk]:
        """Split a document into smaller chunks."""
        # Simple splitting by paragraphs or sentences
        paragraphs = document.content.split('\n\n')
        chunks = []
        
        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                chunk = self._create_basic_chunk(
                    paragraph.strip(),
                    document,
                    i
                )
                chunks.append(chunk)
        
        return chunks
    
    def _apply_sliding_window_compression(self, chunks: List[Chunk]) -> List[Chunk]:
        """Apply sliding window compression to chunks."""
        compressed_chunks = []
        
        for i in range(0, len(chunks), self.window_size):
            window_chunks = chunks[i:i + self.window_size]
            
            if len(window_chunks) == 1:
                # Single chunk, no compression needed
                compressed_chunks.append(window_chunks[0])
            else:
                # Compress multiple chunks
                compressed_chunk = self._compress_window(window_chunks)
                compressed_chunks.append(compressed_chunk)
        
        return compressed_chunks
    
    def _compress_window(self, window_chunks: List[Chunk]) -> Chunk:
        """Compress a window of chunks into a single chunk."""
        # Combine content from all chunks in the window
        combined_content = '\n\n'.join([chunk.content for chunk in window_chunks])
        
        # Simple compression: take first part of each chunk
        compressed_content = self._simple_compression(combined_content)
        
        # Merge metadata from all chunks
        merged_metadata = {}
        for chunk in window_chunks:
            merged_metadata.update(chunk.metadata)
        
        merged_metadata.update({
            'chunking_strategy': 'sliding_window',
            'window_size': self.window_size,
            'compression_ratio': self.compression_ratio,
            'compressed_from': len(window_chunks)
        })
        
        # Create compressed chunk
        chunk_id = f"sliding_window_{hash(combined_content)}"
        
        return Chunk(
            content=compressed_content,
            metadata=merged_metadata,
            chunk_id=chunk_id,
            token_count=len(compressed_content.split()),
            source_documents=list(set([chunk.source_documents[0] for chunk in window_chunks if chunk.source_documents]))
        )
    
    def _simple_compression(self, content: str) -> str:
        """Simple compression by taking first part of content."""
        sentences = content.split('.')
        target_sentences = int(len(sentences) * self.compression_ratio)
        compressed_sentences = sentences[:max(1, target_sentences)]
        return '. '.join(compressed_sentences) + ('.' if compressed_sentences[-1] else '')
    
    def _create_basic_chunk(self, content: str, document: Document, chunk_id: int) -> Chunk:
        """Create a basic chunk from document content."""
        metadata = document.metadata.copy()
        metadata.update({
            'chunking_strategy': 'sliding_window_basic',
            'chunk_id': chunk_id,
            'original_chunk_id': document.chunk_id
        })
        
        chunk_id_str = f"{document.chunk_id}_sliding_basic_{chunk_id}" if document.chunk_id else f"sliding_basic_{chunk_id}"
        
        return Chunk(
            content=content,
            metadata=metadata,
            chunk_id=chunk_id_str,
            token_count=len(content.split()),
            source_documents=[document.source]
        )
    
    def get_chunking_info(self) -> Dict[str, Any]:
        """Get information about the chunking strategy."""
        return {
            'strategy': 'sliding_window',
            'window_size': self.window_size,
            'compression_ratio': self.compression_ratio,
            'stats': self.get_stats()
        } 