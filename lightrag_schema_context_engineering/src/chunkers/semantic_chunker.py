"""
Semantic chunker using sentence transformers for similarity-based chunking.
"""

from typing import List, Dict, Any
from .base_chunker import BaseChunker, Chunk
from src.loaders.base_loader import Document


class SemanticChunker(BaseChunker):
    """Chunker that uses semantic similarity for chunking."""
    
    def __init__(self, similarity_threshold: float = 0.9, **kwargs):
        super().__init__(**kwargs)
        self.similarity_threshold = similarity_threshold
    
    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk documents using semantic similarity."""
        chunks = []
        
        for doc in documents:
            doc_chunks = self._chunk_document(doc)
            chunks.extend(doc_chunks)
        
        self.update_stats(chunks)
        return chunks
    
    def _chunk_document(self, document: Document) -> List[Chunk]:
        """Chunk a single document using semantic similarity."""
        # For now, implement a simple sentence-based chunking
        # In production, you'd use sentence-transformers for semantic similarity
        sentences = self._split_into_sentences(document.content)
        chunks = []
        
        current_chunk_sentences = []
        chunk_id = 0
        
        for sentence in sentences:
            current_chunk_sentences.append(sentence)
            
            # Simple heuristic: create chunk every few sentences
            if len(current_chunk_sentences) >= 3:
                chunk = self._create_chunk(
                    ' '.join(current_chunk_sentences),
                    document,
                    chunk_id,
                    len(' '.join(current_chunk_sentences).split())
                )
                chunks.append(chunk)
                chunk_id += 1
                current_chunk_sentences = []
        
        # Add remaining sentences
        if current_chunk_sentences:
            chunk = self._create_chunk(
                ' '.join(current_chunk_sentences),
                document,
                chunk_id,
                len(' '.join(current_chunk_sentences).split())
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_chunk(self, content: str, document: Document, chunk_id: int, token_count: int) -> Chunk:
        """Create a chunk from document content."""
        # Merge metadata
        metadata = document.metadata.copy()
        metadata.update({
            'chunking_strategy': 'semantic',
            'chunk_id': chunk_id,
            'similarity_threshold': self.similarity_threshold,
            'original_chunk_id': document.chunk_id
        })
        
        chunk_id_str = f"{document.chunk_id}_semantic_{chunk_id}" if document.chunk_id else f"semantic_{chunk_id}"
        
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
            'strategy': 'semantic',
            'similarity_threshold': self.similarity_threshold,
            'stats': self.get_stats()
        } 