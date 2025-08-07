"""
Recursive chunker using LangChain's recursive text splitter for mixed prose and code.
"""

from typing import List, Dict, Any
from .base_chunker import BaseChunker, Chunk
from src.loaders.base_loader import Document


class RecursiveChunker(BaseChunker):
    """Chunker that uses recursive text splitting for mixed content."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128, 
                 separators: List[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
    
    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk documents using recursive text splitting."""
        chunks = []
        
        for doc in documents:
            doc_chunks = self._chunk_document(doc)
            chunks.extend(doc_chunks)
        
        self.update_stats(chunks)
        return chunks
    
    def _chunk_document(self, document: Document) -> List[Chunk]:
        """Chunk a single document using recursive splitting."""
        chunks = []
        
        # Simple recursive splitting implementation
        # In production, you'd use LangChain's RecursiveCharacterTextSplitter
        text = document.content
        current_chunk = ""
        chunk_id = 0
        
        # Split by separators in order of preference
        for separator in self.separators:
            if separator in text:
                parts = text.split(separator)
                break
        else:
            # No separators found, split by characters
            parts = [text]
        
        for part in parts:
            if not part.strip():
                continue
                
            # If adding this part would exceed chunk size
            if len(current_chunk.split()) + len(part.split()) > self.chunk_size:
                if current_chunk.strip():
                    # Save current chunk
                    chunk = self._create_chunk(
                        current_chunk.strip(),
                        document,
                        chunk_id,
                        len(current_chunk.split())
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                    
                    # Start new chunk with overlap
                    if self.chunk_overlap > 0:
                        words = current_chunk.split()
                        overlap_words = words[-self.chunk_overlap:] if len(words) >= self.chunk_overlap else words
                        current_chunk = ' '.join(overlap_words) + separator + part
                    else:
                        current_chunk = part
                else:
                    # Current chunk is empty, start with this part
                    current_chunk = part
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += separator + part
                else:
                    current_chunk = part
        
        # Add final chunk
        if current_chunk.strip():
            chunk = self._create_chunk(
                current_chunk.strip(),
                document,
                chunk_id,
                len(current_chunk.split())
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, content: str, document: Document, chunk_id: int, token_count: int) -> Chunk:
        """Create a chunk from document content."""
        # Merge metadata
        metadata = document.metadata.copy()
        metadata.update({
            'chunking_strategy': 'recursive',
            'chunk_id': chunk_id,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'separators': self.separators,
            'original_chunk_id': document.chunk_id
        })
        
        chunk_id_str = f"{document.chunk_id}_recursive_{chunk_id}" if document.chunk_id else f"recursive_{chunk_id}"
        
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
            'strategy': 'recursive',
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'separators': self.separators,
            'stats': self.get_stats()
        } 