"""
Chunker factory for creating different chunking strategies.
"""

from typing import Dict, Any
from .base_chunker import BaseChunker
from .fixed_window_chunker import FixedWindowChunker
from .recursive_chunker import RecursiveChunker
from .graph_unit_chunker import GraphUnitChunker
from .semantic_chunker import SemanticChunker
from .sliding_window_chunker import SlidingWindowChunker


class ChunkerFactory:
    """Factory for creating chunking strategies."""
    
    # Strategy name to chunker class mapping
    CHUNKER_MAPPING = {
        'fixed_window': FixedWindowChunker,
        'recursive': RecursiveChunker,
        'graph_unit': GraphUnitChunker,
        'semantic': SemanticChunker,
        'sliding_window': SlidingWindowChunker,
    }
    
    @classmethod
    def create_chunker(cls, strategy_name: str, **kwargs) -> BaseChunker:
        """Create a chunker with the specified strategy."""
        chunker_class = cls.CHUNKER_MAPPING.get(strategy_name)
        
        if chunker_class is None:
            raise ValueError(f"Unknown chunking strategy: {strategy_name}. "
                           f"Available strategies: {list(cls.CHUNKER_MAPPING.keys())}")
        
        return chunker_class(**kwargs)
    
    @classmethod
    def get_available_strategies(cls) -> Dict[str, str]:
        """Get available chunking strategies with descriptions."""
        return {
            'fixed_window': 'Fixed token window with overlap (N tokens, M overlap)',
            'recursive': 'Recursive text splitting for mixed prose and code',
            'graph_unit': 'One chunk per graph unit (node/relationship)',
            'semantic': 'Semantic similarity-based chunking for long narrative files',
            'sliding_window': 'Sliding window with contextual compression for overlong context',
        }
    
    @classmethod
    def get_default_config(cls, strategy_name: str) -> Dict[str, Any]:
        """Get default configuration for a chunking strategy."""
        defaults = {
            'fixed_window': {
                'chunk_size': 512,
                'overlap': 128
            },
            'recursive': {
                'chunk_size': 512,
                'chunk_overlap': 128,
                'separators': ["\n\n", "\n", " ", ""]
            },
            'graph_unit': {},
            'semantic': {
                'similarity_threshold': 0.9
            },
            'sliding_window': {
                'window_size': 3,
                'compression_ratio': 0.5
            }
        }
        
        return defaults.get(strategy_name, {})


def chunker_factory(strategy_name: str, **kwargs) -> BaseChunker:
    """Convenience function for creating chunkers."""
    return ChunkerFactory.create_chunker(strategy_name, **kwargs) 