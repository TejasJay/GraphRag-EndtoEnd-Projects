"""
Loader factory for automatically selecting the appropriate loader based on file format.
"""

import os
from typing import List, Dict, Any
from pathlib import Path

from .base_loader import BaseLoader, Document
from .json_loader import JSONSchemaLoader
from .markdown_loader import MarkdownLoader
from .text_loader import TextLoader


class LoaderFactory:
    """Factory for creating appropriate document loaders."""
    
    # File extension to loader mapping
    LOADER_MAPPING = {
        '.json': JSONSchemaLoader,
        '.md': MarkdownLoader,
        '.markdown': MarkdownLoader,
        '.txt': TextLoader,
        '.text': TextLoader,
    }
    
    @classmethod
    def create_loader(cls, file_path: str, **kwargs) -> BaseLoader:
        """Create the appropriate loader for the given file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file extension
        extension = file_path.suffix.lower()
        
        # Get the appropriate loader class
        loader_class = cls.LOADER_MAPPING.get(extension)
        
        if loader_class is None:
            # Default to text loader for unknown extensions
            loader_class = TextLoader
        
        # Create and return the loader instance
        return loader_class(str(file_path), **kwargs)
    
    @classmethod
    def load_documents(cls, file_path: str, **kwargs) -> List[Document]:
        """Load documents from a file using the appropriate loader."""
        loader = cls.create_loader(file_path, **kwargs)
        return loader.load()
    
    @classmethod
    def load_multiple_files(cls, file_paths: List[str], **kwargs) -> List[Document]:
        """Load documents from multiple files."""
        all_documents = []
        
        for file_path in file_paths:
            try:
                documents = cls.load_documents(file_path, **kwargs)
                all_documents.extend(documents)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                continue
        
        return all_documents
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get list of supported file extensions."""
        return list(cls.LOADER_MAPPING.keys())
    
    @classmethod
    def get_file_info(cls, file_path: str, **kwargs) -> Dict[str, Any]:
        """Get information about a file using the appropriate loader."""
        loader = cls.create_loader(file_path, **kwargs)
        return loader.get_file_info()


def loader_factory(file_path: str, **kwargs) -> BaseLoader:
    """Convenience function for creating loaders."""
    return LoaderFactory.create_loader(file_path, **kwargs) 