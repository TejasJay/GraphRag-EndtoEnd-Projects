"""
Base loader class defining the interface for document loaders.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    """Canonical document object with metadata."""
    content: str
    metadata: Dict[str, Any]
    source: str
    chunk_id: Optional[str] = None


class BaseLoader(ABC):
    """Abstract base class for document loaders."""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
    
    @abstractmethod
    def load(self) -> List[Document]:
        """Load documents from the file and return canonical Document objects."""
        pass
    
    @abstractmethod
    def get_file_info(self) -> Dict[str, Any]:
        """Get information about the loaded file."""
        pass
    
    def validate(self) -> bool:
        """Validate that the file can be loaded by this loader."""
        return self.file_path.exists() and self.file_path.stat().st_size > 0 