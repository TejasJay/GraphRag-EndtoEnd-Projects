"""
Markdown loader for handling design documents.
Creates paragraph-level chunks with section headers as metadata.
"""

import re
from typing import List, Dict, Any
from .base_loader import BaseLoader, Document


class MarkdownLoader(BaseLoader):
    """Loader for Markdown design documents."""
    
    def __init__(self, file_path: str, chunk_by_paragraph: bool = True):
        super().__init__(file_path)
        self.chunk_by_paragraph = chunk_by_paragraph
    
    def load(self) -> List[Document]:
        """Load Markdown file and create chunks by paragraph or section."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if self.chunk_by_paragraph:
            return self._chunk_by_paragraph(content)
        else:
            return self._chunk_by_section(content)
    
    def _chunk_by_paragraph(self, content: str) -> List[Document]:
        """Create chunks by paragraph with section headers as metadata."""
        documents = []
        
        # Split content into lines
        lines = content.split('\n')
        
        current_section = "Introduction"
        current_paragraph = []
        paragraph_count = 0
        
        for line in lines:
            # Check if line is a header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if header_match:
                # Save previous paragraph if it exists
                if current_paragraph:
                    doc = self._create_paragraph_document(
                        current_paragraph, 
                        current_section, 
                        paragraph_count
                    )
                    documents.append(doc)
                    paragraph_count += 1
                
                # Start new section
                level = len(header_match.group(1))
                current_section = header_match.group(2).strip()
                current_paragraph = []
                
                # Create document for the header itself
                header_doc = self._create_header_document(current_section, level)
                documents.append(header_doc)
                
            elif line.strip():
                # Non-empty line - add to current paragraph
                current_paragraph.append(line.strip())
            else:
                # Empty line - end of paragraph
                if current_paragraph:
                    doc = self._create_paragraph_document(
                        current_paragraph, 
                        current_section, 
                        paragraph_count
                    )
                    documents.append(doc)
                    paragraph_count += 1
                    current_paragraph = []
        
        # Handle last paragraph
        if current_paragraph:
            doc = self._create_paragraph_document(
                current_paragraph, 
                current_section, 
                paragraph_count
            )
            documents.append(doc)
        
        return documents
    
    def _chunk_by_section(self, content: str) -> List[Document]:
        """Create chunks by section."""
        documents = []
        
        # Split by headers
        sections = re.split(r'^(#{1,6}\s+.+)$', content, flags=re.MULTILINE)
        
        current_section_title = "Introduction"
        current_section_content = []
        
        for i, section in enumerate(sections):
            if re.match(r'^#{1,6}\s+.+$', section.strip()):
                # This is a header
                if current_section_content:
                    # Save previous section
                    doc = self._create_section_document(
                        current_section_content,
                        current_section_title
                    )
                    documents.append(doc)
                
                # Start new section
                current_section_title = section.strip()
                current_section_content = []
            else:
                # This is content
                current_section_content.append(section.strip())
        
        # Handle last section
        if current_section_content:
            doc = self._create_section_document(
                current_section_content,
                current_section_title
            )
            documents.append(doc)
        
        return documents
    
    def _create_paragraph_document(self, paragraph_lines: List[str], section: str, paragraph_id: int) -> Document:
        """Create a document for a paragraph."""
        content = '\n'.join(paragraph_lines)
        
        metadata = {
            'chunk_type': 'paragraph',
            'section': section,
            'paragraph_id': paragraph_id,
            'source_file': str(self.file_path),
            'format': 'markdown'
        }
        
        chunk_id = f"md_para_{section.replace(' ', '_')}_{paragraph_id}"
        
        return Document(
            content=content,
            metadata=metadata,
            source=str(self.file_path),
            chunk_id=chunk_id
        )
    
    def _create_header_document(self, header: str, level: int) -> Document:
        """Create a document for a header."""
        content = header
        
        metadata = {
            'chunk_type': 'header',
            'header_level': level,
            'section': header,
            'source_file': str(self.file_path),
            'format': 'markdown'
        }
        
        chunk_id = f"md_header_{header.replace(' ', '_')}"
        
        return Document(
            content=content,
            metadata=metadata,
            source=str(self.file_path),
            chunk_id=chunk_id
        )
    
    def _create_section_document(self, section_lines: List[str], section_title: str) -> Document:
        """Create a document for a section."""
        content = f"{section_title}\n\n" + '\n'.join(section_lines)
        
        metadata = {
            'chunk_type': 'section',
            'section': section_title,
            'source_file': str(self.file_path),
            'format': 'markdown'
        }
        
        chunk_id = f"md_section_{section_title.replace(' ', '_')}"
        
        return Document(
            content=content,
            metadata=metadata,
            source=str(self.file_path),
            chunk_id=chunk_id
        )
    
    def get_file_info(self) -> Dict[str, Any]:
        """Get information about the Markdown file."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count headers
        headers = re.findall(r'^(#{1,6})\s+(.+)$', content, flags=re.MULTILINE)
        header_counts = {}
        for header in headers:
            level = len(header[0])
            header_counts[f"h{level}"] = header_counts.get(f"h{level}", 0) + 1
        
        # Count paragraphs
        paragraphs = re.split(r'\n\s*\n', content)
        paragraph_count = len([p for p in paragraphs if p.strip()])
        
        return {
            'file_path': str(self.file_path),
            'file_size': self.file_path.stat().st_size,
            'format': 'markdown',
            'total_headers': len(headers),
            'header_distribution': header_counts,
            'paragraph_count': paragraph_count,
            'chunk_strategy': 'paragraph' if self.chunk_by_paragraph else 'section'
        } 