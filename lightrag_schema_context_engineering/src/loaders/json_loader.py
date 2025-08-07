"""
JSON schema loader for handling schema exports.
Creates one chunk per node/relationship with structured metadata.
"""

import json
from typing import List, Dict, Any
from .base_loader import BaseLoader, Document


class JSONSchemaLoader(BaseLoader):
    """Loader for JSON schema exports."""
    
    def load(self) -> List[Document]:
        """Load JSON schema and create one chunk per node/relationship."""
        documents = []
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON schema formats
        if isinstance(data, dict):
            documents.extend(self._process_schema_dict(data))
        elif isinstance(data, list):
            documents.extend(self._process_schema_list(data))
        
        return documents
    
    def _process_schema_dict(self, data: Dict[str, Any]) -> List[Document]:
        """Process schema data in dictionary format."""
        documents = []
        
        # Handle nodes
        if 'nodes' in data:
            for node in data['nodes']:
                doc = self._create_node_document(node)
                documents.append(doc)
        
        # Handle relationships
        if 'relationships' in data:
            for rel in data['relationships']:
                doc = self._create_relationship_document(rel)
                documents.append(doc)
        
        # Handle properties
        if 'properties' in data:
            for prop_name, prop_info in data['properties'].items():
                doc = self._create_property_document(prop_name, prop_info)
                documents.append(doc)
        
        # Handle other schema elements
        for key, value in data.items():
            if key not in ['nodes', 'relationships', 'properties']:
                doc = self._create_generic_document(key, value)
                documents.append(doc)
        
        return documents
    
    def _process_schema_list(self, data: List[Dict[str, Any]]) -> List[Document]:
        """Process schema data in list format."""
        documents = []
        
        for item in data:
            if 'type' in item:
                if item['type'] == 'node':
                    doc = self._create_node_document(item)
                elif item['type'] == 'relationship':
                    doc = self._create_relationship_document(item)
                else:
                    doc = self._create_generic_document(item.get('type', 'unknown'), item)
                documents.append(doc)
            else:
                # Assume it's a node or relationship based on structure
                if 'labels' in item or 'label' in item:
                    doc = self._create_node_document(item)
                elif 'type' in item or 'relationship_type' in item:
                    doc = self._create_relationship_document(item)
                else:
                    doc = self._create_generic_document('schema_element', item)
                documents.append(doc)
        
        return documents
    
    def _create_node_document(self, node: Dict[str, Any]) -> Document:
        """Create a document for a node."""
        labels = node.get('labels', [])
        if isinstance(labels, str):
            labels = [labels]
        
        properties = node.get('properties', {})
        content = f"Node: {', '.join(labels)}\n"
        if properties:
            content += f"Properties: {json.dumps(properties, indent=2)}"
        
        metadata = {
            'node_type': 'node',
            'labels': labels,
            'properties': properties,
            'source_file': str(self.file_path),
            'chunk_type': 'node'
        }
        
        chunk_id = f"node_{'_'.join(labels)}_{hash(str(properties))}"
        
        return Document(
            content=content,
            metadata=metadata,
            source=str(self.file_path),
            chunk_id=chunk_id
        )
    
    def _create_relationship_document(self, rel: Dict[str, Any]) -> Document:
        """Create a document for a relationship."""
        rel_type = rel.get('type') or rel.get('relationship_type', 'UNKNOWN')
        start_node = rel.get('start_node', {})
        end_node = rel.get('end_node', {})
        properties = rel.get('properties', {})
        
        content = f"Relationship: {rel_type}\n"
        if start_node:
            content += f"From: {start_node}\n"
        if end_node:
            content += f"To: {end_node}\n"
        if properties:
            content += f"Properties: {json.dumps(properties, indent=2)}"
        
        metadata = {
            'node_type': 'relationship',
            'relationship_type': rel_type,
            'start_node': start_node,
            'end_node': end_node,
            'properties': properties,
            'source_file': str(self.file_path),
            'chunk_type': 'relationship'
        }
        
        chunk_id = f"rel_{rel_type}_{hash(str(properties))}"
        
        return Document(
            content=content,
            metadata=metadata,
            source=str(self.file_path),
            chunk_id=chunk_id
        )
    
    def _create_property_document(self, prop_name: str, prop_info: Dict[str, Any]) -> Document:
        """Create a document for a property definition."""
        content = f"Property: {prop_name}\n{json.dumps(prop_info, indent=2)}"
        
        metadata = {
            'node_type': 'property',
            'property_name': prop_name,
            'property_info': prop_info,
            'source_file': str(self.file_path),
            'chunk_type': 'property'
        }
        
        chunk_id = f"prop_{prop_name}"
        
        return Document(
            content=content,
            metadata=metadata,
            source=str(self.file_path),
            chunk_id=chunk_id
        )
    
    def _create_generic_document(self, element_type: str, data: Dict[str, Any]) -> Document:
        """Create a document for generic schema elements."""
        content = f"{element_type.title()}: {json.dumps(data, indent=2)}"
        
        metadata = {
            'node_type': 'schema_element',
            'element_type': element_type,
            'data': data,
            'source_file': str(self.file_path),
            'chunk_type': 'schema_element'
        }
        
        chunk_id = f"schema_{element_type}_{hash(str(data))}"
        
        return Document(
            content=content,
            metadata=metadata,
            source=str(self.file_path),
            chunk_id=chunk_id
        )
    
    def get_file_info(self) -> Dict[str, Any]:
        """Get information about the JSON schema file."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        info = {
            'file_path': str(self.file_path),
            'file_size': self.file_path.stat().st_size,
            'format': 'json'
        }
        
        if isinstance(data, dict):
            info['structure'] = 'dict'
            info['keys'] = list(data.keys())
        elif isinstance(data, list):
            info['structure'] = 'list'
            info['length'] = len(data)
        
        return info 