#!/usr/bin/env python3
"""
Test script to fix SME agent formatting issue with Gemini.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.agents import SMEAgent
from src.retrievers.base_retriever import RetrievalResult
from src.chunkers.base_chunker import Chunk

def test_sme_formatting():
    """Test SME agent formatting with a simple example."""
    
    # Create a simple SME agent
    sme_agent = SMEAgent(llm_model="gemini-1.5-flash")
    
    # Create mock retrieval results
    mock_chunks = [
        Chunk(
            chunk_id="test_chunk_1",
            content='Schema_Element: { "category": "relationship", "pattern": "(:User)-[:MEMBER_OF]->(:Group)", "content": "This relationship indicates that a User is a member of a specific Group." }',
            metadata={'chunk_type': 'schema_element'}
        ),
        Chunk(
            chunk_id="test_chunk_2",
            content='Schema_Element: { "category": "node", "label": "User", "properties": ["name", "email"] }',
            metadata={'chunk_type': 'schema_element'}
        ),
        Chunk(
            chunk_id="test_chunk_3",
            content='Schema_Element: { "category": "node", "label": "Group", "properties": ["name", "description"] }',
            metadata={'chunk_type': 'schema_element'}
        )
    ]
    
    mock_results = [
        RetrievalResult(chunk=chunk, score=0.5, rank=i+1, retrieval_method="test", metadata={})
        for i, chunk in enumerate(mock_chunks)
    ]
    
    # Test the SME agent
    question = "Show me all employees in the engineering department"
    
    print("🧪 Testing SME Agent Formatting...")
    print("=" * 50)
    print(f"Question: {question}")
    print(f"Mock chunks: {len(mock_chunks)}")
    
    try:
        response = sme_agent.process(question, mock_results)
        
        print("\n📋 SME Agent Response:")
        print(f"Content: {response.content}")
        print(f"Processing Time: {response.processing_time:.3f}s")
        print(f"LLM Model: {response.metadata.get('llm_model', 'unknown')}")
        
        # Check if response is properly formatted
        if "**Node Labels:**" in response.content or "Node Labels:" in response.content:
            print("✅ SME Agent formatting looks good!")
        elif "Schema_Element:" in response.content:
            print("❌ SME Agent still returning raw JSON")
        else:
            print("⚠️  SME Agent response format unclear")
            
    except Exception as e:
        print(f"❌ Error testing SME agent: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sme_formatting() 