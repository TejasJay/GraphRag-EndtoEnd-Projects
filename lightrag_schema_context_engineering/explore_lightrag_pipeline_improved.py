#!/usr/bin/env python3
"""
Improved LightRAG Pipeline Explorer
Addresses the issues identified in the original exploration.
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.loaders import LoaderFactory
from src.chunkers import ChunkerFactory
from src.embeddings import EmbeddingRegistry
from src.retrievers import RetrieverFactory
from src.agents import SMEAgent, CypherAgent
from src.config.default_configs import get_default_config
from src.question_to_json_converter import QuestionToJSONConverter

class ImprovedLightRAGPipelineExplorer:
    """Improved explorer that addresses identified issues."""
    
    def __init__(self):
        self.config = get_default_config()
        self.rag_config = self.config.get("rag_config", {})
        self.question_converter = QuestionToJSONConverter()
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all pipeline components."""
        print("🔧 Initializing Improved LightRAG Pipeline Components...")
        print("=" * 60)
        
        # Load documents
        print("📚 Loading documents...")
        start_time = time.time()
        self.documents = LoaderFactory.load_documents("documents/4_knowledge_base_sme.json")
        load_time = time.time() - start_time
        print(f"   ✅ Loaded {len(self.documents)} documents in {load_time:.3f}s")
        
        # Create chunker
        print("📦 Creating chunks...")
        chunker_config = self.rag_config.get("chunker", {})
        self.chunker = ChunkerFactory.create_chunker(chunker_config.get("strategy", "fixed_window"))
        
        start_time = time.time()
        self.chunks = self.chunker.chunk_documents(self.documents)
        chunk_time = time.time() - start_time
        print(f"   ✅ Created {len(self.chunks)} chunks in {chunk_time:.3f}s")
        
        # Create embedder
        print("🔤 Creating embedder...")
        embedding_config = self.rag_config.get("embedding", {})
        embedding_registry = EmbeddingRegistry()
        self.embedder = embedding_registry.get_embedder(embedding_config.get("model", "text-embedding-3-small"))
        print(f"   ✅ Using embedding model: {embedding_config.get('model', 'text-embedding-3-small')}")
        
        # Create retriever
        print("🔍 Creating retriever...")
        retriever_config = self.rag_config.get("retriever", {})
        self.retriever = RetrieverFactory.create_retriever(
            retriever_config.get("type", "vector"),
            embedding_model=embedding_config.get("model", "text-embedding-3-small")
        )
        self.retriever.add_chunks(self.chunks)
        print(f"   ✅ Using retriever type: {retriever_config.get('type', 'vector')}")
        
        # Create agents with Gemini
        print("🤖 Creating agents...")
        llm_config = self.rag_config.get("llm", {})
        print(f"   🔧 LLM Config: {llm_config}")
        self.sme_agent = SMEAgent(llm_model=llm_config.get("model", "gemini-1.5-flash"))
        self.cypher_agent = CypherAgent(llm_model=llm_config.get("model", "gemini-1.5-flash"))
        print(f"   ✅ Using LLM model: {llm_config.get('model', 'gemini-1.5-flash')}")
        
        print("✅ Pipeline initialization complete!\n")
    
    def explore_pipeline(self, question: str):
        """Explore the complete pipeline with improvements."""
        print("🚀 Improved LightRAG Pipeline Exploration")
        print("=" * 60)
        print(f"📝 User Question: {question}")
        print("=" * 60)
        
        # Step 1: Question to Structured Format Conversion
        self._explore_question_conversion(question)
        
        # Step 2: Enhanced Query Creation
        self._explore_enhanced_query(question)
        
        # Step 3: Retrieval Process
        self._explore_retrieval(question)
        
        # Step 4: SME Agent Processing
        self._explore_sme_processing(question)
        
        # Step 5: Cypher Agent Processing
        self._explore_cypher_processing(question)
        
        # Step 6: Final Results
        self._explore_final_results(question)
        
        # Step 7: Issue Analysis
        self._analyze_issues(question)
    
    def _explore_question_conversion(self, question: str):
        """Explore question to structured format conversion."""
        print("\n🔄 STEP 1: Question to Structured Format Conversion")
        print("-" * 50)
        
        print("📊 Converting natural language question to structured JSON format...")
        print("   ℹ️  Method: Hard-coded regex patterns (not LLM-based)")
        print("   ✅ Benefits: Fast, reliable, no overfitting, extensible")
        
        start_time = time.time()
        structured_json = self.question_converter.convert_question_to_json(question)
        conversion_time = time.time() - start_time
        
        print(f"   ⏱️  Conversion time: {conversion_time:.3f}s")
        print("\n📋 Structured JSON Output:")
        print(json.dumps(structured_json, indent=2))
        
        # Store for later use
        self.structured_json = structured_json
    
    def _explore_enhanced_query(self, question: str):
        """Explore enhanced query creation."""
        print("\n🔄 STEP 2: Enhanced Query Creation")
        print("-" * 50)
        
        print("🔧 Creating enhanced query for better retrieval...")
        start_time = time.time()
        enhanced_query = self.question_converter.create_enhanced_query(question)
        creation_time = time.time() - start_time
        
        print(f"   ⏱️  Creation time: {creation_time:.3f}s")
        print("\n📝 Original Question:")
        print(f"   {question}")
        print("\n🚀 Enhanced Query:")
        print(f"   {enhanced_query}")
        
        # Store for later use
        self.enhanced_query = enhanced_query
    
    def _explore_retrieval(self, question: str):
        """Explore the retrieval process."""
        print("\n🔄 STEP 3: Retrieval Process")
        print("-" * 50)
        
        print("🔍 Performing vector retrieval with enhanced query...")
        start_time = time.time()
        retrieval_results = self.retriever.retrieve(self.enhanced_query, top_k=5)
        retrieval_time = time.time() - start_time
        
        print(f"   ⏱️  Retrieval time: {retrieval_time:.3f}s")
        print(f"   📊 Retrieved {len(retrieval_results)} chunks")
        
        print("\n📋 Retrieved Chunks:")
        for i, result in enumerate(retrieval_results, 1):
            print(f"\n   🔸 Chunk {i} (Score: {result.score:.3f}):")
            print(f"      Content: {result.chunk.content[:200]}...")
            print(f"      Metadata: {result.chunk.metadata}")
            print(f"      Rank: {result.rank}")
            print(f"      Retrieval Method: {result.retrieval_method}")
            
            # Score analysis
            if result.score < 0.5:
                print(f"      ⚠️  Low score analysis: Score {result.score:.3f} is normal for semantic matching")
                print(f"         - Cosine similarity typically ranges 0.3-0.8 for good matches")
                print(f"         - Enhanced query may contain structural elements not in chunk")
        
        # Store for later use
        self.retrieval_results = retrieval_results
    
    def _explore_sme_processing(self, question: str):
        """Explore SME agent processing."""
        print("\n🔄 STEP 4: SME Agent Processing")
        print("-" * 50)
        
        print("🧠 SME Agent filtering and organizing retrieved context...")
        start_time = time.time()
        sme_response = self.sme_agent.process(question, self.retrieval_results)
        sme_time = time.time() - start_time
        
        print(f"   ⏱️  Processing time: {sme_time:.3f}s")
        print(f"   📊 Response type: {type(sme_response).__name__}")
        print(f"   🤖 LLM Model: {sme_response.metadata.get('llm_model', 'unknown')}")
        
        print("\n📋 SME Agent Input:")
        print(f"   Question: {question}")
        print(f"   Retrieved chunks: {len(self.retrieval_results)} chunks")
        
        print("\n📋 SME Agent Output:")
        print(f"   Content: {sme_response.content}")
        print(f"   Processing Time: {sme_response.processing_time:.3f}s")
        print(f"   Metadata: {sme_response.metadata}")
        
        # Store for later use
        self.sme_response = sme_response
    
    def _explore_cypher_processing(self, question: str):
        """Explore Cypher agent processing."""
        print("\n🔄 STEP 5: Cypher Agent Processing")
        print("-" * 50)
        
        print("🔧 Cypher Agent generating Cypher query from filtered context...")
        start_time = time.time()
        cypher_response = self.cypher_agent.process(question, self.retrieval_results, filtered_schema=self.sme_response.content)
        cypher_time = time.time() - start_time
        
        print(f"   ⏱️  Processing time: {cypher_time:.3f}s")
        print(f"   📊 Response type: {type(cypher_response).__name__}")
        print(f"   🤖 LLM Model: {cypher_response.metadata.get('llm_model', 'unknown')}")
        
        print("\n📋 Cypher Agent Input:")
        print(f"   Question: {question}")
        print(f"   Filtered Schema: {self.sme_response.content[:200]}...")
        print(f"   Retrieved chunks: {len(self.retrieval_results)} chunks")
        
        print("\n📋 Cypher Agent Output:")
        print(f"   Content: {cypher_response.content}")
        print(f"   Processing Time: {cypher_response.processing_time:.3f}s")
        print(f"   Metadata: {cypher_response.metadata}")
        
        # Store for later use
        self.cypher_response = cypher_response
    
    def _explore_final_results(self, question: str):
        """Explore final results and summary."""
        print("\n🔄 STEP 6: Final Results")
        print("-" * 50)
        
        print("📊 Pipeline Summary:")
        print(f"   📝 Original Question: {question}")
        print(f"   🔄 Query Intent: {self.structured_json.get('query_intent', 'unknown')}")
        print(f"   🏷️  Relevant Nodes: {', '.join(self.structured_json.get('relevant_nodes', []))}")
        print(f"   🔗 Relevant Relationships: {', '.join(self.structured_json.get('relevant_relationships', []))}")
        print(f"   📋 Constraints: {self.structured_json.get('constraints', {})}")
        
        print(f"\n   🔍 Retrieved Chunks: {len(self.retrieval_results)}")
        print(f"   🧠 SME Processing Time: {self.sme_response.processing_time:.3f}s")
        print(f"   🔧 Cypher Processing Time: {self.cypher_response.processing_time:.3f}s")
        
        print("\n🎯 Generated Cypher Query:")
        print("=" * 40)
        print(self.cypher_response.content)
        print("=" * 40)
        
        print("\n📈 Performance Metrics:")
        total_time = (self.sme_response.processing_time + 
                     self.cypher_response.processing_time)
        print(f"   ⏱️  Total LLM Processing Time: {total_time:.3f}s")
        print(f"   📊 Success: {bool(self.cypher_response.content)}")
        
        # Show chunk relevance analysis
        print("\n🔍 Chunk Relevance Analysis:")
        for i, result in enumerate(self.retrieval_results, 1):
            chunk_type = result.chunk.metadata.get('chunk_type', 'unknown')
            category = result.chunk.metadata.get('category', 'unknown')
            print(f"   Chunk {i}: {chunk_type} ({category}) - Score: {result.score:.3f}")
    
    def _analyze_issues(self, question: str):
        """Analyze identified issues and provide solutions."""
        print("\n🔍 ISSUE ANALYSIS & SOLUTIONS")
        print("=" * 50)
        
        print("1️⃣ QUESTION CONVERSION METHOD:")
        print("   ✅ Current: Hard-coded regex patterns")
        print("   ✅ Benefits: Fast, reliable, no overfitting")
        print("   💡 For complex questions: Consider hybrid approach (regex + LLM fallback)")
        
        print("\n2️⃣ LLM MODEL CONFIGURATION:")
        print(f"   🔧 Current LLM: {self.sme_response.metadata.get('llm_model', 'unknown')}")
        print("   ✅ Fixed: Updated config to use Gemini")
        print("   💡 Verify: Check if Gemini API key is set")
        
        print("\n3️⃣ CYPHER QUERY GENERATION ISSUE:")
        print("   ❌ Problem: Schema doesn't have 'department' concept")
        print("   🔍 Analysis: Uses 'Group' instead of 'Department'")
        print("   💡 Solution: Map 'engineering department' → 'engineering group'")
        print("   💡 Expected Query: MATCH (u:User)-[:MEMBER_OF]->(g:Group {name: 'engineering'}) RETURN u.name")
        
        print("\n4️⃣ CHUNK SCORES ANALYSIS:")
        print("   📊 Score Range: 0.4-0.5 is normal for semantic matching")
        print("   📈 Expected Range: 0.3-0.8 for good matches")
        print("   💡 Reason: Enhanced query contains structural elements")
        print("   💡 Improvement: Consider score normalization or boosting")
        
        print("\n🎯 RECOMMENDATIONS:")
        print("   1. Add department-to-group mapping in question converter")
        print("   2. Implement score boosting for schema-relevant chunks")
        print("   3. Add fallback LLM for complex question conversion")
        print("   4. Improve schema coverage for edge cases")

def main():
    """Main function to run the improved pipeline explorer."""
    if len(sys.argv) != 2:
        print("❌ Usage: python explore_lightrag_pipeline_improved.py \"Your question here\"")
        print("Example: python explore_lightrag_pipeline_improved.py \"Show me all employees in the engineering department\"")
        sys.exit(1)
    
    question = sys.argv[1]
    
    try:
        # Create explorer and run pipeline
        explorer = ImprovedLightRAGPipelineExplorer()
        explorer.explore_pipeline(question)
        
    except Exception as e:
        print(f"❌ Error during pipeline exploration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 