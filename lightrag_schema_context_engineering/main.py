"""
LightRAG Production API

FastAPI backend for the LightRAG system, optimized for GCP Cloud Run deployment.
"""

import os
import sys
import time
import json
from typing import Dict, Any, Optional, List
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import LightRAG components
from src.loaders import LoaderFactory
from src.chunkers import ChunkerFactory
from src.embeddings import EmbeddingRegistry
from src.retrievers import RetrieverFactory
from src.agents import SMEAgent, CypherAgent
from src.config import ConfigManager
from src.config.default_configs import get_default_config

# Global variables for pipeline components
pipeline = None
config_manager = None

# Pydantic models
class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question to convert to Cypher")
    config: Optional[Dict[str, Any]] = Field(default={}, description="Optional configuration overrides")
    include_context: bool = Field(default=False, description="Include retrieved context in response")
    include_metrics: bool = Field(default=False, description="Include processing metrics in response")

class QueryResponse(BaseModel):
    cypher_query: str = Field(..., description="Generated Cypher query")
    confidence: float = Field(..., description="Confidence score (0-1)")
    processing_time: float = Field(..., description="Processing time in seconds")
    success: bool = Field(..., description="Whether the query generation was successful")
    context_used: Optional[str] = Field(None, description="Retrieved context (if requested)")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Processing metrics (if requested)")
    error: Optional[str] = Field(None, description="Error message if failed")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    pipeline_ready: bool = Field(..., description="Whether the RAG pipeline is ready")
    config: Dict[str, Any] = Field(..., description="Current configuration")

class ConfigUpdateRequest(BaseModel):
    config: Dict[str, Any] = Field(..., description="Configuration to update")

# Initialize FastAPI app
app = FastAPI(
    title="LightRAG API",
    description="Production API for LightRAG - Natural Language to Cypher Query Generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def create_lightrag_pipeline():
    """Create and initialize the LightRAG pipeline."""
    global pipeline, config_manager
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = get_default_config()
        
        print("🔄 Initializing LightRAG pipeline...")
        
        # Load documents
        data_file = os.getenv("LIGHTRAG_DATA_FILE", "documents/4_knowledge_base_sme.json")
        documents = LoaderFactory.load_documents(data_file)
        print(f"✅ Loaded {len(documents)} documents")
        
        # Create chunker
        rag_config = config.get("rag_config", {})
        chunker_config = rag_config.get("chunker", {})
        chunker = ChunkerFactory.create_chunker(chunker_config.get("strategy", "fixed_window"))
        chunks = chunker.chunk_documents(documents)
        print(f"✅ Created {len(chunks)} chunks")
        
        # Create embedder
        embedding_registry = EmbeddingRegistry()
        embedding_config = rag_config.get("embedding", {})
        embedder = embedding_registry.get_embedder(embedding_config.get("model", "text-embedding-3-small"))
        
        # Create retriever
        retriever_config = rag_config.get("retriever", {})
        retriever = RetrieverFactory.create_retriever(
            retriever_config.get("type", "vector"),
            embedding_model=embedding_config.get("model", "text-embedding-3-small")
        )
        retriever.add_chunks(chunks)
        
        # Create agents
        llm_config = rag_config.get("llm", {})
        sme_agent = SMEAgent(llm_model=llm_config.get("model", "gpt-4o"))
        cypher_agent = CypherAgent(llm_model=llm_config.get("model", "gpt-4o"))
        
        # Create pipeline class
        class LightRAGPipeline:
            def __init__(self, retriever, sme_agent, cypher_agent, config):
                self.retriever = retriever
                self.sme_agent = sme_agent
                self.cypher_agent = cypher_agent
                self.config = config
                self.stats = {
                    "total_queries": 0,
                    "successful_queries": 0,
                    "failed_queries": 0,
                    "avg_processing_time": 0
                }
            
            def process_question(self, question: str, include_context: bool = False):
                """Process a natural language question and return Cypher query."""
                start_time = time.time()
                
                try:
                    # Retrieve relevant chunks
                    retrieval_results = self.retriever.retrieve(question, top_k=5)
                    
                    # SME agent filters context
                    sme_response = self.sme_agent.process(question, retrieval_results)
                    context = sme_response.content
                    
                    # Cypher agent generates query
                    cypher_response = self.cypher_agent.process(question, retrieval_results, filtered_schema=context)
                    
                    processing_time = time.time() - start_time
                    
                    # Update stats
                    self.stats["total_queries"] += 1
                    if cypher_response.content:
                        self.stats["successful_queries"] += 1
                    else:
                        self.stats["failed_queries"] += 1
                    
                    # Update average processing time
                    if self.stats["total_queries"] == 1:
                        self.stats["avg_processing_time"] = processing_time
                    else:
                        self.stats["avg_processing_time"] = (
                            (self.stats["avg_processing_time"] * (self.stats["total_queries"] - 1) + processing_time) 
                            / self.stats["total_queries"]
                        )
                    
                    result = {
                        "question": question,
                        "retrieval_results": retrieval_results,
                        "filtered_context": context,
                        "generated_cypher": cypher_response.content,
                        "success": bool(cypher_response.content),
                        "processing_time": processing_time,
                        "confidence": 0.8  # Default confidence
                    }
                    
                    return result
                    
                except Exception as e:
                    processing_time = time.time() - start_time
                    self.stats["total_queries"] += 1
                    self.stats["failed_queries"] += 1
                    
                    raise e
        
        pipeline = LightRAGPipeline(retriever, sme_agent, cypher_agent, config)
        print("✅ LightRAG pipeline initialized successfully")
        
        return pipeline
        
    except Exception as e:
        print(f"❌ Failed to initialize LightRAG pipeline: {e}")
        raise e

@app.on_event("startup")
async def startup_event():
    """Initialize the LightRAG pipeline on startup."""
    global pipeline
    try:
        pipeline = create_lightrag_pipeline()
        print("✅ LightRAG pipeline initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize LightRAG pipeline: {e}")
        print(f"❌ Startup failed: {e}")
        # Don't raise here - let the health check handle it

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "LightRAG API is running",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global pipeline, config_manager
    
    try:
        config = get_default_config() if config_manager else {}
        
        return HealthResponse(
            status="healthy" if pipeline else "initializing",
            version="1.0.0",
            pipeline_ready=pipeline is not None,
            config=config
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            pipeline_ready=False,
            config={},
            error=str(e)
        )

@app.post("/generate-cypher", response_model=QueryResponse)
async def generate_cypher(request: QueryRequest):
    """Generate Cypher query from natural language question."""
    global pipeline
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not ready. Please try again in a moment.")
    
    try:
        start_time = time.time()
        
        # Process the question
        result = pipeline.process_question(request.question, request.include_context)
        
        # Prepare response
        response_data = {
            "cypher_query": result["generated_cypher"],
            "confidence": result["confidence"],
            "processing_time": result["processing_time"],
            "success": result["success"]
        }
        
        # Add optional fields
        if request.include_context:
            response_data["context_used"] = result["filtered_context"]
        
        if request.include_metrics:
            response_data["metrics"] = {
                "pipeline_stats": pipeline.stats,
                "retrieval_count": len(result["retrieval_results"]),
                "context_length": len(result["filtered_context"]) if result["filtered_context"] else 0
            }
        
        return QueryResponse(**response_data)
        
    except Exception as e:
        processing_time = time.time() - start_time
        return QueryResponse(
            cypher_query="",
            confidence=0.0,
            processing_time=processing_time,
            success=False,
            error=str(e)
        )

@app.get("/stats")
async def get_stats():
    """Get pipeline statistics."""
    global pipeline
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    
    return {
        "pipeline_stats": pipeline.stats,
        "config": pipeline.config
    }

@app.post("/config/update")
async def update_config(request: ConfigUpdateRequest):
    """Update pipeline configuration."""
    global config_manager
    
    if not config_manager:
        raise HTTPException(status_code=503, detail="Configuration manager not ready")
    
    try:
        # Update configuration
        config_manager.update_config(request.config)
        
        # Reinitialize pipeline with new config
        global pipeline
        pipeline = create_lightrag_pipeline()
        
        return {"message": "Configuration updated successfully", "config": request.config}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")

@app.get("/config")
async def get_config():
    """Get current configuration."""
    global config_manager
    
    if not config_manager:
        raise HTTPException(status_code=503, detail="Configuration manager not ready")
    
    return config_manager.get_config()

@app.post("/batch-generate")
async def batch_generate(questions: List[str]):
    """Generate Cypher queries for multiple questions."""
    global pipeline
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    
    results = []
    
    for question in questions:
        try:
            result = pipeline.process_question(question)
            results.append({
                "question": question,
                "cypher_query": result["generated_cypher"],
                "success": result["success"],
                "processing_time": result["processing_time"]
            })
        except Exception as e:
            results.append({
                "question": question,
                "cypher_query": "",
                "success": False,
                "error": str(e)
            })
    
    return {"results": results}

if __name__ == "__main__":
    # Run with uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🚀 Starting LightRAG API on {host}:{port}")
    uvicorn.run(app, host=host, port=port) 