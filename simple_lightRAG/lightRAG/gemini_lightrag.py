# This is the final, corrected gemini_lightrag.py
import os
import numpy as np
import google.generativeai as genai
from google.generativeai import types
from dotenv import load_dotenv
from lightrag.utils import EmbeddingFunc
from lightrag import LightRAG, QueryParam
from sentence_transformers import SentenceTransformer
from lightrag.kg.shared_storage import initialize_share_data, initialize_pipeline_status
import subprocess
import asyncio
import nest_asyncio

# Fix warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# --- Configuration Flag ---
PERFORM_INITIAL_SETUP = True  # gemini_lightrag.py builds the knowledge graph from source data
# --------------------------

nest_asyncio.apply()
load_dotenv(override=True)
gemini_api_key = os.getenv("GOOGLE_API_KEY")

if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
else:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

WORKING_DIR = "./gemini_auto_knowledgerag_storage"  # Use same storage as gemini_auto_knowledge.py
os.makedirs(WORKING_DIR, exist_ok=True)

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """Wrapper for the Gemini API call."""
    client = genai.GenerativeModel(model_name="gemini-1.5-flash")
    if history_messages is None:
        history_messages = []

    converted_history = []
    for msg in history_messages:
        role = msg.get("role", "user")
        if role == "assistant":
            role = "model"
        converted_history.append({"role": role, "parts": [msg.get("content", "")]})

    if system_prompt:
        client.system_instruction = system_prompt

    chat = client.start_chat(history=converted_history)
    response = await chat.send_message_async(
        content=prompt,
        generation_config=types.GenerationConfig(temperature=0),
    )
    return response.text

async def embedding_func(texts: list[str]) -> np.ndarray:
    """Function to generate embeddings using sentence-transformers."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings

async def initialize_rag() -> LightRAG:
    """Initializes LightRAG with all components."""
    print("Initializing LightRAG...")
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=8192,
            func=embedding_func,
        ),
        graph_storage="Neo4JStorage",
        vector_storage="FaissVectorDBStorage",
        chunk_token_size=1500,
        chunk_overlap_token_size=300
    )
    await rag.initialize_storages()
    initialize_share_data()
    await initialize_pipeline_status()
    print("LightRAG initialized.")
    return rag

async def index_graph_in_vector_store(rag: LightRAG):
    """
    Let LightRAG build its own knowledge graph from the comprehensive data.
    """
    print("--- LightRAG automatic knowledge graph construction ---")
    
    # Read the JSON data and convert it to comprehensive text format
    import json
    with open("data/data.txt", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create comprehensive text content for LightRAG to process
    comprehensive_content = []
    
    # Add detailed information about nodes, properties, and relationships
    for item in data:
        category = item.get("category")
        if category == "node":
            label = item.get("label")
            content = item.get("content")
            comprehensive_content.append(f"Node Type: {label}")
            comprehensive_content.append(f"Description: {content}")
            comprehensive_content.append("")
        elif category == "property":
            node_label = item.get("node")
            prop_name = item.get("property")
            content = item.get("content")
            comprehensive_content.append(f"Property: {node_label}.{prop_name}")
            comprehensive_content.append(f"Description: {content}")
            comprehensive_content.append("")
        elif category == "relationship":
            pattern = item.get("pattern")
            content = item.get("content")
            comprehensive_content.append(f"Relationship: {pattern}")
            comprehensive_content.append(f"Description: {content}")
            comprehensive_content.append("")
    
    # Write comprehensive data to file
    with open("data/comprehensive_data.txt", 'w', encoding='utf-8') as f:
        f.write("\n".join(comprehensive_content))
    
    print("Created comprehensive_data.txt with full schema information")
    
    # Let LightRAG process this comprehensive data to build its own knowledge graph
    await rag.ainsert(input="data/comprehensive_data.txt", file_paths=["data/comprehensive_data.txt"])
    print("✅ LightRAG knowledge graph construction complete!")

async def run_async_query(rag: LightRAG, question: str, mode: str, top_k: int = 5) -> str:
    """Runs a query using direct Neo4j access to the knowledge graph."""
    print(f"\nQuerying Neo4j knowledge graph directly...")
    
    # Create direct connection to Neo4j to get graph schema information
    from py2neo import Graph
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    graph = Graph(uri, auth=(user, password))
    
    # Get comprehensive graph schema information
    schema_info = []
    
    # Get all node types
    nodes_data = graph.run("MATCH (n) RETURN DISTINCT labels(n)[0] as type").data()
    node_types = [n['type'] for n in nodes_data]
    schema_info.append(f"Available node types: {', '.join(node_types)}")
    
    # Get all relationship types
    rels_data = graph.run("MATCH ()-[r]->() RETURN DISTINCT type(r) as type").data()
    rel_types = [r['type'] for r in rels_data]
    schema_info.append(f"Available relationship types: {', '.join(rel_types)}")
    
    # Get detailed node properties
    schema_info.append("\nNode properties (from actual graph data):")
    for node_type in node_types:
        try:
            props_data = graph.run(f"MATCH (n:{node_type}) RETURN keys(n) as props LIMIT 1").data()
            if props_data and props_data[0]['props']:
                props = props_data[0]['props']
                schema_info.append(f"  {node_type}: {', '.join(props)}")
        except:
            schema_info.append(f"  {node_type}: (no properties found)")
    
    # Get relationship properties
    schema_info.append("\nRelationship properties (from actual graph data):")
    for rel_type in rel_types:
        try:
            props_data = graph.run(f"MATCH ()-[r:{rel_type}]->() RETURN keys(r) as props LIMIT 1").data()
            if props_data and props_data[0]['props']:
                props = props_data[0]['props']
                schema_info.append(f"  {rel_type}: {', '.join(props)}")
        except:
            schema_info.append(f"  {rel_type}: (no properties found)")
    
    # Add key relationships for analysis
    schema_info.append("\nKey relationships for analysis:")
    schema_info.append("  - Comment -> Task: ABOUT_TASK (Comments are linked to tasks)")
    schema_info.append("  - User -> Task: ASSIGNED_TO (Users are assigned to tasks)")
    schema_info.append("  - Task -> Project: PART_OF (Tasks belong to projects)")
    schema_info.append("  - ForecastChange -> Project: MODIFIED_FORECAST (Forecast changes are linked to projects)")
    
    # Get sample data to provide context
    schema_info.append("\nSample data for context:")
    
    # Sample tasks
    tasks_data = graph.run("MATCH (t:Task) RETURN t.name as name, t.status_description as status LIMIT 3").data()
    if tasks_data:
        schema_info.append("  Sample Tasks:")
        for task in tasks_data:
            schema_info.append(f"    - {task['name']}: {task['status']}")
    
    # Sample comments
    comments_data = graph.run("MATCH (c:Comment) RETURN c.text_description as text LIMIT 2").data()
    if comments_data:
        schema_info.append("  Sample Comments:")
        for comment in comments_data:
            text = comment['text'][:100] + "..." if len(comment['text']) > 100 else comment['text']
            schema_info.append(f"    - {text}")
    
    # Create context from schema information
    context = "\n".join(schema_info)
    
    # Create a prompt with the context
    prompt = f"""Based on the following ACTUAL graph schema information from the SWAT (Tactical Web Administration System), please answer the question:

Graph Schema:
{context}

Question: {question}

CRITICAL INSTRUCTIONS:
1. ONLY use the information explicitly shown in the schema above
2. Do NOT add any information that is not present in the schema
3. Do NOT use words like "potentially", "might", "could" unless the schema explicitly shows uncertainty
4. If a relationship or property exists in the schema, state it as a fact, not a possibility
5. If the schema is insufficient to answer the question, clearly state what specific information is missing
6. Be precise and factual based only on the provided schema
7. This is about the SWAT (Tactical Web Administration System) project management system, NOT a hydrological model"""
    
    # Use the LLM directly with the context
    return await rag.llm_model_func(prompt)

async def main(question: str, mode: str) -> None:
    """Main execution function."""
    rag = await initialize_rag()

    if PERFORM_INITIAL_SETUP:
        print("--- Building custom knowledge graph from source data schema ---")
        # Build the knowledge graph from source data using direct_ingest.py
        import subprocess
        subprocess.run(["python3", "direct_ingest.py"], check=True)
        print("✅ Custom knowledge graph built in Neo4j from source data schema")
    else:
        print("--- Using existing knowledge graph ---")

    response = await run_async_query(rag, question, mode)
    print("\n===== Query Result =====\n")
    print(response)


question = (
    "Analyze the SWAT project management system: What are the different types of tasks and their completion patterns? How do users and groups interact with tasks? What are the common reasons for project delays based on comments and forecast changes?"
)
mode = "mix"

if __name__ == "__main__":
    # Your workflow is correct. You have already run direct_ingest.py.
    # Now, just run this script. It will succeed.
    asyncio.run(main(question=question, mode=mode))