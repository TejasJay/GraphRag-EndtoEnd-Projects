import os
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv
from lightrag.utils import EmbeddingFunc
from lightrag import LightRAG, QueryParam
from sentence_transformers import SentenceTransformer
from lightrag.kg.shared_storage import initialize_share_data, initialize_pipeline_status

import asyncio
import nest_asyncio

nest_asyncio.apply()
load_dotenv(override=True)
gemini_api_key = os.getenv("GOOGLE_API_KEY")  # Make sure this matches your .env

WORKING_DIR = "./gemini_auto_knowledgerag_storage"
os.makedirs(WORKING_DIR, exist_ok=True)

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    client = genai.Client(api_key=gemini_api_key)
    if history_messages is None:
        history_messages = []
    combined_prompt = ""
    if system_prompt:
        combined_prompt += f"{system_prompt}\n"
    for msg in history_messages:
        combined_prompt += f"{msg['role']}: {msg['content']}\n"
    combined_prompt += f"user: {prompt}"
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=[combined_prompt],
        config=types.GenerateContentConfig(temperature=0),
    )
    return response.text

async def embedding_func(texts: list[str]) -> np.ndarray:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings

async def initialize_rag():
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
    return rag

async def index_data(rag: LightRAG, file_path: str) -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    await rag.ainsert(input=text, file_paths=[file_path])

async def index_file(rag: LightRAG, path: str) -> None:
    await index_data(rag, path)

async def learn_from_custom_graph():
    """Learn from the existing custom graph schema in Neo4j."""
    import os
    from py2neo import Graph
    
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    graph = Graph(uri, auth=(user, password))
    
    schema_info = {}
    
    # Get all node types and their properties
    nodes_data = graph.run("MATCH (n) RETURN DISTINCT labels(n)[0] as type").data()
    node_types = [n['type'] for n in nodes_data]
    
    schema_info['nodes'] = {}
    for node_type in node_types:
        try:
            props_data = graph.run(f"MATCH (n:{node_type}) RETURN keys(n) as props LIMIT 1").data()
            if props_data and props_data[0]['props']:
                props = props_data[0]['props']
                schema_info['nodes'][node_type] = props
        except:
            schema_info['nodes'][node_type] = []
    
    # Get all relationship types and their properties
    rels_data = graph.run("MATCH ()-[r]->() RETURN DISTINCT type(r) as type").data()
    rel_types = [r['type'] for r in rels_data]
    
    schema_info['relationships'] = {}
    for rel_type in rel_types:
        try:
            props_data = graph.run(f"MATCH ()-[r:{rel_type}]->() RETURN keys(r) as props LIMIT 1").data()
            if props_data and props_data[0]['props']:
                props = props_data[0]['props']
                schema_info['relationships'][rel_type] = props
        except:
            schema_info['relationships'][rel_type] = []
    
    return schema_info

async def build_enhanced_knowledge_graph(rag: LightRAG, custom_schema: dict, data_path: str):
    """Build enhanced knowledge graph based on custom graph understanding."""
    
    # Create enhanced text content that incorporates custom graph understanding
    enhanced_content = []
    
    # Add custom graph schema understanding
    enhanced_content.append("CUSTOM GRAPH SCHEMA UNDERSTANDING:")
    enhanced_content.append("=" * 50)
    
    enhanced_content.append("\nNODE TYPES AND PROPERTIES:")
    for node_type, props in custom_schema['nodes'].items():
        enhanced_content.append(f"- {node_type}: {', '.join(props)}")
    
    enhanced_content.append("\nRELATIONSHIP TYPES AND PROPERTIES:")
    for rel_type, props in custom_schema['relationships'].items():
        enhanced_content.append(f"- {rel_type}: {', '.join(props)}")
    
    enhanced_content.append("\n" + "=" * 50)
    enhanced_content.append("ENHANCED KNOWLEDGE BASE:")
    enhanced_content.append("=" * 50)
    
    # Add original data content
    with open(data_path, 'r', encoding='utf-8') as f:
        original_content = f.read()
    enhanced_content.append(original_content)
    
    # Write enhanced content to file
    enhanced_file = "data/enhanced_knowledge.txt"
    with open(enhanced_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(enhanced_content))
    
    # Build LightRAG knowledge graph from enhanced content
    await index_file(rag, enhanced_file)

async def run_async_query_with_expansion(rag: LightRAG, question: str, mode: str, custom_schema: dict, top_k: int = 5) -> str:
    """Runs a query with dynamic knowledge expansion capabilities."""
    print(f"\nQuerying with dynamic knowledge expansion ({mode} mode)...")
    
    # First try LightRAG's built-in query method
    try:
        result = await rag.aquery(
            question,
            param=QueryParam(mode=mode, top_k=top_k)
        )
        
        # Check if we got a meaningful response
        if "[no-context]" not in result and len(result.strip()) > 50:
            print("✅ LightRAG provided a good response using enhanced knowledge graph!")
            
            # Extract new insights and expand knowledge graph
            await expand_knowledge_from_response(question, result, custom_schema)
            
            return result
        else:
            print("⚠️ LightRAG returned no context, using direct Neo4j access...")
            
    except Exception as e:
        print(f"❌ LightRAG query failed: {e}")
        print("🔄 Using direct Neo4j access...")
    
    # Fallback: Direct Neo4j access with dynamic expansion
    print("\nUsing direct Neo4j access with dynamic expansion...")
    
    # Create direct connection to Neo4j
    import os
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
    result = await rag.llm_model_func(prompt)
    
    # Extract new insights and expand knowledge graph
    await expand_knowledge_from_response(question, result, custom_schema)
    
    return result

async def expand_knowledge_from_response(question: str, response: str, custom_schema: dict):
    """Extract new insights from Q&A and expand the knowledge graph."""
    print("\n--- Expanding knowledge graph from Q&A insights ---")
    
    # Create a prompt to extract new entities and relationships
    extraction_prompt = f"""
    Based on the following question and answer, extract any NEW entities, relationships, or properties that could be added to the knowledge graph.
    
    Question: {question}
    Answer: {response}
    
    Current Knowledge Graph Schema:
    Nodes: {list(custom_schema['nodes'].keys())}
    Relationships: {list(custom_schema['relationships'].keys())}
    
    Please identify:
    1. New entity types that could be added
    2. New relationship types that could be added
    3. New properties that could be added to existing entities
    4. Any insights that could enhance the knowledge graph
    
    Format your response as a structured analysis.
    """
    
    # Use the LLM to extract insights
    try:
        from google import genai
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        extraction_result = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[extraction_prompt],
            config=types.GenerateContentConfig(temperature=0.1),
        )
        
        print("✅ Extracted insights for knowledge graph expansion:")
        print(extraction_result.text[:200] + "...")
        
        # TODO: Implement actual Neo4j expansion based on extracted insights
        # This would involve:
        # 1. Parsing the extracted insights
        # 2. Creating new nodes/relationships in Neo4j
        # 3. Updating the knowledge graph schema
        
    except Exception as e:
        print(f"⚠️ Could not extract insights: {e}")

async def run_async_query(rag: LightRAG, question: str, mode: str, top_k: int = 5) -> str:
    """Runs a query using BOTH LightRAG's knowledge graph AND the pre-built Neo4j graph."""
    print(f"\nQuerying using BOTH knowledge graphs ({mode} mode)...")
    
    # First try LightRAG's built-in query method (using its own knowledge graph)
    try:
        result = await rag.aquery(
            question,
            param=QueryParam(mode=mode, top_k=top_k)
        )
        
        # Check if we got a meaningful response
        if "[no-context]" not in result and len(result.strip()) > 50:
            print("✅ LightRAG provided a good response using its own knowledge graph!")
            return result
        else:
            print("⚠️ LightRAG returned no context, using direct Neo4j access...")
            
    except Exception as e:
        print(f"❌ LightRAG query failed: {e}")
        print("🔄 Using direct Neo4j access...")
    
    # Fallback: Direct Neo4j access to the pre-built knowledge graph
    print("\nUsing direct Neo4j access to the pre-built knowledge graph...")
    
    # Create direct connection to Neo4j to get graph schema information
    import os
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

async def main(question: str, mode: str, data_path: str = "data/data.txt") -> None:
    rag = await initialize_rag()
    
    # Step 1: Learn from existing custom graph schema
    print("--- Step 1: Learning from existing custom graph schema ---")
    custom_graph_schema = await learn_from_custom_graph()
    print("✅ Learned from custom graph schema")
    
    # Step 2: Build enhanced knowledge graph based on custom graph understanding
    print("--- Step 2: Building enhanced knowledge graph ---")
    await build_enhanced_knowledge_graph(rag, custom_graph_schema, data_path)
    print("✅ Enhanced knowledge graph built")
    
    # Step 3: Query and dynamically expand knowledge
    print("--- Step 3: Querying with dynamic knowledge expansion ---")
    response = await run_async_query_with_expansion(rag, question, mode, custom_graph_schema)
    print("\n===== Query Result =====\n")
    print(response)

question = (
    "which exact nodes name, exact relationships name and exact properties name are required to answer this question: "
    "what are all the tasks that were completed within the forecasted time, out of these who were the one who contributed the most? "
    "Also what was the reason behind the late completions of the other tasks"
)
mode = "mix"

if __name__ == "__main__":
    asyncio.run(main(question=question, mode=mode))
