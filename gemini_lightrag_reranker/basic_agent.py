import os
import json
import shutil
import asyncio
# import nest_asyncio
import numpy as np
from functools import lru_cache

from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, CrossEncoder
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

# Apply nest_asyncio to solve event loop issues in environments like Jupyter notebooks
# nest_asyncio.apply()

# --- 1. SETUP AND CONFIGURATION ---

# Load environment variables from .env file
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

# Configure the Gemini client ONCE at the start of the script.
genai.configure(api_key=gemini_api_key)

# Set up the working directory for LightRAG's data
WORKING_DIR = "./swat_knowledge_graph_from_file"
if os.path.exists(WORKING_DIR):
    print(f"Removing existing working directory: {WORKING_DIR}")
    shutil.rmtree(WORKING_DIR)
print(f"Creating new working directory: {WORKING_DIR}")
os.mkdir(WORKING_DIR)

# Define the new system prompt for the analytical agent persona
ANALYTICAL_SYSTEM_PROMPT = """
You are an expert data analyst with deep knowledge of a graph database for a project management system called SWAT.
Your task is to answer the user's analytical question based **ONLY** on the context provided.
Do not invent, infer, or add any information that is not present in the context.

Your response MUST be a concise document with the following structure:
1.  **Summary of Findings:** A brief, one or two-sentence summary answering the user's question.
2.  **Key Nodes:** A bulleted list of the **exact** Node labels involved in the answer (e.g., `Project`, `ForecastChange`).
3.  **Key Relationships:** A bulleted list of the **exact** Relationship patterns that connect these nodes (e.g., `(:ForecastChange)-[:LOGGED_FOR]->(:Project)`).
4.  **Key Properties:** A bulleted list of the **exact** Property names from the relevant nodes that are crucial for the analysis (e.g., `new_forecast_date`, `justification`, `change_timestamp`).

Stick to this format strictly. Use the exact names for nodes, relationships, and properties as found in the context.
"""

# --- 2. DATA TRANSFORMATION ---

def transform_json_to_custom_kg(json_file_path: str):
    """Transforms the specific JSON data structure from a file into the custom_kg format."""
    print(f"Transforming {json_file_path} to LightRAG format...")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    entities, relationships, chunks, entity_map = [], [], [], {}

    for item in data:
        category = item.get("category")
        if category == "node":
            entity_name = item.get("label")
            if entity_name not in entity_map:
                entity = {"entity_name": entity_name, "entity_type": "Node", "description": item.get("content", ""), "source_id": entity_name}
                entities.append(entity)
                entity_map[entity_name] = entity
            chunks.append({"content": item.get("content", ""), "source_id": entity_name, "source_chunk_index": len(chunks)})
        elif category == "property":
            node_name = item.get("node")
            if node_name in entity_map:
                prop_desc = f"\n- Property '{item.get('property')}': {item.get('content', '')}"
                entity_map[node_name]["description"] += prop_desc
        elif category == "relationship":
            try:
                pattern = item.get("pattern", "")
                src_label = pattern.split(")-[")[0].split(":")[1]
                tgt_label = pattern.split("->(")[1].split(":")[1].replace(")", "")
                rel_type = pattern.split("-[")[1].split("]->")[0].split(":")[1]
                relationships.append({"src_id": src_label, "tgt_id": tgt_label, "description": item.get("content", ""), "keywords": rel_type, "weight": 0.9, "source_id": f"{src_label}_{rel_type}_{tgt_label}"})
                chunks.append({"content": item.get("content", ""), "source_id": f"{src_label}_{rel_type}_{tgt_label}", "source_chunk_index": len(chunks)})
            except IndexError:
                print(f"Warning: Could not parse relationship pattern: {item.get('pattern')}")

    return {"entities": entities, "relationships": relationships, "chunks": chunks}


# --- 3. MODEL AND FUNCTION DEFINITIONS ---

async def llm_model_func(prompt, system_prompt=None, **kwargs) -> str:
    """Wrapper for the Gemini API for language model completion."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    try:
        response = await model.generate_content_async(full_prompt)
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "Error: Could not get a response from the model."

@lru_cache(maxsize=None)
def get_embedding_model():
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    return SentenceTransformer("all-MiniLM-L6-v2")

async def embedding_func(texts: list[str]) -> np.ndarray:
    """Embedding function using a cached local SentenceTransformer model."""
    model = get_embedding_model()
    return model.encode(texts, convert_to_numpy=True)

@lru_cache(maxsize=None)
def get_reranker_model():
    print("Loading reranker model (ms-marco-MiniLM-L-6-v2)...")
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

async def local_rerank_func(query: str, documents: list, top_n: int = 5, **kwargs):
    """Reranks documents using a cached local CrossEncoder model."""
    model = get_reranker_model()
    doc_contents = [doc.get('content', '') for doc in documents]
    pairs = [[query, doc] for doc in doc_contents]
    scores = model.predict(pairs)
    scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs[:top_n]]

async def ask_analytical_question(rag: LightRAG, query: str):
    """
    Asks an analytical question and uses a custom system prompt to
    force the LLM to return a structured document of schema components.
    """
    print(f"\n[Analytical Query] {query}")
    response = await rag.aquery(
        query=query,
        param=QueryParam(
            mode="hybrid",
            top_k=10,  # Retrieve more context for better analysis
            response_type="default",  # Allow for a longer, structured response
        ),
        system_prompt=ANALYTICAL_SYSTEM_PROMPT  # Use our custom analytical prompt
    )
    return response


# --- 4. MAIN EXECUTION BLOCK ---

async def main():
    """
    Main function to initialize RAG, insert the custom knowledge graph,
    and run both standard and analytical queries.
    """
    YOUR_JSON_FILE_PATH = "my_swat_data.json"

    if not os.path.exists(YOUR_JSON_FILE_PATH):
        print(f"FATAL ERROR: The file '{YOUR_JSON_FILE_PATH}' was not found.")
        return

    print("\n--- Starting LightRAG with Gemini Demo ---")

    print("Initializing LightRAG with custom functions...")
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(embedding_dim=384, max_token_size=8192, func=embedding_func),
        rerank_model_func=local_rerank_func,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    print("LightRAG initialized successfully.")

    custom_kg_data = transform_json_to_custom_kg(YOUR_JSON_FILE_PATH)
    print("Inserting custom knowledge graph into LightRAG...")
    await rag.ainsert_custom_kg(custom_kg_data)
    print("Insertion complete.")

    # --- Running Standard Queries (for comparison) ---
    print("\n--- Running Standard Conversational Query ---")
    standard_query = "In simple terms, what is the relationship between a Project and a Template?"
    standard_response = await rag.aquery(
        query=standard_query,
        param=QueryParam(mode="mix", top_k=3, response_type="default"),
    )
    print(f"Standard Response:\n{standard_response}")

    # --- Running Analytical Queries ---
    print("\n--- Running Analytical Schema Queries ---")

    # Example 1: Find projects that are delayed or at risk.
    analytical_q1 = "How can I find all projects that are cancelled or have a justification indicating a delay?"
    analytical_a1 = await ask_analytical_question(rag, analytical_q1)
    print(f"Analytical Response:\n{analytical_a1}")

    # Example 2: Find out who is working on what.
    analytical_q2 = "What components are needed to see which users are assigned to tasks for a specific project?"
    analytical_a2 = await ask_analytical_question(rag, analytical_q2)
    print(f"Analytical Response:\n{analytical_a2}")

    # Example 3: Track project timeline history.
    analytical_q3 = "I need to build a complete timeline for a project, including original plans, changes, and actual milestone dates. What do I need?"
    analytical_a3 = await ask_analytical_question(rag, analytical_q3)
    print(f"Analytical Response:\n{analytical_a3}")

    # Example 4: Complex question.
    analytical_q4 = "what are all the relationships between Task and User?"
    analytical_q4 = await ask_analytical_question(rag, analytical_q4)
    print(f"Analytical Response:\n{analytical_q4}")

    print("\n--- Demo Finished ---")


if __name__ == "__main__":
    asyncio.run(main())