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

# Define the system prompt for the analytical agent persona
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


# --- 3. CORE MODEL AND RAG FUNCTIONS ---

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
    model = get_embedding_model()
    return model.encode(texts, convert_to_numpy=True)

@lru_cache(maxsize=None)
def get_reranker_model():
    print("Loading reranker model (ms-marco-MiniLM-L-6-v2)...")
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

async def local_rerank_func(query: str, documents: list, top_n: int = 5, **kwargs):
    model = get_reranker_model()
    doc_contents = [doc.get('content', '') for doc in documents]
    pairs = [[query, doc] for doc in doc_contents]
    scores = model.predict(pairs)
    scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs[:top_n]]

async def ask_analytical_question(rag: LightRAG, query: str):
    """Uses a custom system prompt to force the LLM to return a structured document of schema components."""
    print(f"\n[Analytical Sub-Query] {query}")
    response = await rag.aquery(
        query=query,
        param=QueryParam(mode="mix", top_k=10, response_type="default"),
        system_prompt=ANALYTICAL_SYSTEM_PROMPT
    )
    return response


# --- 4. ADVANCED AGENTIC WORKFLOW (QUERY DECOMPOSITION) ---

async def decompose_question(complex_query: str) -> list[str]:
    """Uses the LLM to break a complex question into simple, direct sub-questions."""
    print("\n--- Step 1: Decomposing complex question ---")
    decomposition_prompt = f"""
    You are a query analysis expert. Your task is to decompose the user's complex question into a series of simple, self-contained, direct questions. Each question should focus on identifying a single piece of information or relationship from a knowledge graph.

    User's Complex Question: "{complex_query}"

    Decompose this into a JSON array of strings. For example:
    ["What nodes and properties define a task's completion time?", "What nodes and properties identify the user assigned to a task?", "What nodes and properties explain the reason for a task's delay?"]

    Now, decompose the user's question. Your output must be only the JSON array.
    """
    response_text = await llm_model_func(decomposition_prompt)
    try:
        json_str = response_text.strip().replace("```json", "").replace("```", "")
        decomposed_queries = json.loads(json_str)
        print(f"Decomposed into: {decomposed_queries}")
        return decomposed_queries
    except json.JSONDecodeError:
        print(f"Failed to parse decomposed questions as JSON. Response was:\n{response_text}")
        return []

async def synthesize_answer(original_query: str, all_schema_components: str) -> str:
    """Uses the LLM to synthesize a final answer from the collected schema components."""
    print("\n--- Step 3: Synthesizing final answer ---")
    synthesis_prompt = f"""
    You are a senior data analyst. Your task is to answer the user's original, complex question.
    You must base your answer **exclusively** on the provided schema components. Do not add any information not supported by the components.
    First, provide a step-by-step analytical plan describing how you would use the components to find the answer.
    Second, provide a final summary conclusion that directly answers the user's question.

    **Original User Question:**
    {original_query}

    **Available Schema Components:**
    {all_schema_components}

    Now, generate the analysis plan and conclusion.
    """
    final_answer = await llm_model_func(synthesis_prompt)
    return final_answer


# --- 5. MAIN EXECUTION BLOCK ---

async def main():
    """
    Main function to initialize RAG and run standard, analytical, and advanced agentic queries.
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

    # --- Running Advanced Agentic Workflow for a Complex Query ---
    print("\n\n--- Running Advanced Agentic Workflow for Complex Query ---")
    complex_query = "what are all the relationships between tasks and forecasted date?"
    print(f"Complex Query: {complex_query}")

    # Step 1: Decompose the complex query
    sub_questions = await decompose_question(complex_query)

    # Step 2: Execute each sub-question to gather schema components
    all_components_text = ""
    if sub_questions:
        print("\n--- Step 2: Gathering schema components for each sub-question ---")
        for q in sub_questions:
            components = await ask_analytical_question(rag, q)
            all_components_text += f"\n\nFor the question '{q}', the required components are:\n{components}"

    # Step 3: Synthesize the final answer
    if all_components_text:
        final_answer = await synthesize_answer(complex_query, all_components_text)
        print("\n\n======================================")
        print("--- FINAL SYNTHESIZED ANALYSIS ---")
        print("======================================")
        print(final_answer)

    print("\n--- Demo Finished ---")


if __name__ == "__main__":
    asyncio.run(main())