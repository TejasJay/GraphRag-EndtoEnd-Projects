# save this as direct_ingest.py
import os
import json
import re
from dotenv import load_dotenv
from py2neo import Graph

load_dotenv(override=True)

def run_direct_ingestion():
    """
    Reads the structured JSON data and directly builds a clean, accurate
    knowledge graph in Neo4j, bypassing the RAG ingestion pipeline.
    """
    print("--- Starting Direct Graph Ingestion ---")
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    graph = Graph(uri, auth=(user, password))

    # Check if database already has data
    existing_nodes = graph.run("MATCH (n) RETURN count(n) as count").data()[0]['count']
    
    if existing_nodes > 0:
        print(f"Database already contains {existing_nodes} nodes. Skipping ingestion.")
        print("✅ Using existing knowledge graph.")
        return
    
    # Only clear and rebuild if database is empty
    print("Database is empty. Building knowledge graph...")

    with open("data/data.txt", 'r', encoding='utf-8') as f:
        content = f.read()
        if content.startswith('\ufeff'):
            content = content[1:]
        data = json.loads(content)

    # --- Ingestion in 3 Stages for correctness ---

    # Stage 1: Create all the primary nodes first
    print("Stage 1: Creating all primary nodes...")
    nodes_data = [item for item in data if item.get("category") == "node"]
    tx = graph.begin()
    for item in nodes_data:
        label = item.get("label")
        description = item.get("content")
        if label:
            # MERGE is idempotent: it creates if not exists, matches if it does.
            query = f"MERGE (n:`{label}` {{name: '{label}'}}) SET n.description = $desc"
            tx.run(query, desc=description)
    graph.commit(tx)
    print(f"Created {len(nodes_data)} primary node types.")

    # Stage 2: Set all properties on the newly created nodes
    print("Stage 2: Setting properties on nodes...")
    properties_data = [item for item in data if item.get("category") == "property"]
    tx = graph.begin()
    for item in properties_data:
        node_label = item.get("node")
        prop_name = item.get("property")
        prop_desc = item.get("content")
        if node_label and prop_name:
            # Note the backticks around the property name to handle any special characters
            query = f"MATCH (n:`{node_label}`) SET n.`{prop_name}_description` = $desc"
            tx.run(query, desc=prop_desc)
    graph.commit(tx)
    print(f"Set {len(properties_data)} properties.")

    # Stage 3: Create all relationships between the nodes
    print("Stage 3: Creating relationships...")
    relationships_data = [item for item in data if item.get("category") == "relationship"]
    tx = graph.begin()
    for item in relationships_data:
        pattern = item.get("pattern")
        rel_desc = item.get("content")
        # Use regex to parse the pattern string like "(:User)-[:MEMBER_OF]->(:Group)"
        match = re.match(r"\(:(\w+)\)-\[:(\w+)\]->\(:(\w+)\)", pattern)
        if match:
            source_label, rel_type, target_label = match.groups()
            query = f"""
            MATCH (a:`{source_label}`), (b:`{target_label}`)
            MERGE (a)-[r:`{rel_type}`]->(b)
            SET r.description = $desc
            """
            tx.run(query, desc=rel_desc)
    graph.commit(tx)
    print(f"Created {len(relationships_data)} relationship types.")

    print("✅ Direct graph ingestion complete. The knowledge graph is now perfectly structured.")

if __name__ == "__main__":
    run_direct_ingestion()