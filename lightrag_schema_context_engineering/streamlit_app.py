"""
LightRAG Streamlit UI

A simple Streamlit interface for testing the LightRAG chatbot.
"""

import streamlit as st
import requests
import json
import time
from typing import Dict, Any, List

# Configuration
API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000")

def check_api_health():
    """Check if the API is healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def generate_cypher(question: str, include_context: bool = False, include_metrics: bool = False):
    """Generate Cypher query from natural language question."""
    try:
        payload = {
            "question": question,
            "include_context": include_context,
            "include_metrics": include_metrics
        }
        
        response = requests.post(
            f"{API_BASE_URL}/generate-cypher",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"API Error: {response.status_code} - {response.text}"}
            
    except Exception as e:
        return False, {"error": str(e)}

def get_pipeline_stats():
    """Get pipeline statistics."""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception:
        return None

def get_sample_questions():
    """Get sample questions for testing."""
    return [
        "What are all the Comment nodes?",
        "Show me all User nodes",
        "Find all Project nodes",
        "How many comments are there?",
        "Find all comments made by users",
        "Show me all projects and their tasks",
        "Find all tasks assigned to users",
        "How many tasks were completed within the forecasted date in June 2025?",
        "What are the projects tied to tasks completed on time?",
        "How many projects were cancelled but still completed some tasks?",
        "Find the user who made the most comments",
        "Which project has the most tasks?",
        "Find all users who haven't made any comments",
        "Show me projects with more than 5 tasks",
        "Find tasks that are overdue",
        "Which groups have the most users?",
        "Find projects that have been delayed more than 30 days",
        "Calculate the average completion time for tasks by project",
        "Find the most active users (those with most task assignments)",
        "Show me projects with their completion percentage",
        "Find tasks that were reassigned multiple times",
        "Calculate the average forecast accuracy by project",
        "Find all comments made in the last 30 days",
        "Show me tasks due this week",
        "Find projects started in Q1 2024",
        "Show me forecast changes made in the last month",
        "Find all users who commented on tasks in projects they're not assigned to",
        "Show me projects where the project manager has made comments on tasks",
        "Find tasks that are part of milestones that are behind schedule",
        "Find all users who are both assigned to tasks and have made comments",
        "Show me projects that have both completed and pending tasks",
        "Find users who belong to multiple groups"
    ]

def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="LightRAG Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .cypher-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🤖 LightRAG Chatbot</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Natural Language to Cypher Query Generation</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Health Check
        st.subheader("API Status")
        health_status, health_data = check_api_health()
        
        if health_status:
            st.success("✅ API Connected")
            if health_data.get("pipeline_ready"):
                st.success("✅ Pipeline Ready")
            else:
                st.warning("⚠️ Pipeline Initializing")
        else:
            st.error("❌ API Disconnected")
            st.error(f"Error: {health_data.get('error', 'Unknown error')}")
        
        # Pipeline Stats
        if health_status:
            stats = get_pipeline_stats()
            if stats:
                st.subheader("📊 Pipeline Stats")
                pipeline_stats = stats.get("pipeline_stats", {})
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Queries", pipeline_stats.get("total_queries", 0))
                    st.metric("Success Rate", f"{pipeline_stats.get('successful_queries', 0)}/{pipeline_stats.get('total_queries', 1)}")
                
                with col2:
                    st.metric("Avg Time", f"{pipeline_stats.get('avg_processing_time', 0):.2f}s")
                    st.metric("Failed", pipeline_stats.get("failed_queries", 0))
        
        # Options
        st.subheader("⚙️ Options")
        include_context = st.checkbox("Include Context", value=False, help="Show retrieved context in response")
        include_metrics = st.checkbox("Include Metrics", value=False, help="Show processing metrics in response")
        
        # Sample Questions
        st.subheader("📝 Sample Questions")
        sample_questions = get_sample_questions()
        
        if st.button("Load Sample Questions"):
            st.session_state.sample_questions = sample_questions
        
        if "sample_questions" in st.session_state:
            selected_question = st.selectbox(
                "Choose a sample question:",
                [""] + st.session_state.sample_questions
            )
            if selected_question:
                st.session_state.question = selected_question
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask a Question")
        
        # Question input
        question = st.text_area(
            "Enter your natural language question:",
            value=st.session_state.get("question", ""),
            height=100,
            placeholder="e.g., How many tasks were completed within the forecasted date in June 2025?"
        )
        
        # Generate button
        if st.button("🚀 Generate Cypher Query", type="primary", use_container_width=True):
            if question.strip():
                with st.spinner("Generating Cypher query..."):
                    success, result = generate_cypher(question, include_context, include_metrics)
                    
                    if success:
                        st.session_state.last_result = result
                        st.session_state.last_question = question
                        st.success("✅ Cypher query generated successfully!")
                    else:
                        st.error(f"❌ Failed to generate query: {result.get('error', 'Unknown error')}")
            else:
                st.warning("Please enter a question.")
        
        # Display results
        if "last_result" in st.session_state and "last_question" in st.session_state:
            st.header("📋 Results")
            
            result = st.session_state.last_result
            question = st.session_state.last_question
            
            # Question
            st.subheader("Question")
            st.write(question)
            
            # Cypher Query
            st.subheader("Generated Cypher Query")
            if result.get("success"):
                st.markdown(f'<div class="cypher-box">{result.get("cypher_query", "")}</div>', unsafe_allow_html=True)
            else:
                st.error("Failed to generate Cypher query")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{result.get('confidence', 0):.2%}")
            with col2:
                st.metric("Processing Time", f"{result.get('processing_time', 0):.2f}s")
            with col3:
                st.metric("Success", "✅" if result.get("success") else "❌")
            
            # Context (if requested)
            if include_context and result.get("context_used"):
                with st.expander("🔍 Retrieved Context"):
                    st.text(result["context_used"])
            
            # Metrics (if requested)
            if include_metrics and result.get("metrics"):
                with st.expander("📊 Processing Metrics"):
                    st.json(result["metrics"])
            
            # Error (if any)
            if result.get("error"):
                st.error(f"Error: {result['error']}")
    
    with col2:
        st.header("📚 Quick Examples")
        
        examples = [
            "What are all the Comment nodes?",
            "How many users are there?",
            "Find all projects and their tasks",
            "Show me completed tasks",
            "Find overdue tasks",
            "Which project has the most tasks?",
            "Find users who haven't made comments",
            "Calculate average completion time"
        ]
        
        for example in examples:
            if st.button(example, key=f"example_{example}"):
                st.session_state.question = example
                st.rerun()
        
        st.header("🎯 Tips")
        st.markdown("""
        - Be specific in your questions
        - Use natural language
        - Ask about nodes, relationships, and properties
        - Include time-based filters when needed
        - Use aggregations for counts and calculations
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>LightRAG API - Natural Language to Cypher Query Generation</p>
            <p>Powered by FastAPI, Streamlit, and Advanced RAG Pipeline</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 