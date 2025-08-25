"""
Test Production Setup

Quick test to verify the production FastAPI backend and Streamlit UI work correctly.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import requests
import json
import time

def test_api_health():
    """Test API health endpoint."""
    print("🧪 Testing API Health...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Health: {data['status']}")
            print(f"   Version: {data['version']}")
            print(f"   Pipeline Ready: {data['pipeline_ready']}")
            return True
        else:
            print(f"❌ API Health Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API Health Error: {e}")
        return False

def test_query_generation():
    """Test Cypher query generation."""
    print("\n🧪 Testing Query Generation...")
    
    test_questions = [
        "What are all the Comment nodes?",
        "How many users are there?",
        "Find all projects and their tasks"
    ]
    
    for question in test_questions:
        try:
            payload = {
                "question": question,
                "include_context": False,
                "include_metrics": True
            }
            
            response = requests.post(
                "http://localhost:8000/generate-cypher",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Question: {question}")
                print(f"   Cypher: {result['cypher_query']}")
                print(f"   Success: {result['success']}")
                print(f"   Time: {result['processing_time']:.2f}s")
                print(f"   Confidence: {result['confidence']:.2%}")
            else:
                print(f"❌ Failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return True

def test_batch_generation():
    """Test batch query generation."""
    print("\n🧪 Testing Batch Generation...")
    
    questions = [
        "What are all the Comment nodes?",
        "How many users are there?",
        "Find all projects and their tasks"
    ]
    
    try:
        response = requests.post(
            "http://localhost:8000/batch-generate",
            json=questions,
            timeout=60
        )
        
        if response.status_code == 200:
            results = response.json()["results"]
            print(f"✅ Batch processed {len(results)} questions")
            
            for i, result in enumerate(results):
                print(f"   {i+1}. {result['question']}")
                print(f"      Success: {result['success']}")
                if result['success']:
                    print(f"      Cypher: {result['cypher_query']}")
        else:
            print(f"❌ Batch failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Batch error: {e}")
    
    return True

def test_config_management():
    """Test configuration management."""
    print("\n🧪 Testing Configuration Management...")
    
    try:
        # Get current config
        response = requests.get("http://localhost:8000/config", timeout=5)
        if response.status_code == 200:
            config = response.json()
            print(f"✅ Current config: {config.get('chunker', 'unknown')} chunker, {config.get('retriever', 'unknown')} retriever")
        else:
            print(f"❌ Config get failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Config error: {e}")
    
    return True

def test_stats():
    """Test statistics endpoint."""
    print("\n🧪 Testing Statistics...")
    
    try:
        response = requests.get("http://localhost:8000/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            pipeline_stats = stats.get("pipeline_stats", {})
            print(f"✅ Pipeline Stats:")
            print(f"   Total Queries: {pipeline_stats.get('total_queries', 0)}")
            print(f"   Successful: {pipeline_stats.get('successful_queries', 0)}")
            print(f"   Failed: {pipeline_stats.get('failed_queries', 0)}")
            print(f"   Avg Time: {pipeline_stats.get('avg_processing_time', 0):.2f}s")
        else:
            print(f"❌ Stats failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Stats error: {e}")
    
    return True

def main():
    """Run all tests."""
    print("🚀 Testing LightRAG Production Setup")
    print("=" * 50)
    
    # Check if API is running
    print("📡 Checking if API is running on localhost:8000...")
    
    tests = [
        test_api_health,
        test_query_generation,
        test_batch_generation,
        test_config_management,
        test_stats
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Production setup is working correctly.")
        print("\n📋 Next Steps:")
        print("   1. Deploy to GCP Cloud Run using the deployment guide")
        print("   2. Test with Streamlit UI: streamlit run streamlit_app.py")
        print("   3. Run experimental sweeps with the enhanced gold dataset")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        print("\n💡 Make sure to:")
        print("   1. Start the FastAPI server: python3 main.py")
        print("   2. Set your API keys in environment variables")
        print("   3. Check that all dependencies are installed")

if __name__ == "__main__":
    main() 