#!/usr/bin/env python3
"""
Integration test for Days 6-14
Tests retrieval system and Flask setup
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_retrieval_system():
    """Test retrieval system"""
    print("\n" + "=" * 70)
    print("Testing Retrieval System")
    print("=" * 70)
    
    try:
        from src.retrieval.embedder import SentenceEmbedder
        from src.retrieval.vector_store import FAISSVectorStore
        from src.retrieval.retriever import DocumentRetriever
        from src.data.loader import Document
        
        # Test embedder
        print("\n1. Testing Embedder...")
        embedder = SentenceEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        test_texts = ["This is a test", "Another test sentence"]
        embeddings = embedder.embed(test_texts)
        print(f"   ✓ Generated embeddings shape: {embeddings.shape}")
        
        # Test vector store
        print("\n2. Testing Vector Store...")
        vector_store = FAISSVectorStore(
            embedding_dim=embedder.get_embedding_dim(),
            index_type="flat"
        )
        
        test_docs = [
            Document(title="Doc 1", content="COVID-19 pandemic information"),
            Document(title="Doc 2", content="Brexit and UK politics"),
            Document(title="Doc 3", content="Climate change effects")
        ]
        
        doc_texts = [doc.content for doc in test_docs]
        doc_embeddings = embedder.embed(doc_texts)
        vector_store.add(doc_embeddings, doc_texts)
        print(f"   ✓ Added {vector_store.get_size()} documents to index")
        
        # Test retrieval
        print("\n3. Testing Retrieval...")
        retriever = DocumentRetriever(embedder, vector_store)
        query = "COVID pandemic"
        results = retriever.retrieve(query, top_k=2)
        print(f"   ✓ Retrieved {len(results)} documents")
        print(f"   Top result: {results[0]['content'][:50]}...")
        
        print("\n✅ Retrieval System Tests PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Retrieval System Tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_flask_app():
    """Test Flask application"""
    print("\n" + "=" * 70)
    print("Testing Flask Application")
    print("=" * 70)
    
    try:
        from app import create_app
        
        # Create app
        print("\n1. Creating Flask app...")
        app = create_app(config_name='testing')
        print("   ✓ Flask app created")
        
        # Test client
        print("\n2. Testing routes...")
        with app.test_client() as client:
            # Test home page
            response = client.get('/')
            assert response.status_code == 200
            print("   ✓ Home page: OK")
            
            # Test API health
            response = client.get('/api/health')
            assert response.status_code == 200
            data = response.get_json()
            assert data['status'] == 'healthy'
            print("   ✓ API health: OK")
            
            # Test API info
            response = client.get('/api/info')
            assert response.status_code == 200
            print("   ✓ API info: OK")
            
            # Test dashboard
            response = client.get('/dashboard')
            assert response.status_code == 200
            print("   ✓ Dashboard: OK")
            
            # Test predict page
            response = client.get('/predict')
            assert response.status_code == 200
            print("   ✓ Predict page: OK")
        
        print("\n✅ Flask Application Tests PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Flask Application Tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_prediction():
    """Test API prediction endpoint"""
    print("\n" + "=" * 70)
    print("Testing API Prediction Endpoint")
    print("=" * 70)
    
    try:
        from app import create_app
        
        app = create_app(config_name='testing')
        
        with app.test_client() as client:
            # Test prediction
            print("\n1. Testing prediction endpoint...")
            
            test_data = {
                "target_event": "Iran issued a travel ban",
                "options": {
                    "A": "U.S. port closures",
                    "B": "COVID-19 lockdowns",
                    "C": "Economic sanctions",
                    "D": "Insufficient information"
                },
                "docs": [
                    {
                        "title": "COVID-19 Updates",
                        "content": "Countries worldwide implemented lockdowns due to COVID-19 pandemic"
                    }
                ]
            }
            
            response = client.post('/api/predict', json=test_data)
            assert response.status_code == 200
            result = response.get_json()
            
            print(f"   ✓ Prediction: {result['prediction']}")
            print(f"   ✓ Status: {result['status']}")
            
            # Test missing data
            print("\n2. Testing error handling...")
            response = client.post('/api/predict', json={})
            assert response.status_code == 400
            print("   ✓ Error handling: OK")
        
        print("\n✅ API Prediction Tests PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ API Prediction Tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print("AER System Integration Tests (Days 6-14)")
    print("=" * 70)
    
    results = {
        'Retrieval System': test_retrieval_system(),
        'Flask Application': test_flask_app(),
        'API Prediction': test_api_prediction()
    }
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\n🎉 Days 6-14 implementation is complete!")
        print("\nNext steps:")
        print("  1. Start the web application: python run.py --debug")
        print("  2. Open browser: http://localhost:5000")
        print("  3. Test the prediction interface")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        print("=" * 70)
        print("\nPlease fix the errors above before proceeding.")
        return 1


if __name__ == '__main__':
    sys.exit(main())