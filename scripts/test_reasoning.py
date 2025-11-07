#!/usr/bin/env python3
"""
Test reasoning system (Days 21-23)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import AERDataLoader, Document
from src.models.llm_wrapper import create_llm
from src.reasoning.abductive import AbductiveReasoner, ChainOfThoughtReasoner, FewShotReasoner
from src.reasoning.evidence_scorer import EvidenceScorer
from src.reasoning.answer_extractor import AnswerExtractor
from src.reasoning.confidence import ConfidenceEstimator
from src.retrieval.embedder import SentenceEmbedder
from src.utils.logger import setup_logger

logger = setup_logger("test_reasoning", log_dir="outputs/logs")


def test_answer_extractor():
    """Test answer extraction"""
    print("\n" + "=" * 70)
    print("Testing Answer Extractor")
    print("=" * 70)
    
    try:
        extractor = AnswerExtractor()
        
        test_cases = [
            ("Answer: A", "A"),
            ("The answer is B,C", "B,C"),
            ("Based on the evidence, A and D are correct.", "A,D"),
            ("Option C", "C"),
            ("I think it's A", "A"),
            ("This is unclear, probably D", "D"),
            ("Random text without answer", "D"),  # Should default to D
        ]
        
        print("\nTesting extraction:")
        all_passed = True
        for text, expected in test_cases:
            extracted = extractor.extract(text)
            passed = extracted == expected
            status = "✓" if passed else "✗"
            print(f"  {status} '{text[:40]}...' -> '{extracted}' (expected: '{expected}')")
            if not passed:
                all_passed = False
        
        if all_passed:
            print("\n✅ Answer Extractor test PASSED")
            return True
        else:
            print("\n❌ Some tests FAILED")
            return False
    
    except Exception as e:
        print(f"\n❌ Answer Extractor test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evidence_scorer():
    """Test evidence scoring"""
    print("\n" + "=" * 70)
    print("Testing Evidence Scorer")
    print("=" * 70)
    
    try:
        print("\n1. Initializing embedder...")
        embedder = SentenceEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        print("\n2. Creating evidence scorer...")
        scorer = EvidenceScorer(embedder)
        
        # Test documents
        docs = [
            Document(title="COVID-19", content="The COVID-19 pandemic caused global lockdowns and travel restrictions."),
            Document(title="Brexit", content="The United Kingdom held a referendum on EU membership in 2016."),
            Document(title="Climate", content="Climate change is affecting weather patterns worldwide.")
        ]
        
        query = "Why did countries close borders?"
        
        print(f"\n3. Scoring documents for query: '{query}'")
        doc_scores = scorer.score_documents(query, docs)
        
        print("\n   Results:")
        for doc, score in doc_scores:
            print(f"   - {doc.title}: {score:.4f}")
        
        # Check that COVID doc has highest score
        assert doc_scores[0][0].title == "COVID-19", "COVID-19 doc should be most relevant"
        
        print("\n✅ Evidence Scorer test PASSED")
        return True
    
    except Exception as e:
        print(f"\n❌ Evidence Scorer test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_abductive_reasoner():
    """Test abductive reasoner"""
    print("\n" + "=" * 70)
    print("Testing Abductive Reasoner")
    print("=" * 70)
    
    try:
        # Load sample data
        print("\n1. Loading sample data...")
        loader = AERDataLoader('data/sample')
        instances = loader.load()
        
        if len(instances) == 0:
            print("   ⚠️  No sample data found. Skipping test.")
            return True
        
        instance = instances[0]
        print(f"   Instance: {instance.target_event[:50]}...")
        
        # Create model and reasoner
        print("\n2. Creating reasoner...")
        model = create_llm('mock', provider='mock')
        reasoner = AbductiveReasoner(model)
        
        # Test reasoning
        print("\n3. Performing reasoning...")
        result = reasoner.reason(instance, return_reasoning=True)
        
        print(f"   Prediction: {result['prediction']}")
        print(f"   Reasoning: {result['reasoning'][:100]}...")
        
        # Validate result format
        assert 'prediction' in result
        assert 'uuid' in result
        assert result['uuid'] == instance.uuid
        
        print("\n✅ Abductive Reasoner test PASSED")
        return True
    
    except Exception as e:
        print(f"\n❌ Abductive Reasoner test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_confidence_estimator():
    """Test confidence estimation"""
    print("\n" + "=" * 70)
    print("Testing Confidence Estimator")
    print("=" * 70)
    
    try:
        # Load sample instance
        loader = AERDataLoader('data/sample')
        instances = loader.load()
        
        if len(instances) == 0:
            print("   ⚠️  No sample data found. Skipping test.")
            return True
        
        instance = instances[0]
        
        # Create estimator
        estimator = ConfidenceEstimator()
        
        # Test estimation
        reasoning = "Option A is the most plausible cause based on the evidence."
        prediction = "A"
        
        confidence = estimator.estimate(instance, reasoning, prediction)
        
        print("\nConfidence scores:")
        for opt, score in confidence.items():
            print(f"  {opt}: {score:.4f}")
        
        # Validate
        assert sum(confidence.values()) - 1.0 < 0.01, "Confidence scores should sum to ~1.0"
        assert confidence['A'] > confidence['B'], "Predicted option should have highest confidence"
        
        print("\n✅ Confidence Estimator test PASSED")
        return True
    
    except Exception as e:
        print(f"\n❌ Confidence Estimator test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print("Reasoning System Tests (Days 21-23)")
    print("=" * 70)
    
    results = {
        'Answer Extractor': test_answer_extractor(),
        'Evidence Scorer': test_evidence_scorer(),
        'Abductive Reasoner': test_abductive_reasoner(),
        'Confidence Estimator': test_confidence_estimator()
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
        print("\n🎉 Days 21-23 implementation is complete!")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        print("=" * 70)
        return 1


if __name__ == '__main__':
    sys.exit(main())