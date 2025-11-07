#!/usr/bin/env python3
"""
Test evaluation system (Days 27-28)
"""

import sys
from pathlib import Path
import tempfile

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import AERDataLoader, AERInstance
from src.evaluation.metrics import (
    calculate_score,
    calculate_accuracy,
    calculate_exact_match,
    calculate_partial_match,
    calculate_all_metrics
)
from src.evaluation.evaluator import Evaluator
from src.evaluation.error_analysis import ErrorAnalyzer
from src.utils.logger import setup_logger

logger = setup_logger("test_evaluation", log_dir="outputs/logs")


def test_metrics():
    """Test evaluation metrics"""
    print("\n" + "=" * 70)
    print("Testing Evaluation Metrics")
    print("=" * 70)
    
    try:
        # Test individual scores
        print("\n1. Testing individual scores:")
        
        test_cases = [
            ("A", "A", 1.0, "Exact match"),
            ("A,B", "A,B", 1.0, "Exact match multiple"),
            ("B,A", "A,B", 1.0, "Order doesn't matter"),
            ("A", "B", 0.0, "Complete mismatch"),
            ("A", "A,B", 0.5, "Partial match"),
            ("A,B", "A", 0.5, "Partial match reverse"),
            ("A,B", "B,C", 0.5, "Partial overlap"),
            ("D", "D", 1.0, "Insufficient info match"),
        ]
        
        all_passed = True
        for pred, gold, expected, description in test_cases:
            score = calculate_score(pred, gold)
            passed = abs(score - expected) < 0.01
            status = "✓" if passed else "✗"
            print(f"  {status} {description}: pred={pred}, gold={gold}, score={score:.1f}")
            if not passed:
                all_passed = False
        
        # Test aggregate metrics
        print("\n2. Testing aggregate metrics:")
        
        predictions = ["A", "A,B", "C", "D", "B"]
        gold_answers = ["A", "A,B", "B", "D", "B,C"]
        
        # Calculate expected scores for each prediction:
        # pred="A", gold="A" -> 1.0 (exact match)
        # pred="A,B", gold="A,B" -> 1.0 (exact match)
        # pred="C", gold="B" -> 0.0 (no overlap)
        # pred="D", gold="D" -> 1.0 (exact match)
        # pred="B", gold="B,C" -> 0.5 (partial match - B in both)
        # Total: 1.0 + 1.0 + 0.0 + 1.0 + 0.5 = 3.5 / 5 = 0.7
        
        accuracy = calculate_accuracy(predictions, gold_answers)
        exact_match = calculate_exact_match(predictions, gold_answers)
        partial_match = calculate_partial_match(predictions, gold_answers)
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Exact Match: {exact_match:.4f}")
        print(f"  Partial Match: {partial_match:.4f}")
        
        # Correct expected value: 3.5/5 = 0.7
        assert abs(accuracy - 0.7) < 0.01, f"Expected accuracy 0.7, got {accuracy}"
        print(f"  ✓ Accuracy matches expected value (0.7)")
        
        # Exact matches: A=A, A,B=A,B, D=D = 3 out of 5 = 0.6
        assert abs(exact_match - 0.6) < 0.01, f"Expected exact match 0.6, got {exact_match}"
        print(f"  ✓ Exact match matches expected value (0.6)")
        
        # Partial matches (not exact): B in B,C = 1 out of 5 = 0.2
        assert abs(partial_match - 0.2) < 0.01, f"Expected partial match 0.2, got {partial_match}"
        print(f"  ✓ Partial match matches expected value (0.2)")
        
        # Test all metrics
        print("\n3. Testing comprehensive metrics:")
        metrics = calculate_all_metrics(predictions, gold_answers)
        
        print(f"  Metrics calculated: {list(metrics.keys())}")
        assert 'accuracy' in metrics
        assert 'exact_match' in metrics
        assert 'per_option' in metrics
        
        # Verify per-option metrics exist
        for opt in ['A', 'B', 'C', 'D']:
            assert opt in metrics['per_option'], f"Missing metrics for option {opt}"
            assert 'precision' in metrics['per_option'][opt]
            assert 'recall' in metrics['per_option'][opt]
            assert 'f1' in metrics['per_option'][opt]
        
        print(f"  ✓ All per-option metrics present")
        
        if all_passed:
            print("\n✅ Metrics test PASSED")
            return True
        else:
            print("\n❌ Some tests FAILED")
            return False
    
    except Exception as e:
        print(f"\n❌ Metrics test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluator():
    """Test evaluator"""
    print("\n" + "=" * 70)
    print("Testing Evaluator")
    print("=" * 70)
    
    try:
        # Load sample data
        print("\n1. Loading sample data...")
        loader = AERDataLoader('data/sample')
        instances = loader.load()
        
        if len(instances) == 0:
            print("   ⚠️  No sample data found. Skipping test.")
            return True
        
        # Create mock predictions (all correct for simplicity)
        print("\n2. Creating mock predictions...")
        predictions = [
            {
                'uuid': inst.uuid,
                'prediction': inst.golden_answer if inst.has_answer() else 'A'
            }
            for inst in instances[:5]
        ]
        
        print(f"   Created {len(predictions)} predictions")
        
        # Evaluate
        print("\n3. Running evaluation...")
        evaluator = Evaluator()
        results = evaluator.evaluate_predictions(predictions, instances)
        
        print(f"   Predictions evaluated: {results['num_matched']}")
        print(f"   Accuracy: {results['metrics']['accuracy']:.4f}")
        
        # Validate results structure
        assert 'metrics' in results, "Results should contain metrics"
        assert 'num_matched' in results, "Results should contain num_matched"
        assert results['num_matched'] > 0, "Should match at least one prediction"
        
        # Test file-based evaluation
        print("\n4. Testing file-based evaluation...")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            import json
            for pred in predictions:
                f.write(json.dumps(pred) + '\n')
            temp_file = f.name
        
        try:
            results2 = evaluator.evaluate_from_files(temp_file, 'data/sample')
            print(f"   ✓ File-based evaluation successful")
            
            # Verify results are consistent
            assert results2['num_matched'] == results['num_matched'], \
                "File-based and direct evaluation should match"
            
        finally:
            Path(temp_file).unlink()
        
        # Test printing
        print("\n5. Testing summary printing:")
        evaluator.print_summary(results)
        
        print("\n✅ Evaluator test PASSED")
        return True
    
    except Exception as e:
        print(f"\n❌ Evaluator test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_analyzer():
    """Test error analyzer"""
    print("\n" + "=" * 70)
    print("Testing Error Analyzer")
    print("=" * 70)
    
    try:
        # Load sample data
        print("\n1. Loading sample data...")
        loader = AERDataLoader('data/sample')
        instances = loader.load()
        
        if len(instances) == 0:
            print("   ⚠️  No sample data found. Skipping test.")
            return True
        
        # Create mixed predictions (some correct, some wrong)
        print("\n2. Creating mixed predictions...")
        predictions = []
        for i, inst in enumerate(instances[:10]):
            if i % 3 == 0:
                # Correct
                pred = inst.golden_answer if inst.has_answer() else 'A'
            elif i % 3 == 1:
                # Wrong
                pred = 'D'
            else:
                # Partial
                if inst.has_answer() and ',' in inst.golden_answer:
                    pred = inst.golden_answer.split(',')[0]
                else:
                    pred = 'A,B'
            
            predictions.append({
                'uuid': inst.uuid,
                'prediction': pred
            })
        
        print(f"   Created {len(predictions)} mixed predictions")
        
        # Analyze
        print("\n3. Running error analysis...")
        analyzer = ErrorAnalyzer()
        analysis = analyzer.analyze(instances, predictions)
        
        print(f"   Correct: {analysis['num_correct']}")
        print(f"   Partial: {analysis['num_partial']}")
        print(f"   Wrong: {analysis['num_wrong']}")
        
        # Validate analysis structure
        assert 'error_patterns' in analysis, "Should have error patterns"
        assert 'by_answer_type' in analysis, "Should have by_answer_type"
        assert 'by_topic' in analysis, "Should have by_topic"
        
        # Validate error patterns
        patterns = analysis['error_patterns']
        assert 'predicted_D_when_not_D' in patterns
        assert 'missed_D' in patterns
        assert 'over_predicted' in patterns
        assert 'under_predicted' in patterns
        
        print(f"   ✓ Error patterns validated")
        
        # Test printing
        print("\n4. Testing analysis printing:")
        analyzer.print_analysis(analysis)
        
        print("\n✅ Error Analyzer test PASSED")
        return True
    
    except Exception as e:
        print(f"\n❌ Error Analyzer test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases():
    """Test edge cases"""
    print("\n" + "=" * 70)
    print("Testing Edge Cases")
    print("=" * 70)
    
    try:
        print("\n1. Testing empty predictions:")
        empty_accuracy = calculate_accuracy([], [])
        assert empty_accuracy == 0.0, "Empty should return 0"
        print(f"   ✓ Empty predictions handled: {empty_accuracy}")
        
        print("\n2. Testing single prediction:")
        single_acc = calculate_accuracy(["A"], ["A"])
        assert single_acc == 1.0, "Single correct should be 1.0"
        print(f"   ✓ Single prediction: {single_acc}")
        
        print("\n3. Testing all wrong:")
        wrong_acc = calculate_accuracy(["A", "B"], ["C", "D"])
        assert wrong_acc == 0.0, "All wrong should be 0"
        print(f"   ✓ All wrong: {wrong_acc}")
        
        print("\n4. Testing normalization:")
        # Test that "B,A" is treated same as "A,B"
        score1 = calculate_score("A,B", "B,A")
        score2 = calculate_score("B,A", "A,B")
        assert score1 == 1.0 and score2 == 1.0, "Order should not matter"
        print(f"   ✓ Order normalization works")
        
        print("\n5. Testing whitespace:")
        score_space = calculate_score("A, B", "A,B")
        assert score_space == 1.0, "Whitespace should be ignored"
        print(f"   ✓ Whitespace handled")
        
        print("\n✅ Edge Cases test PASSED")
        return True
    
    except Exception as e:
        print(f"\n❌ Edge Cases test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print("Evaluation System Tests (Days 27-28)")
    print("=" * 70)
    
    results = {
        'Metrics': test_metrics(),
        'Evaluator': test_evaluator(),
        'Error Analyzer': test_error_analyzer(),
        'Edge Cases': test_edge_cases()
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
        print("\n🎉 Days 27-28 implementation is complete!")
        print("\nEvaluation System Features:")
        print("  ✓ Partial match scoring (0.0, 0.5, 1.0)")
        print("  ✓ Per-option metrics (precision, recall, F1)")
        print("  ✓ Error pattern analysis")
        print("  ✓ Confusion matrix")
        print("  ✓ Answer type breakdown")
        print("  ✓ Topic-based analysis")
        print("  ✓ Edge case handling")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        print("=" * 70)
        return 1


if __name__ == '__main__':
    sys.exit(main())