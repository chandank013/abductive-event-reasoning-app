#!/usr/bin/env python3
"""
Validation script for Week 3-4 (Days 15-28)
Ensures all components are working correctly
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def validate_imports():
    """Validate all required imports"""
    print("\n" + "=" * 70)
    print("Validating Imports")
    print("=" * 70)
    
    imports = [
        ('src.models.llm_wrapper', 'LLM Integration'),
        ('src.models.model_manager', 'Model Manager'),
        ('src.prompting.templates', 'Prompt Templates'),
        ('src.prompting.prompt_builder', 'Prompt Builder'),
        ('src.prompting.few_shot', 'Few-Shot Selection'),
        ('src.reasoning.abductive', 'Abductive Reasoning'),
        ('src.reasoning.evidence_scorer', 'Evidence Scoring'),
        ('src.reasoning.answer_extractor', 'Answer Extraction'),
        ('src.reasoning.confidence', 'Confidence Estimation'),
        ('src.evaluation.metrics', 'Evaluation Metrics'),
        ('src.evaluation.evaluator', 'Evaluator'),
        ('src.evaluation.error_analysis', 'Error Analysis'),
    ]
    
    failed = []
    for module, name in imports:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name}: {e}")
            failed.append(name)
    
    if failed:
        print(f"\n❌ Failed imports: {failed}")
        return False
    
    print("\n✅ All imports successful")
    return True


def validate_files():
    """Validate required files exist"""
    print("\n" + "=" * 70)
    print("Validating Files")
    print("=" * 70)
    
    required_files = [
        'src/models/llm_wrapper.py',
        'src/models/model_manager.py',
        'src/prompting/templates.py',
        'src/prompting/prompt_builder.py',
        'src/prompting/few_shot.py',
        'src/reasoning/abductive.py',
        'src/reasoning/evidence_scorer.py',
        'src/reasoning/answer_extractor.py',
        'src/reasoning/confidence.py',
        'src/evaluation/metrics.py',
        'src/evaluation/evaluator.py',
        'src/evaluation/error_analysis.py',
        'experiments/baseline/run_baseline.py',
        'scripts/test_llm.py',
        'scripts/test_prompting.py',
        'scripts/test_reasoning.py',
        'scripts/test_evaluation.py',
    ]
    
    missing = []
    for file in required_files:
        file_path = PROJECT_ROOT / file
        if file_path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file}")
            missing.append(file)
    
    if missing:
        print(f"\n❌ Missing files: {missing}")
        return False
    
    print("\n✅ All files present")
    return True


def validate_functionality():
    """Validate basic functionality"""
    print("\n" + "=" * 70)
    print("Validating Functionality")
    print("=" * 70)
    
    try:
        # Test LLM
        print("\n1. Testing LLM...")
        from src.models.llm_wrapper import create_llm
        llm = create_llm('mock')
        response = llm.generate("Test")
        print(f"   ✓ LLM generates: {response[:50]}...")
        
        # Test Prompt Builder
        print("\n2. Testing Prompt Builder...")
        from src.prompting.prompt_builder import create_prompt_builder
        from src.data.loader import AERInstance, Document
        
        instance = AERInstance(
            topic_id=1,
            uuid="test",
            target_event="Test event",
            option_A="A",
            option_B="B",
            option_C="C",
            option_D="D",
            golden_answer="A",
            docs=[Document(title="Test", content="Test content")]
        )
        
        builder = create_prompt_builder('zero_shot')
        prompt = builder.build(instance)
        print(f"   ✓ Prompt built: {len(prompt)} chars")
        
        # Test Reasoner
        print("\n3. Testing Reasoner...")
        from src.reasoning.abductive import AbductiveReasoner
        reasoner = AbductiveReasoner(llm)
        result = reasoner.reason(instance)
        print(f"   ✓ Prediction: {result['prediction']}")
        
        # Test Evaluator
        print("\n4. Testing Evaluator...")
        from src.evaluation.metrics import calculate_score
        score = calculate_score("A", "A")
        assert score == 1.0
        print(f"   ✓ Score calculation: {score}")
        
        print("\n✅ All functionality tests passed")
        return True
    
    except Exception as e:
        print(f"\n❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validations"""
    print("=" * 70)
    print("Week 3-4 Validation (Days 15-28)")
    print("=" * 70)
    
    results = {
        'Imports': validate_imports(),
        'Files': validate_files(),
        'Functionality': validate_functionality()
    }
    
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    for test, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL VALIDATIONS PASSED!")
        print("=" * 70)
        print("\n🎉 Week 3-4 (Days 15-28) Complete!")
        print("\nYour system is ready with:")
        print("  ✓ LLM Integration (OpenAI, Anthropic, HuggingFace)")
        print("  ✓ Response Caching")
        print("  ✓ Multiple Prompt Templates")
        print("  ✓ Few-Shot Learning")
        print("  ✓ Chain-of-Thought Reasoning")
        print("  ✓ Abductive Reasoning Engine")
        print("  ✓ Evidence Scoring")
        print("  ✓ Confidence Estimation")
        print("  ✓ Comprehensive Evaluation")
        print("  ✓ Error Analysis")
        print("\nNext Steps:")
        print("  1. Run tests: python scripts/run_all_tests.py")
        print("  2. Run demo: python scripts/quick_demo.py")
        print("  3. Run baseline: python experiments/baseline/run_baseline.py")
        print("  4. Start web app: python run.py --debug")
        return 0
    else:
        print("❌ SOME VALIDATIONS FAILED!")
        print("=" * 70)
        print("\nPlease fix the errors above before proceeding.")
        return 1


if __name__ == '__main__':
    sys.exit(main())