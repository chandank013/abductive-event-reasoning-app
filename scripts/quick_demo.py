#!/usr/bin/env python3
"""
Quick demo of the complete system
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import AERDataLoader, AERInstance, Document
from src.models.llm_wrapper import create_llm
from src.prompting.prompt_builder import create_prompt_builder
from src.reasoning.abductive import AbductiveReasoner
from src.evaluation.metrics import calculate_score
from src.utils.logger import setup_logger

logger = setup_logger("demo", log_dir="outputs/logs")


def create_demo_instance():
    """Create a demo instance"""
    return AERInstance(
        topic_id=1,
        uuid="demo-001",
        target_event="Iran issued an intercity travel ban and closed schools nationwide.",
        option_A="The United States mandated port closures",
        option_B="COVID-19 pandemic forced lockdowns",
        option_C="COVID-19 caused health restrictions",
        option_D="A new virus was identified in China",
        golden_answer="B,C",
        docs=[
            Document(
                title="COVID-19 Pandemic Response",
                content="In early 2020, countries worldwide implemented lockdowns and travel restrictions in response to the COVID-19 pandemic. Iran was among the first nations in the Middle East to report cases and took swift action by closing schools and restricting intercity travel to prevent virus spread."
            ),
            Document(
                title="Global Health Emergency",
                content="The World Health Organization declared COVID-19 a pandemic in March 2020, prompting nations to implement emergency health measures including travel bans, school closures, and quarantine protocols."
            ),
            Document(
                title="Iran's Response",
                content="Iranian authorities implemented strict measures to combat COVID-19, including closing educational institutions, restricting domestic travel, and enforcing social distancing regulations."
            )
        ],
        topic="COVID-19 Response"
    )


def demo_zero_shot():
    """Demo zero-shot reasoning"""
    print("\n" + "=" * 70)
    print("Demo 1: Zero-Shot Reasoning")
    print("=" * 70)
    
    # Create instance
    instance = create_demo_instance()
    
    print("\n📋 Question:")
    print(f"Event: {instance.target_event}")
    print("\nOptions:")
    for opt, text in instance.get_options_dict().items():
        print(f"  {opt}) {text}")
    
    print(f"\n✅ Gold Answer: {instance.golden_answer}")
    
    # Create model and reasoner
    print("\n🤖 Creating model and reasoner...")
    model = create_llm('mock')
    prompt_builder = create_prompt_builder('zero_shot')
    reasoner = AbductiveReasoner(model, prompt_builder)
    
    # Reason
    print("\n🧠 Reasoning...")
    result = reasoner.reason(instance, return_reasoning=True)
    
    print(f"\n🎯 Prediction: {result['prediction']}")
    print(f"\n📝 Reasoning:")
    print(f"{result['reasoning'][:300]}...")
    
    # Evaluate
    score = calculate_score(result['prediction'], instance.golden_answer)
    print(f"\n📊 Score: {score:.1f}")
    
    if score == 1.0:
        print("✅ Perfect match!")
    elif score == 0.5:
        print("⚠️  Partial match")
    else:
        print("❌ Incorrect")


def demo_with_confidence():
    """Demo reasoning with confidence scores"""
    print("\n" + "=" * 70)
    print("Demo 2: Reasoning with Confidence Scores")
    print("=" * 70)
    
    instance = create_demo_instance()
    
    # Create reasoner
    model = create_llm('mock')
    reasoner = AbductiveReasoner(model)
    
    # Reason with confidence
    print("\n🧠 Reasoning with confidence estimation...")
    result = reasoner.reason_with_confidence(instance)
    
    print(f"\n🎯 Prediction: {result['prediction']}")
    print(f"\n📊 Confidence Scores:")
    for opt, conf in sorted(result['confidence'].items()):
        bar = "█" * int(conf * 50)
        print(f"  {opt}: {conf:.3f} {bar}")
    
    score = calculate_score(result['prediction'], instance.golden_answer)
    print(f"\n📈 Final Score: {score:.1f}")


def demo_batch_processing():
    """Demo batch processing"""
    print("\n" + "=" * 70)
    print("Demo 3: Batch Processing")
    print("=" * 70)
    
    # Load sample data
    print("\n📥 Loading sample data...")
    try:
        loader = AERDataLoader('data/sample')
        instances = loader.load()
        
        if len(instances) == 0:
            print("   ⚠️  No sample data found. Skipping demo.")
            return
        
        # Take first 3 instances
        instances = instances[:3]
        print(f"   Loaded {len(instances)} instances")
        
        # Process batch
        print("\n🧠 Processing batch...")
        model = create_llm('mock')
        reasoner = AbductiveReasoner(model)
        
        results = reasoner.reason_batch(instances, show_progress=True)
        
        # Show results
        print("\n📊 Results:")
        correct = 0
        for inst, result in zip(instances, results):
            score = calculate_score(result['prediction'], inst.golden_answer) if inst.has_answer() else 0
            if score == 1.0:
                correct += 1
            status = "✅" if score == 1.0 else "⚠️" if score == 0.5 else "❌"
            print(f"  {status} {inst.uuid}: pred={result['prediction']}, gold={inst.golden_answer}")
        
        print(f"\n✅ Correct: {correct}/{len(instances)}")
    
    except Exception as e:
        print(f"   ⚠️  Error: {e}")


def demo_evaluation():
    """Demo evaluation system"""
    print("\n" + "=" * 70)
    print("Demo 4: Evaluation System")
    print("=" * 70)
    
    try:
        from src.evaluation.evaluator import Evaluator
        
        # Load sample data
        print("\n📥 Loading sample data...")
        loader = AERDataLoader('data/sample')
        instances = loader.load()
        
        if len(instances) == 0:
            print("   ⚠️  No sample data found. Skipping demo.")
            return
        
        # Create predictions
        print("\n🧠 Generating predictions...")
        model = create_llm('mock')
        reasoner = AbductiveReasoner(model)
        
        predictions = reasoner.reason_batch(instances[:5], show_progress=False)
        
        # Evaluate
        print("\n📊 Evaluating predictions...")
        evaluator = Evaluator()
        results = evaluator.evaluate_predictions(predictions, instances)
        
        # Print summary
        evaluator.print_summary(results)
    
    except Exception as e:
        print(f"   ⚠️  Error: {e}")


def main():
    """Run all demos"""
    print("=" * 70)
    print("AER System - Complete Demo")
    print("=" * 70)
    print("\nThis demo showcases the complete system functionality:")
    print("  1. Zero-shot reasoning")
    print("  2. Confidence estimation")
    print("  3. Batch processing")
    print("  4. Evaluation system")
    
    try:
        demo_zero_shot()
        demo_with_confidence()
        demo_batch_processing()
        demo_evaluation()
        
        print("\n" + "=" * 70)
        print("✅ Demo Complete!")
        print("=" * 70)
        print("\nTo run experiments:")
        print("  python experiments/baseline/run_baseline.py")
        print("\nTo start web interface:")
        print("  python run.py --debug")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())