#!/usr/bin/env python3
"""
Test prompting system (Days 18-20)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import AERDataLoader, AERInstance, Document
from src.prompting.prompt_builder import PromptBuilder, create_prompt_builder
from src.prompting.few_shot import FewShotSelector
from src.prompting.templates import list_templates, get_template
from src.prompting.chain_of_thought import (
    ChainOfThoughtPrompt,
    MultiStepReasoner,
    create_cot_prompt
)
from src.utils.logger import setup_logger

logger = setup_logger("test_prompting", log_dir="outputs/logs")


def create_sample_instance() -> AERInstance:
    """Create a sample AER instance for testing"""
    return AERInstance(
        topic_id=1,
        uuid="test-001",
        target_event="Iran issued an intercity travel ban and closed schools.",
        option_A="U.S. mandated port closures",
        option_B="COVID-19 forced lockdowns",
        option_C="COVID-19 caused restrictions",
        option_D="Virus was identified in China",
        golden_answer="B,C",
        docs=[
            Document(
                title="COVID-19 Pandemic",
                content="Countries worldwide implemented lockdowns and travel restrictions in response to the COVID-19 pandemic. Iran was among the nations that closed schools and restricted movement."
            ),
            Document(
                title="Global Response",
                content="The World Health Organization declared COVID-19 a pandemic in March 2020, prompting nations to take preventive measures including travel bans."
            )
        ],
        topic="COVID-19 Response"
    )


def test_templates():
    """Test prompt templates"""
    print("\n" + "=" * 70)
    print("Testing Prompt Templates")
    print("=" * 70)
    
    try:
        # List templates
        print("\n1. Available templates:")
        templates = list_templates()
        for t in templates:
            print(f"   - {t}")
        
        assert len(templates) >= 5, "Should have at least 5 templates"
        
        # Test each template
        instance = create_sample_instance()
        
        print("\n2. Testing templates:")
        for template_name in ['zero_shot_simple', 'zero_shot_context', 'chain_of_thought']:
            print(f"\n   Template: {template_name}")
            print("   " + "-" * 60)
            
            template = get_template(template_name)
            
            # Prepare variables
            variables = {
                'target_event': instance.target_event,
                'option_A': instance.option_A,
                'option_B': instance.option_B,
                'option_C': instance.option_C,
                'option_D': instance.option_D,
                'documents': "\n".join([f"{d.title}: {d.content}" for d in instance.docs])
            }
            
            prompt = template.format(**variables)
            assert len(prompt) > 0, f"Prompt should not be empty for {template_name}"
            print(f"   Length: {len(prompt)} chars")
            print(f"   Preview: {prompt[:150]}...")
        
        print("\n✅ Template test PASSED")
        return True
    
    except Exception as e:
        print(f"\n❌ Template test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prompt_builder():
    """Test prompt builder"""
    print("\n" + "=" * 70)
    print("Testing Prompt Builder")
    print("=" * 70)
    
    try:
        instance = create_sample_instance()
        
        # Test different strategies
        print("\n1. Testing different strategies:")
        
        for strategy in ['zero_shot', 'cot', 'abductive']:
            print(f"\n   Strategy: {strategy}")
            print("   " + "-" * 60)
            
            builder = create_prompt_builder(strategy=strategy)
            prompt = builder.build(instance)
            
            assert len(prompt) > 0, f"Prompt should not be empty for {strategy}"
            print(f"   Length: {len(prompt)} chars")
            print(f"   Preview: {prompt[:200]}...")
        
        # Test batch building
        print("\n2. Testing batch building:")
        instances = [instance for _ in range(3)]
        builder = PromptBuilder()
        prompts = builder.build_batch(instances)
        
        print(f"   Generated {len(prompts)} prompts")
        assert len(prompts) == 3, "Should generate 3 prompts"
        
        # Test without documents
        print("\n3. Testing without documents:")
        prompt_no_docs = builder.build(instance, include_documents=False)
        assert len(prompt_no_docs) > 0, "Should generate prompt without docs"
        assert len(prompt_no_docs) < len(prompt), "Prompt without docs should be shorter"
        print(f"   Prompt without docs: {len(prompt_no_docs)} chars")
        
        print("\n✅ Prompt Builder test PASSED")
        return True
    
    except Exception as e:
        print(f"\n❌ Prompt Builder test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_few_shot():
    """Test few-shot selection"""
    print("\n" + "=" * 70)
    print("Testing Few-Shot Selection")
    print("=" * 70)
    
    try:
        # Create sample instances
        instances = [create_sample_instance() for _ in range(10)]
        for i, inst in enumerate(instances):
            inst.uuid = f"test-{i:03d}"
            inst.golden_answer = ['A', 'B', 'C', 'A,B'][i % 4]
        
        query_instance = create_sample_instance()
        
        # Test random selection
        print("\n1. Testing random selection:")
        selector = FewShotSelector(strategy='random')
        examples = selector.select(query_instance, instances, k=3)
        
        print(f"   Selected {len(examples)} examples")
        for ex in examples:
            print(f"   - {ex.uuid}: {ex.golden_answer}")
        
        assert len(examples) == 3, "Should select 3 examples"
        
        # Test diverse selection
        print("\n2. Testing diverse selection:")
        selector = FewShotSelector(strategy='diverse')
        examples = selector.select(query_instance, instances, k=3)
        
        print(f"   Selected {len(examples)} examples")
        answers = [ex.golden_answer for ex in examples]
        print(f"   Answers: {answers}")
        
        # Should have different answer types if possible
        unique_answers = len(set(answers))
        print(f"   Unique answer types: {unique_answers}")
        
        print("\n✅ Few-Shot test PASSED")
        return True
    
    except Exception as e:
        print(f"\n❌ Few-Shot test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chain_of_thought():
    """Test chain-of-thought prompting"""
    print("\n" + "=" * 70)
    print("Testing Chain-of-Thought Prompting")
    print("=" * 70)
    
    try:
        instance = create_sample_instance()
        
        # Test basic CoT
        print("\n1. Testing basic CoT prompt:")
        cot = ChainOfThoughtPrompt()
        prompt = cot.create_reasoning_steps(instance)
        
        assert len(prompt) > 0, "CoT prompt should not be empty"
        assert "Step 1" in prompt, "Should have step-by-step structure"
        assert "Step 2" in prompt, "Should have multiple steps"
        print(f"   ✓ Generated {len(prompt)} chars")
        print(f"   Preview: {prompt[:200]}...")
        
        # Test analytical CoT
        print("\n2. Testing analytical CoT:")
        analytical_prompt = cot.create_analytical_prompt(instance)
        
        assert "TEMPORAL REASONING" in analytical_prompt, "Should have analytical framework"
        assert "LOGICAL CONNECTION" in analytical_prompt, "Should analyze logic"
        print(f"   ✓ Generated {len(analytical_prompt)} chars")
        
        # Test comparative CoT
        print("\n3. Testing comparative CoT:")
        comparative_prompt = cot.create_comparative_prompt(instance)
        
        assert "Option A:" in comparative_prompt, "Should compare options"
        assert "Option B:" in comparative_prompt, "Should include all options"
        assert "Pros:" in comparative_prompt or "Likelihood:" in comparative_prompt, "Should analyze each option"
        print(f"   ✓ Generated {len(comparative_prompt)} chars")
        
        # Test with examples
        print("\n4. Testing CoT with examples:")
        examples = [create_sample_instance() for _ in range(2)]
        for i, ex in enumerate(examples):
            ex.uuid = f"example-{i}"
            ex.golden_answer = ['A', 'B'][i]
        
        example_prompt = cot.create_with_examples(instance, examples)
        
        assert "Example 1:" in example_prompt, "Should include examples"
        assert "Example 2:" in example_prompt, "Should have multiple examples"
        print(f"   ✓ Generated {len(example_prompt)} chars with {len(examples)} examples")
        
        # Test multi-step reasoner
        print("\n5. Testing multi-step reasoner:")
        reasoner = MultiStepReasoner()
        
        # Test problem decomposition
        sub_questions = reasoner.decompose_problem(instance)
        assert len(sub_questions) >= 5, "Should decompose into multiple sub-questions"
        print(f"   ✓ Decomposed into {len(sub_questions)} sub-questions")
        
        # Test step-by-step prompt
        step_prompt = reasoner.create_step_by_step_prompt(instance)
        assert len(step_prompt) > 0, "Step-by-step prompt should not be empty"
        print(f"   ✓ Generated step-by-step prompt: {len(step_prompt)} chars")
        
        # Test factory function
        print("\n6. Testing factory function:")
        for style in ['standard', 'analytical', 'comparative', 'step_by_step']:
            prompt = create_cot_prompt(instance, style=style)
            assert len(prompt) > 0, f"Should generate prompt for style: {style}"
            print(f"   ✓ Style '{style}': {len(prompt)} chars")
        
        # Test verification prompt
        print("\n7. Testing verification prompt:")
        verification_prompt = cot.create_verification_prompt(instance, "B,C")
        assert "VERIFY" in verification_prompt.upper(), "Should be verification prompt"
        assert "B,C" in verification_prompt, "Should include initial answer"
        print(f"   ✓ Generated verification prompt: {len(verification_prompt)} chars")
        
        print("\n✅ Chain-of-Thought test PASSED")
        return True
    
    except Exception as e:
        print(f"\n❌ Chain-of-Thought test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print("Prompting System Tests (Days 18-20)")
    print("=" * 70)
    
    results = {
        'Templates': test_templates(),
        'Prompt Builder': test_prompt_builder(),
        'Few-Shot Selection': test_few_shot(),
        'Chain-of-Thought': test_chain_of_thought()
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
        print("\n🎉 Days 18-20 implementation is complete!")
        print("\nChain-of-Thought Features:")
        print("  ✓ Basic step-by-step reasoning")
        print("  ✓ Analytical framework")
        print("  ✓ Comparative analysis")
        print("  ✓ Multi-step decomposition")
        print("  ✓ Verification prompts")
        print("  ✓ Example-based CoT")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        print("=" * 70)
        return 1


if __name__ == '__main__':
    sys.exit(main())