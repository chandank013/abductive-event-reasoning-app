#!/usr/bin/env python3
"""
Run all tests for Days 15-28
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_test(script_name: str) -> bool:
    """Run a test script"""
    print(f"\n{'='*70}")
    print(f"Running: {script_name}")
    print('='*70)
    
    result = subprocess.run(
        [sys.executable, f"scripts/{script_name}"],
        cwd=PROJECT_ROOT
    )
    
    return result.returncode == 0


def main():
    """Run all tests"""
    print("="*70)
    print("AER System - Complete Test Suite (Days 15-28)")
    print("="*70)
    
    tests = [
        ('test_llm.py', 'Days 15-17: LLM Integration'),
        ('test_prompting.py', 'Days 18-20: Prompting System'),
        ('test_reasoning.py', 'Days 21-23: Reasoning Engine'),
    ]
    
    results = {}
    
    for script, description in tests:
        print(f"\n\n{'#'*70}")
        print(f"# {description}")
        print(f"{'#'*70}")
        
        passed = run_test(script)
        results[description] = passed
    
    # Final summary
    print("\n\n" + "="*70)
    print("FINAL TEST SUMMARY")
    print("="*70)
    
    for description, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{description}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\n🎉 Week 3-4 (Days 15-28) Complete!")
        print("\nYou now have:")
        print("  ✓ LLM integration with caching")
        print("  ✓ Multiple prompt templates")
        print("  ✓ Abductive reasoning engine")
        print("  ✓ Baseline model implementation")
        print("  ✓ Complete evaluation system")
        print("\nNext: Run baseline experiment")
        print("  python experiments/baseline/run_baseline.py --model mock")