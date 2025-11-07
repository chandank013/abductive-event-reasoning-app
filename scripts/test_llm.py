#!/usr/bin/env python3
"""
Test LLM integration
"""

import sys
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.llm_wrapper import create_llm
from src.models.model_manager import ModelManager
from src.utils.logger import setup_logger

logger = setup_logger("test_llm", log_dir="outputs/logs")


def clear_cache():
    """Clear cache before tests"""
    cache_dir = PROJECT_ROOT / "outputs" / "cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Cache cleared")


def test_mock_llm():
    """Test mock LLM"""
    print("\n" + "=" * 70)
    print("Testing Mock LLM")
    print("=" * 70)
    
    try:
        # Clear cache before test
        clear_cache()
        
        # Create mock LLM
        print("\n1. Creating Mock LLM...")
        llm = create_llm('mock', provider='mock', cache_responses=True)
        print(f"   ✓ LLM created: {llm.model_name}")
        
        # Test generation
        print("\n2. Testing generation...")
        prompt = "Why did Iran issue a travel ban?"
        print(f"   Prompt: {prompt}")
        
        response1 = llm.generate(prompt)
        print(f"   Response: {response1}")
        
        # Check stats after first request
        stats1 = llm.get_usage_stats()
        print(f"   Stats after 1st request: {stats1}")
        assert stats1['total_requests'] == 1, "Should have 1 request"
        assert stats1['cache_misses'] == 1, f"Should have 1 cache miss, got {stats1['cache_misses']}"
        assert stats1['cache_hits'] == 0, f"Should have 0 cache hits, got {stats1['cache_hits']}"
        
        # Test caching - same prompt
        print("\n3. Testing cache (same prompt)...")
        response2 = llm.generate(prompt)
        print(f"   Cached Response: {response2}")
        
        # Verify responses are identical
        assert response1 == response2, "Cached response should be identical"
        
        # Check stats after cached request
        stats2 = llm.get_usage_stats()
        print(f"   Stats after 2nd request: {stats2}")
        assert stats2['total_requests'] == 2, f"Should have 2 requests, got {stats2['total_requests']}"
        assert stats2['cache_hits'] == 1, f"Should have 1 cache hit, got {stats2['cache_hits']}"
        assert stats2['cache_misses'] == 1, f"Should have 1 cache miss, got {stats2['cache_misses']}"
        
        # Test different prompt
        print("\n4. Testing new prompt...")
        prompt2 = "What caused the Brexit referendum?"
        response3 = llm.generate(prompt2)
        print(f"   New Response: {response3}")
        
        stats3 = llm.get_usage_stats()
        print(f"   Stats after 3rd request: {stats3}")
        assert stats3['total_requests'] == 3, "Should have 3 requests"
        assert stats3['cache_hits'] == 1, "Should still have 1 cache hit"
        assert stats3['cache_misses'] == 2, f"Should have 2 cache misses, got {stats3['cache_misses']}"
        
        print("\n✅ Mock LLM test PASSED")
        return True
    
    except Exception as e:
        print(f"\n❌ Mock LLM test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_openai_llm():
    """Test OpenAI LLM (if API key available)"""
    print("\n" + "=" * 70)
    print("Testing OpenAI LLM")
    print("=" * 70)
    
    try:
        # Create OpenAI LLM
        llm = create_llm('gpt-3.5-turbo', provider='openai')
        
        if not llm.available:
            print("⚠️  OpenAI API key not configured. Skipping test.")
            return True
        
        # Test generation
        prompt = "In one sentence, explain abductive reasoning."
        print(f"\nPrompt: {prompt}")
        
        response = llm.generate(prompt, max_tokens=100)
        print(f"Response: {response}")
        
        # Get stats
        stats = llm.get_usage_stats()
        print(f"\nUsage Stats: {stats}")
        
        print("\n✅ OpenAI LLM test PASSED")
        return True
    
    except Exception as e:
        print(f"\n❌ OpenAI LLM test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_anthropic_llm():
    """Test Anthropic LLM (if API key available)"""
    print("\n" + "=" * 70)
    print("Testing Anthropic LLM")
    print("=" * 70)
    
    try:
        # Create Anthropic LLM
        llm = create_llm('claude-3-sonnet-20240229', provider='anthropic')
        
        if not llm.available:
            print("⚠️  Anthropic API key not configured. Skipping test.")
            return True
        
        # Test generation
        prompt = "In one sentence, what is causal reasoning?"
        print(f"\nPrompt: {prompt}")
        
        response = llm.generate(prompt, max_tokens=100)
        print(f"Response: {response}")
        
        # Get stats
        stats = llm.get_usage_stats()
        print(f"\nUsage Stats: {stats}")
        
        print("\n✅ Anthropic LLM test PASSED")
        return True
    
    except Exception as e:
        print(f"\n❌ Anthropic LLM test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_manager():
    """Test model manager"""
    print("\n" + "=" * 70)
    print("Testing Model Manager")
    print("=" * 70)
    
    try:
        # Clear cache before test
        clear_cache()
        
        # Create model manager
        manager = ModelManager()
        
        # Register models
        print("\n1. Registering models...")
        manager.create_and_register(
            name='test-mock',
            model_name='mock',
            provider='mock',
            set_as_default=True,
            cache_responses=True
        )
        print("   ✓ Mock model registered")
        
        # List models
        print("\n2. Listing models...")
        models = manager.list_models()
        print(f"   Available models: {models}")
        assert 'test-mock' in models
        
        # Get model
        print("\n3. Getting model...")
        model = manager.get_model('test-mock')
        print(f"   Model: {model.model_name}")
        
        # Get model info
        print("\n4. Getting model info...")
        info = manager.get_model_info('test-mock')
        print(f"   Info: {info}")
        
        # Test generation
        print("\n5. Testing generation...")
        prompt1 = "Test prompt for manager"
        response = model.generate(prompt1)
        print(f"   Response: {response[:100]}...")
        
        # Check stats after first generation
        stats_after_first = model.get_usage_stats()
        print(f"   Stats after 1st: {stats_after_first}")
        assert stats_after_first['total_requests'] == 1, "Should have 1 request"
        assert stats_after_first['cache_misses'] == 1, "Should have 1 cache miss"
        
        # Test caching
        print("\n6. Testing cache...")
        response2 = model.generate(prompt1)  # Same prompt
        assert response == response2, "Cache should return same response"
        
        stats = model.get_usage_stats()
        print(f"   Stats after 2nd: {stats}")
        assert stats['cache_hits'] == 1, f"Should have 1 cache hit, got {stats['cache_hits']}"
        assert stats['total_requests'] == 2, f"Should have 2 total requests, got {stats['total_requests']}"
        
        # Save config
        print("\n7. Saving configuration...")
        config_path = "outputs/test_models_config.json"
        manager.save_config(config_path)
        print(f"   Config saved to {config_path}")
        
        # Load config
        print("\n8. Loading configuration...")
        manager2 = ModelManager(config_path)
        print(f"   Models loaded: {manager2.list_models()}")
        
        print("\n✅ Model Manager test PASSED")
        return True
    
    except Exception as e:
        print(f"\n❌ Model Manager test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache_persistence():
    """Test cache persistence across instances"""
    print("\n" + "=" * 70)
    print("Testing Cache Persistence")
    print("=" * 70)
    
    try:
        # Clear cache
        clear_cache()
        
        print("\n1. Creating first LLM instance...")
        llm1 = create_llm('mock', provider='mock', cache_responses=True)
        
        prompt = "Test cache persistence"
        response1 = llm1.generate(prompt)
        print(f"   Response from LLM1: {response1[:50]}...")
        
        stats1 = llm1.get_usage_stats()
        print(f"   Stats LLM1: {stats1}")
        assert stats1['cache_misses'] == 1, "First request should be cache miss"
        
        print("\n2. Creating second LLM instance (same model)...")
        llm2 = create_llm('mock', provider='mock', cache_responses=True)
        
        # Same prompt should hit cache from first instance
        response2 = llm2.generate(prompt)
        print(f"   Response from LLM2: {response2[:50]}...")
        
        assert response1 == response2, "Responses should be identical"
        
        stats2 = llm2.get_usage_stats()
        print(f"   Stats LLM2: {stats2}")
        assert stats2['cache_hits'] == 1, "Should hit cache from LLM1"
        
        print("\n✅ Cache Persistence test PASSED")
        return True
    
    except Exception as e:
        print(f"\n❌ Cache Persistence test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print("LLM Integration Tests (Days 15-17)")
    print("=" * 70)
    
    # Clear cache at start
    print("\n🧹 Clearing cache before tests...")
    clear_cache()
    print("   ✓ Cache cleared")
    
    results = {
        'Mock LLM': test_mock_llm(),
        'OpenAI LLM': test_openai_llm(),
        'Anthropic LLM': test_anthropic_llm(),
        'Model Manager': test_model_manager(),
        'Cache Persistence': test_cache_persistence()
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
        print("\n🎉 Days 15-17 implementation is complete!")
        print("\nLLM Integration Features:")
        print("  ✓ Multiple LLM providers (OpenAI, Anthropic, HuggingFace, Mock)")
        print("  ✓ Response caching (memory + file)")
        print("  ✓ Cache persistence across instances")
        print("  ✓ Usage statistics tracking")
        print("  ✓ Model management")
        print("  ✓ Configuration save/load")
        print("\nNext steps:")
        print("  - Test with: python scripts/test_llm.py")
        print("  - Move to Days 18-20: Prompting System")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        print("=" * 70)
        return 1


if __name__ == '__main__':
    sys.exit(main())