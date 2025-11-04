#!/usr/bin/env python3
"""
Quick test to verify setup is complete
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import flask
        print("✓ Flask")
    except ImportError as e:
        print(f"✗ Flask: {e}")
        return False
    
    try:
        import torch
        print("✓ PyTorch")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        return False
    
    try:
        import transformers
        print("✓ Transformers")
    except ImportError as e:
        print(f"✗ Transformers: {e}")
        return False
    
    try:
        import pandas
        print("✓ Pandas")
    except ImportError as e:
        print(f"✗ Pandas: {e}")
        return False
    
    try:
        from src.data.loader import AERDataLoader
        print("✓ Data Loader")
    except ImportError as e:
        print(f"✗ Data Loader: {e}")
        return False
    
    try:
        from src.data.preprocessor import TextPreprocessor
        print("✓ Preprocessor")
    except ImportError as e:
        print(f"✗ Preprocessor: {e}")
        return False
    
    try:
        from src.utils.logger import setup_logger
        print("✓ Logger")
    except ImportError as e:
        print(f"✗ Logger: {e}")
        return False
    
    return True


def test_data_loading():
    """Test data loading with sample data"""
    print("\nTesting data loading...")
    
    try:
        from src.data.loader import AERDataLoader
        
        # Try to load sample data
        if Path('data/sample/questions.jsonl').exists():
            loader = AERDataLoader('data/sample')
            instances = loader.load()
            print(f"✓ Loaded {len(instances)} sample instances")
            return True
        else:
            print("⚠️  Sample data not found (this is OK if not set up yet)")
            return True
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False


def main():
    print("=" * 60)
    print("AER Project Setup Verification")
    print("=" * 60)
    print()
    
    # Test imports
    imports_ok = test_imports()
    
    # Test data loading
    data_ok = test_data_loading()
    
    print()
    print("=" * 60)
    if imports_ok and data_ok:
        print("✅ All tests passed!")
        print("=" * 60)
        print("\n Next steps:")
        print("  1. Copy your datasets to data/ directories")
        print("  2. Run: python scripts/analyze_data.py --data-dir data/train")
        return 0
    else:
        print("❌ Some tests failed!")
        print("=" * 60)
        print("\nPlease fix the errors above and try again.")
        return 1


if __name__ == '__main__':
    sys.exit(main())