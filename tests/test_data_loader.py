"""
Unit tests for data loader
"""

import pytest
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import AERDataLoader, AERInstance, Document


def test_data_loader_initialization():
    """Test data loader initialization"""
    # This will fail if data doesn't exist, which is expected
    try:
        loader = AERDataLoader('data/sample')
        assert loader.data_dir == Path('data/sample')
        assert loader.questions_file.name == 'questions.jsonl'
        assert loader.docs_file.name == 'docs.json'
    except FileNotFoundError:
        pytest.skip("Sample data not found")


def test_load_data():
    """Test data loading"""
    try:
        loader = AERDataLoader('data/sample')
        instances = loader.load()
        
        assert len(instances) > 0
        assert all(isinstance(inst, AERInstance) for inst in instances)
        
        # Test first instance
        first = instances[0]
        assert hasattr(first, 'topic_id')
        assert hasattr(first, 'target_event')
        assert hasattr(first, 'option_A')
        assert hasattr(first, 'docs')
        
    except FileNotFoundError:
        pytest.skip("Sample data not found")


def test_instance_methods():
    """Test AERInstance methods"""
    try:
        loader = AERDataLoader('data/sample')
        instances = loader.load()
        
        if len(instances) > 0:
            inst = instances[0]
            
            # Test options methods
            options_dict = inst.get_options_dict()
            assert 'A' in options_dict
            assert 'B' in options_dict
            assert 'C' in options_dict
            assert 'D' in options_dict
            
            options_list = inst.get_options_list()
            assert len(options_list) == 4
            
            # Test answer methods
            if inst.has_answer():
                answer_list = inst.get_answer_list()
                assert isinstance(answer_list, list)
                assert all(opt in ['A', 'B', 'C', 'D'] for opt in answer_list)
        
    except FileNotFoundError:
        pytest.skip("Sample data not found")


def test_statistics():
    """Test statistics generation"""
    try:
        loader = AERDataLoader('data/sample')
        stats = loader.get_statistics()
        
        assert 'total_instances' in stats
        assert 'total_documents' in stats
        assert 'avg_docs_per_instance' in stats
        assert stats['total_instances'] > 0
        
    except FileNotFoundError:
        pytest.skip("Sample data not found")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])