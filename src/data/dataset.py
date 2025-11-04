"""
PyTorch Dataset classes for AER
"""

from typing import List, Dict, Optional, Callable
import torch
from torch.utils.data import Dataset

from src.data.loader import AERInstance
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AERDataset(Dataset):
    """PyTorch Dataset for AER task"""
    
    def __init__(
        self,
        instances: List[AERInstance],
        transform: Optional[Callable] = None
    ):
        """
        Initialize dataset
        
        Args:
            instances: List of AER instances
            transform: Optional transform function
        """
        self.instances = instances
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.instances)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get item at index"""
        instance = self.instances[idx]
        
        # Create sample dictionary
        sample = {
            'topic_id': instance.topic_id,
            'uuid': instance.uuid,
            'target_event': instance.target_event,
            'options': instance.get_options_dict(),
            'answer': instance.answer,
            'docs': [
                {
                    'title': doc.title,
                    'content': doc.content,
                    'source': doc.source
                }
                for doc in instance.docs
            ],
            'topic': instance.topic
        }
        
        # Apply transform if provided
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def get_instance(self, idx: int) -> AERInstance:
        """Get raw instance at index"""
        return self.instances[idx]


class AERCollator:
    """Custom collator for batching AER data"""
    
    def __call__(self, batch: List[Dict]) -> Dict:
        """
        Collate batch of samples
        
        Args:
            batch: List of sample dictionaries
            
        Returns:
            Batched dictionary
        """
        batched = {
            'topic_ids': [],
            'uuids': [],
            'target_events': [],
            'options': [],
            'answers': [],
            'docs': [],
            'topics': []
        }
        
        for sample in batch:
            batched['topic_ids'].append(sample['topic_id'])
            batched['uuids'].append(sample['uuid'])
            batched['target_events'].append(sample['target_event'])
            batched['options'].append(sample['options'])
            batched['answers'].append(sample['answer'])
            batched['docs'].append(sample['docs'])
            batched['topics'].append(sample['topic'])
        
        return batched