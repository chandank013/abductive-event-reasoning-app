"""
Data loader for SemEval 2026 Task 12 - AER
Handles questions.jsonl + docs.json format
"""

from pathlib import Path
from typing import Dict, List, Optional
import json
import jsonlines
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Document:
    """Single document/article"""
    title: str
    content: str
    source: str = ""
    link: str = ""
    snippet: str = ""
    uuid: str = ""
    
    def get_text(self, max_length: Optional[int] = None) -> str:
        """Get document text, optionally truncated"""
        text = self.content if self.content else self.snippet
        if max_length and len(text) > max_length:
            return text[:max_length] + "..."
        return text
    
    def get_word_count(self) -> int:
        """Get word count"""
        return len(self.get_text().split())


@dataclass
class AERInstance:
    """Single AER task instance"""
    topic_id: int
    uuid: str
    target_event: str  # The observation to explain (O2)
    option_A: str      # Hypothesis A (Ha)
    option_B: str      # Hypothesis B (Hb)
    option_C: str      # Hypothesis C (Hc)
    option_D: str      # Hypothesis D (Hd)
    golden_answer: Optional[str]  # Correct answer(s)
    docs: List[Document]          # Supporting knowledge base
    topic: str = ""               # Topic description
    
    @property
    def question(self) -> str:
        """Alias for target_event"""
        return self.target_event
    
    @property
    def answer(self) -> str:
        """Alias for golden_answer"""
        return self.golden_answer if self.golden_answer else ""
    
    def get_options_dict(self) -> Dict[str, str]:
        """Get options as dictionary"""
        return {
            'A': self.option_A,
            'B': self.option_B,
            'C': self.option_C,
            'D': self.option_D
        }
    
    def get_options_list(self) -> List[str]:
        """Get options as list"""
        return [self.option_A, self.option_B, self.option_C, self.option_D]
    
    def get_answer_list(self) -> List[str]:
        """Get answer as list of option letters"""
        if not self.golden_answer:
            return []
        return [opt.strip() for opt in self.golden_answer.split(',')]
    
    def has_answer(self) -> bool:
        """Check if instance has answer (False for test data)"""
        return self.golden_answer is not None and self.golden_answer != ""
    
    def get_relevant_docs(self, max_docs: Optional[int] = None) -> List[Document]:
        """Get relevant documents, optionally limited"""
        if max_docs:
            return self.docs[:max_docs]
        return self.docs
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'topic_id': self.topic_id,
            'uuid': self.uuid,
            'target_event': self.target_event,
            'option_A': self.option_A,
            'option_B': self.option_B,
            'option_C': self.option_C,
            'option_D': self.option_D,
            'golden_answer': self.golden_answer,
            'topic': self.topic,
            'num_docs': len(self.docs)
        }


class AERDataLoader:
    """Loader for AER dataset"""
    
    def __init__(self, data_dir: str):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing questions.jsonl and docs.json
        """
        self.data_dir = Path(data_dir)
        self.questions_file = self.data_dir / "questions.jsonl"
        self.docs_file = self.data_dir / "docs.json"
        
        # Validate files exist
        if not self.questions_file.exists():
            raise FileNotFoundError(f"Questions file not found: {self.questions_file}")
        if not self.docs_file.exists():
            raise FileNotFoundError(f"Docs file not found: {self.docs_file}")
    
    def load(self) -> List[AERInstance]:
        """
        Load complete dataset
        
        Returns:
            List of AERInstance objects with associated documents
        """
        logger.info(f"Loading data from {self.data_dir}")
        
        # Load questions
        questions = self._load_questions()
        logger.info(f"Loaded {len(questions)} questions")
        
        # Load documents
        docs_map, topics_map = self._load_docs()
        logger.info(f"Loaded documents for {len(docs_map)} topics")
        
        # Merge questions and documents
        instances = []
        missing_docs = 0
        
        for q in questions:
            topic_id = q['topic_id']
            docs = docs_map.get(topic_id, [])
            topic = topics_map.get(topic_id, "")
            
            if not docs:
                missing_docs += 1
                logger.warning(f"No documents found for topic_id: {topic_id}")
            
            # Convert doc dicts to Document objects
            doc_objects = [
                Document(
                    title=doc.get('title', ''),
                    content=doc.get('content', ''),
                    source=doc.get('source', ''),
                    link=doc.get('link', ''),
                    snippet=doc.get('snippet', ''),
                    uuid=doc.get('uuid', '')
                )
                for doc in docs
            ]
            
            instance = AERInstance(
                topic_id=topic_id,
                uuid=q.get('uuid', ''),
                target_event=q['target_event'],
                option_A=q['option_A'],
                option_B=q['option_B'],
                option_C=q['option_C'],
                option_D=q['option_D'],
                golden_answer=q.get('golden_answer'),
                docs=doc_objects,
                topic=topic
            )
            instances.append(instance)
        
        if missing_docs > 0:
            logger.warning(f"{missing_docs} instances missing documents")
        
        logger.info(f"Created {len(instances)} complete instances")
        return instances
    
    def _load_questions(self) -> List[Dict]:
        """Load questions from JSONL file"""
        questions = []
        with jsonlines.open(self.questions_file, 'r') as reader:
            for obj in reader:
                questions.append(obj)
        return questions
    
    def _load_docs(self) -> tuple:
        """
        Load documents from JSON file
        
        Returns:
            Tuple of (docs_map, topics_map)
        """
        with open(self.docs_file, 'r', encoding='utf-8') as f:
            docs_data = json.load(f)
        
        docs_map = {}
        topics_map = {}
        
        # docs.json is a list of topic objects
        for topic_obj in docs_data:
            topic_id = topic_obj['topic_id']
            docs_map[topic_id] = topic_obj.get('docs', [])
            topics_map[topic_id] = topic_obj.get('topic', '')
        
        return docs_map, topics_map
    
    def get_statistics(self) -> Dict:
        """Get comprehensive dataset statistics"""
        instances = self.load()
        
        total_instances = len(instances)
        instances_with_answers = sum(1 for inst in instances if inst.has_answer())
        total_docs = sum(len(inst.docs) for inst in instances)
        avg_docs_per_instance = total_docs / total_instances if total_instances > 0 else 0
        
        # Answer distribution
        answer_distribution = {}
        for inst in instances:
            if inst.has_answer():
                answer = inst.golden_answer
                answer_distribution[answer] = answer_distribution.get(answer, 0) + 1
        
        # Document statistics
        doc_lengths = []
        docs_with_content = 0
        
        for inst in instances:
            for doc in inst.docs:
                if doc.content:
                    docs_with_content += 1
                    doc_lengths.append(len(doc.content))
        
        # Topic statistics
        unique_topics = len(set(inst.topic_id for inst in instances))
        
        stats = {
            'total_instances': total_instances,
            'instances_with_answers': instances_with_answers,
            'instances_without_answers': total_instances - instances_with_answers,
            'total_documents': total_docs,
            'docs_with_content': docs_with_content,
            'avg_docs_per_instance': avg_docs_per_instance,
            'unique_topics': unique_topics,
            'answer_distribution': answer_distribution,
            'avg_doc_length_chars': sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0,
            'max_doc_length_chars': max(doc_lengths) if doc_lengths else 0,
            'min_doc_length_chars': min(doc_lengths) if doc_lengths else 0,
        }
        
        return stats


def load_aer_data(data_dir: str) -> List[AERInstance]:
    """
    Convenience function to load AER data
    
    Args:
        data_dir: Directory containing questions.jsonl and docs.json
        
    Returns:
        List of AERInstance objects
    """
    loader = AERDataLoader(data_dir)
    return loader.load()