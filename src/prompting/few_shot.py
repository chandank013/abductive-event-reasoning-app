"""
Few-shot example selection
"""

from typing import List, Optional
import random
import numpy as np

from src.data.loader import AERInstance
from src.retrieval.embedder import SentenceEmbedder
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FewShotSelector:
    """Select few-shot examples for prompting"""
    
    def __init__(
        self,
        strategy: str = 'random',
        embedder: Optional[SentenceEmbedder] = None
    ):
        """
        Initialize selector
        
        Args:
            strategy: Selection strategy ('random', 'similar', 'diverse')
            embedder: Embedder for similarity-based selection
        """
        self.strategy = strategy
        self.embedder = embedder
        
        if strategy == 'similar' and embedder is None:
            logger.warning("Similarity-based selection requires embedder. Falling back to random.")
            self.strategy = 'random'
        
        logger.info(f"Initialized FewShotSelector with strategy: {self.strategy}")
    
    def select(
        self,
        query_instance: AERInstance,
        candidate_instances: List[AERInstance],
        k: int = 3
    ) -> List[AERInstance]:
        """
        Select k few-shot examples
        
        Args:
            query_instance: Instance to find examples for
            candidate_instances: Pool of candidate examples
            k: Number of examples to select
            
        Returns:
            List of selected examples
        """
        # Filter candidates with answers
        candidates = [inst for inst in candidate_instances if inst.has_answer()]
        
        if len(candidates) == 0:
            logger.warning("No candidates with answers available")
            return []
        
        if len(candidates) <= k:
            return candidates
        
        if self.strategy == 'random':
            return self._select_random(candidates, k)
        elif self.strategy == 'similar':
            return self._select_similar(query_instance, candidates, k)
        elif self.strategy == 'diverse':
            return self._select_diverse(candidates, k)
        else:
            logger.warning(f"Unknown strategy: {self.strategy}. Using random.")
            return self._select_random(candidates, k)
    
    def _select_random(
        self,
        candidates: List[AERInstance],
        k: int
    ) -> List[AERInstance]:
        """Random selection"""
        return random.sample(candidates, k)
    
    def _select_similar(
        self,
        query_instance: AERInstance,
        candidates: List[AERInstance],
        k: int
    ) -> List[AERInstance]:
        """Select most similar examples"""
        # Embed query
        query_text = query_instance.target_event
        query_embedding = self.embedder.embed_query(query_text)
        
        # Embed candidates
        candidate_texts = [inst.target_event for inst in candidates]
        candidate_embeddings = self.embedder.embed(candidate_texts)
        
        # Compute similarities
        similarities = np.dot(candidate_embeddings, query_embedding)
        
        # Get top k
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        return [candidates[i] for i in top_indices]
    
    def _select_diverse(
        self,
        candidates: List[AERInstance],
        k: int
    ) -> List[AERInstance]:
        """Select diverse examples (different answer types)"""
        # Group by answer
        answer_groups = {}
        for inst in candidates:
            answer = inst.golden_answer
            if answer not in answer_groups:
                answer_groups[answer] = []
            answer_groups[answer].append(inst)
        
        # Select one from each group
        selected = []
        for group in answer_groups.values():
            if len(selected) < k:
                selected.append(random.choice(group))
        
        # Fill remaining with random
        if len(selected) < k:
            remaining = [inst for inst in candidates if inst not in selected]
            selected.extend(random.sample(remaining, min(k - len(selected), len(remaining))))
        
        return selected[:k]