"""
Confidence estimation for predictions
"""

from typing import Dict, List
import re

from src.data.loader import AERInstance
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConfidenceEstimator:
    """Estimate confidence scores for predictions"""
    
    def __init__(self):
        """Initialize confidence estimator"""
        logger.debug("Initialized ConfidenceEstimator")
    
    def estimate(
        self,
        instance: AERInstance,
        reasoning: str,
        prediction: str
    ) -> Dict[str, float]:
        """
        Estimate confidence scores for each option
        
        Args:
            instance: AER instance
            reasoning: Model's reasoning text
            prediction: Predicted answer
            
        Returns:
            Dictionary of confidence scores for each option
        """
        # Initialize scores
        scores = {opt: 0.0 for opt in ['A', 'B', 'C', 'D']}
        
        # Give high confidence to predicted options
        predicted_options = prediction.split(',')
        for opt in predicted_options:
            opt = opt.strip()
            if opt in scores:
                scores[opt] = 0.8
        
        # Adjust based on reasoning text
        scores = self._adjust_by_reasoning(scores, reasoning, instance)
        
        # Normalize to sum to 1.0
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        
        return scores
    
    def _adjust_by_reasoning(
        self,
        scores: Dict[str, float],
        reasoning: str,
        instance: AERInstance
    ) -> Dict[str, float]:
        """Adjust scores based on reasoning text"""
        reasoning_lower = reasoning.lower()
        
        # Check for confidence indicators
        confidence_words = {
            'certain': 0.9,
            'definitely': 0.9,
            'clearly': 0.85,
            'strongly': 0.85,
            'likely': 0.7,
            'probably': 0.7,
            'possibly': 0.5,
            'maybe': 0.4,
            'unlikely': 0.3,
            'uncertain': 0.3
        }
        
        # Find confidence indicators
        max_confidence = 0.8
        for word, conf in confidence_words.items():
            if word in reasoning_lower:
                max_confidence = max(max_confidence, conf)
        
        # Adjust predicted option scores
        for opt, score in scores.items():
            if score > 0:
                scores[opt] = max_confidence
        
        # Check for option mentions in reasoning
        options = instance.get_options_dict()
        for opt, opt_text in options.items():
            # Count mentions of this option
            opt_pattern = rf'\b{opt}\b'
            mentions = len(re.findall(opt_pattern, reasoning, re.IGNORECASE))
            
            # Check for option text mentions
            if opt_text.lower() in reasoning_lower:
                mentions += 1
            
            # Boost score based on mentions
            if mentions > 0:
                boost = min(0.2, mentions * 0.1)
                scores[opt] = min(1.0, scores[opt] + boost)
        
        return scores
    
    def estimate_batch(
        self,
        instances: List[AERInstance],
        reasonings: List[str],
        predictions: List[str]
    ) -> List[Dict[str, float]]:
        """
        Estimate confidence for multiple predictions
        
        Args:
            instances: List of instances
            reasonings: List of reasoning texts
            predictions: List of predictions
            
        Returns:
            List of confidence dictionaries
        """
        return [
            self.estimate(inst, reason, pred)
            for inst, reason, pred in zip(instances, reasonings, predictions)
        ]