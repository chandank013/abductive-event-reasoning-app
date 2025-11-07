"""
Abductive reasoning implementation for AER task
"""

from typing import Dict, List, Optional, Tuple
import re

from src.data.loader import AERInstance
from src.models.llm_wrapper import BaseLLM
from src.prompting.prompt_builder import PromptBuilder
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AbductiveReasoner:
    """Abductive reasoning for event causality"""
    
    def __init__(
        self,
        model: BaseLLM,
        prompt_builder: Optional[PromptBuilder] = None,
        temperature: float = 0.3,
        max_tokens: int = 100
    ):
        """
        Initialize abductive reasoner
        
        Args:
            model: LLM model for reasoning
            prompt_builder: Prompt builder (creates default if None)
            temperature: Sampling temperature
            max_tokens: Maximum tokens for generation
        """
        self.model = model
        self.prompt_builder = prompt_builder or PromptBuilder(template_name='abductive')
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        logger.info(f"Initialized AbductiveReasoner with model: {model.model_name}")
    
    def reason(
        self,
        instance: AERInstance,
        return_reasoning: bool = False,
        include_documents: bool = True
    ) -> Dict:
        """
        Perform abductive reasoning on instance
        
        Args:
            instance: AER instance
            return_reasoning: Whether to return the reasoning text
            include_documents: Whether to include context documents
            
        Returns:
            Dictionary with prediction and optionally reasoning
        """
        logger.info(f"Reasoning on instance: {instance.uuid}")
        
        # Build prompt
        prompt = self.prompt_builder.build(
            instance,
            include_documents=include_documents
        )
        
        # Generate response
        response = self.model.generate(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Extract answer
        from src.reasoning.answer_extractor import AnswerExtractor
        extractor = AnswerExtractor()
        prediction = extractor.extract(response)
        
        result = {
            'prediction': prediction,
            'uuid': instance.uuid
        }
        
        if return_reasoning:
            result['reasoning'] = response
            result['prompt'] = prompt
        
        return result
    
    def reason_batch(
        self,
        instances: List[AERInstance],
        include_documents: bool = True,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Perform reasoning on multiple instances
        
        Args:
            instances: List of AER instances
            include_documents: Whether to include documents
            show_progress: Show progress bar
            
        Returns:
            List of results
        """
        logger.info(f"Reasoning on {len(instances)} instances")
        
        results = []
        
        if show_progress:
            from tqdm import tqdm
            instances = tqdm(instances, desc="Reasoning")
        
        for instance in instances:
            try:
                result = self.reason(
                    instance,
                    include_documents=include_documents
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error reasoning on {instance.uuid}: {e}")
                results.append({
                    'prediction': 'D',  # Default to insufficient info
                    'uuid': instance.uuid,
                    'error': str(e)
                })
        
        return results
    
    def reason_with_confidence(
        self,
        instance: AERInstance,
        include_documents: bool = True
    ) -> Dict:
        """
        Reason with confidence scores for each option
        
        Args:
            instance: AER instance
            include_documents: Whether to include documents
            
        Returns:
            Dictionary with prediction and confidence scores
        """
        # Get basic prediction
        result = self.reason(
            instance,
            return_reasoning=True,
            include_documents=include_documents
        )
        
        # Calculate confidence scores
        from src.reasoning.confidence import ConfidenceEstimator
        estimator = ConfidenceEstimator()
        confidence = estimator.estimate(
            instance,
            result['reasoning'],
            result['prediction']
        )
        
        result['confidence'] = confidence
        
        return result


class ChainOfThoughtReasoner(AbductiveReasoner):
    """Chain-of-thought reasoning variant"""
    
    def __init__(self, model: BaseLLM, **kwargs):
        """Initialize with CoT prompt template"""
        prompt_builder = PromptBuilder(template_name='chain_of_thought')
        super().__init__(
            model=model,
            prompt_builder=prompt_builder,
            max_tokens=200,  # CoT needs more tokens
            **kwargs
        )
        logger.info("Initialized Chain-of-Thought Reasoner")


class FewShotReasoner(AbductiveReasoner):
    """Few-shot learning reasoner"""
    
    def __init__(
        self,
        model: BaseLLM,
        training_instances: List[AERInstance],
        num_examples: int = 3,
        **kwargs
    ):
        """
        Initialize few-shot reasoner
        
        Args:
            model: LLM model
            training_instances: Training instances for examples
            num_examples: Number of examples to use
        """
        prompt_builder = PromptBuilder(template_name='few_shot')
        super().__init__(model=model, prompt_builder=prompt_builder, **kwargs)
        
        self.training_instances = training_instances
        self.num_examples = num_examples
        
        logger.info(f"Initialized Few-Shot Reasoner with {num_examples} examples")
    
    def reason(
        self,
        instance: AERInstance,
        return_reasoning: bool = False,
        include_documents: bool = True
    ) -> Dict:
        """Reason with few-shot examples"""
        # Select examples
        from src.prompting.few_shot import FewShotSelector
        selector = FewShotSelector(strategy='random')
        examples = selector.select(
            instance,
            self.training_instances,
            k=self.num_examples
        )
        
        # Build prompt with examples
        prompt = self.prompt_builder.build(
            instance,
            include_documents=include_documents,
            examples=examples
        )
        
        # Generate response
        response = self.model.generate(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Extract answer
        from src.reasoning.answer_extractor import AnswerExtractor
        extractor = AnswerExtractor()
        prediction = extractor.extract(response)
        
        result = {
            'prediction': prediction,
            'uuid': instance.uuid
        }
        
        if return_reasoning:
            result['reasoning'] = response
            result['prompt'] = prompt
            result['examples'] = [ex.uuid for ex in examples]
        
        return result