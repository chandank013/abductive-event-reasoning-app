"""
Dynamic prompt construction for AER task
"""

from typing import List, Dict, Optional
from src.data.loader import AERInstance, Document
from src.prompting.templates import get_template, PromptTemplate
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PromptBuilder:
    """Build prompts for AER task"""
    
    def __init__(
        self,
        template_name: str = 'zero_shot_context',
        max_doc_length: int = 500,
        max_docs: int = 5
    ):
        """
        Initialize prompt builder
        
        Args:
            template_name: Name of prompt template to use
            max_doc_length: Maximum characters per document
            max_docs: Maximum number of documents to include
        """
        self.template = get_template(template_name)
        self.max_doc_length = max_doc_length
        self.max_docs = max_docs
        
        logger.info(f"Initialized PromptBuilder with template: {template_name}")
    
    def build(
        self,
        instance: AERInstance,
        include_documents: bool = True,
        examples: Optional[List[AERInstance]] = None
    ) -> str:
        """
        Build prompt for instance
        
        Args:
            instance: AER instance
            include_documents: Whether to include context documents
            examples: Few-shot examples (if using few-shot template)
            
        Returns:
            Formatted prompt string
        """
        # Prepare variables
        variables = {
            'target_event': instance.target_event,
            'option_A': instance.option_A,
            'option_B': instance.option_B,
            'option_C': instance.option_C,
            'option_D': instance.option_D
        }
        
        # Add documents if needed
        if include_documents and instance.docs:
            documents_text = self._format_documents(instance.docs)
            variables['documents'] = documents_text
        else:
            variables['documents'] = "No additional context provided."
        
        # Add examples for few-shot
        if examples:
            examples_text = self._format_examples(examples)
            variables['examples'] = examples_text
        
        # Format template
        try:
            prompt = self.template.format(**variables)
            return prompt
        except Exception as e:
            logger.error(f"Error building prompt: {e}")
            raise
    
    def _format_documents(self, docs: List[Document]) -> str:
        """
        Format documents for prompt
        
        Args:
            docs: List of documents
            
        Returns:
            Formatted documents string
        """
        # Limit number of documents
        docs = docs[:self.max_docs]
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            content = doc.get_text(max_length=self.max_doc_length)
            formatted.append(f"Document {i}: {doc.title}\n{content}")
        
        return "\n\n".join(formatted)
    
    def _format_examples(self, examples: List[AERInstance]) -> str:
        """
        Format few-shot examples
        
        Args:
            examples: List of example instances
            
        Returns:
            Formatted examples string
        """
        formatted = []
        
        for i, ex in enumerate(examples, 1):
            example_text = f"""Example {i}:
Event: {ex.target_event}
A) {ex.option_A}
B) {ex.option_B}
C) {ex.option_C}
D) {ex.option_D}
Answer: {ex.golden_answer}
"""
            formatted.append(example_text)
        
        return "\n".join(formatted)
    
    def build_batch(
        self,
        instances: List[AERInstance],
        include_documents: bool = True
    ) -> List[str]:
        """
        Build prompts for multiple instances
        
        Args:
            instances: List of AER instances
            include_documents: Whether to include documents
            
        Returns:
            List of prompts
        """
        return [
            self.build(inst, include_documents=include_documents)
            for inst in instances
        ]


def create_prompt_builder(
    strategy: str = 'zero_shot',
    **kwargs
) -> PromptBuilder:
    """
    Factory function to create prompt builder
    
    Args:
        strategy: Prompting strategy ('zero_shot', 'few_shot', 'cot', 'abductive')
        **kwargs: Additional arguments for PromptBuilder
        
    Returns:
        PromptBuilder instance
    """
    template_map = {
        'zero_shot': 'zero_shot_context',
        'zero_shot_simple': 'zero_shot_simple',
        'few_shot': 'few_shot',
        'cot': 'chain_of_thought',
        'abductive': 'abductive'
    }
    
    template_name = template_map.get(strategy, 'zero_shot_context')
    return PromptBuilder(template_name=template_name, **kwargs)