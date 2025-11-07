"""
Prompt templates for AER task
"""

from typing import Dict, List, Optional
from string import Template

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PromptTemplate:
    """Base prompt template"""
    
    def __init__(self, template: str, required_vars: Optional[List[str]] = None):
        """
        Initialize template
        
        Args:
            template: Template string with {variable} placeholders
            required_vars: List of required variables
        """
        self.template = Template(template)
        self.required_vars = required_vars or []
    
    def format(self, **kwargs) -> str:
        """
        Format template with variables
        
        Args:
            **kwargs: Template variables
            
        Returns:
            Formatted string
        """
        # Check required variables
        missing = [var for var in self.required_vars if var not in kwargs]
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        
        try:
            return self.template.safe_substitute(**kwargs)
        except Exception as e:
            logger.error(f"Error formatting template: {e}")
            raise


# Zero-shot prompts
ZERO_SHOT_SIMPLE = PromptTemplate(
    template="""Given the following event and possible causes, identify the most plausible cause(s).

Event: $target_event

Possible Causes:
A) $option_A
B) $option_B
C) $option_C
D) $option_D

Answer with only the letter(s) of the correct cause(s), separated by commas if multiple (e.g., "A", "B,C", "A,D").

Answer:""",
    required_vars=['target_event', 'option_A', 'option_B', 'option_C', 'option_D']
)


ZERO_SHOT_WITH_CONTEXT = PromptTemplate(
    template="""Given the following event, context documents, and possible causes, identify the most plausible cause(s).

Event: $target_event

Context Documents:
$documents

Possible Causes:
A) $option_A
B) $option_B
C) $option_C
D) $option_D

Based on the event and context, answer with only the letter(s) of the correct cause(s), separated by commas if multiple.

Answer:""",
    required_vars=['target_event', 'documents', 'option_A', 'option_B', 'option_C', 'option_D']
)


# Few-shot prompts
FEW_SHOT_TEMPLATE = PromptTemplate(
    template="""Given events and possible causes, identify the most plausible cause(s). Here are some examples:

$examples

Now answer this one:

Event: $target_event

Possible Causes:
A) $option_A
B) $option_B
C) $option_C
D) $option_D

Answer:""",
    required_vars=['examples', 'target_event', 'option_A', 'option_B', 'option_C', 'option_D']
)


# Chain-of-thought prompts
CHAIN_OF_THOUGHT = PromptTemplate(
    template="""Given the following event and possible causes, identify the most plausible cause(s) using step-by-step reasoning.

Event: $target_event

Context Documents:
$documents

Possible Causes:
A) $option_A
B) $option_B
C) $option_C
D) $option_D

Think through this step-by-step:
1. What is the main event that occurred?
2. What information do the context documents provide?
3. Which cause(s) are most consistent with the event and context?
4. Are there any causes that can be ruled out?

Based on your reasoning, what is the most plausible cause(s)?

Answer:""",
    required_vars=['target_event', 'documents', 'option_A', 'option_B', 'option_C', 'option_D']
)


# Abductive reasoning prompt
ABDUCTIVE_REASONING = PromptTemplate(
    template="""Use abductive reasoning to identify the most plausible cause(s) for this event.

Abductive reasoning finds the simplest and most likely explanation for an observation.

Event (Observation): $target_event

Context: $documents

Possible Explanations:
A) $option_A
B) $option_B
C) $option_C
D) $option_D

Reasoning Process:
1. Consider each explanation's plausibility given the observation
2. Evaluate which explanation best accounts for the available evidence
3. Identify the simplest explanation that fits all facts

Most plausible cause(s):""",
    required_vars=['target_event', 'documents', 'option_A', 'option_B', 'option_C', 'option_D']
)


# Template registry
TEMPLATE_REGISTRY = {
    'zero_shot_simple': ZERO_SHOT_SIMPLE,
    'zero_shot_context': ZERO_SHOT_WITH_CONTEXT,
    'few_shot': FEW_SHOT_TEMPLATE,
    'chain_of_thought': CHAIN_OF_THOUGHT,
    'abductive': ABDUCTIVE_REASONING
}


def get_template(name: str) -> PromptTemplate:
    """
    Get template by name
    
    Args:
        name: Template name
        
    Returns:
        PromptTemplate instance
    """
    if name not in TEMPLATE_REGISTRY:
        raise ValueError(f"Template not found: {name}")
    return TEMPLATE_REGISTRY[name]


def list_templates() -> List[str]:
    """List available templates"""
    return list(TEMPLATE_REGISTRY.keys())