"""
Chain-of-Thought (CoT) prompting utilities
"""

from typing import List, Dict, Optional
from src.data.loader import AERInstance, Document
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChainOfThoughtPrompt:
    """Chain-of-Thought prompt generator"""
    
    def __init__(self):
        """Initialize CoT prompt generator"""
        logger.debug("Initialized ChainOfThoughtPrompt")
    
    def create_reasoning_steps(self, instance: AERInstance) -> str:
        """
        Create step-by-step reasoning prompts
        
        Args:
            instance: AER instance
            
        Returns:
            Formatted reasoning steps
        """
        steps = """Let's think through this step-by-step:

Step 1: What is the main event that occurred?
The event is: {event}

Step 2: What context information is provided?
{context}

Step 3: Analyze each possible cause:
A) {option_A}
   - Does this logically lead to the event?
   - Is this supported by the context?

B) {option_B}
   - Does this logically lead to the event?
   - Is this supported by the context?

C) {option_C}
   - Does this logically lead to the event?
   - Is this supported by the context?

D) {option_D}
   - Does this logically lead to the event?
   - Is this supported by the context?

Step 4: Which cause(s) is/are most plausible?
Based on the analysis above, the most plausible cause(s) is/are:"""
        
        # Format context
        context_summary = "No additional context provided."
        if instance.docs:
            doc_summaries = []
            for i, doc in enumerate(instance.docs[:3], 1):
                summary = doc.get_text(max_length=150)
                doc_summaries.append(f"- Document {i}: {summary}")
            context_summary = "\n".join(doc_summaries)
        
        return steps.format(
            event=instance.target_event,
            context=context_summary,
            option_A=instance.option_A,
            option_B=instance.option_B,
            option_C=instance.option_C,
            option_D=instance.option_D
        )
    
    def create_with_examples(
        self,
        instance: AERInstance,
        examples: Optional[List[AERInstance]] = None
    ) -> str:
        """
        Create CoT prompt with examples
        
        Args:
            instance: Current instance
            examples: Example instances with reasoning
            
        Returns:
            Formatted prompt with examples
        """
        prompt_parts = []
        
        # Add instruction
        prompt_parts.append(
            "Solve this causal reasoning problem using step-by-step thinking. "
            "Here are some examples:\n"
        )
        
        # Add examples if provided
        if examples:
            for i, ex in enumerate(examples[:2], 1):  # Limit to 2 examples
                example_reasoning = self._create_example_reasoning(ex)
                prompt_parts.append(f"\nExample {i}:")
                prompt_parts.append(example_reasoning)
                prompt_parts.append(f"Answer: {ex.golden_answer}\n")
        
        # Add current problem
        prompt_parts.append("\nNow solve this problem:")
        prompt_parts.append(self.create_reasoning_steps(instance))
        
        return "\n".join(prompt_parts)
    
    def _create_example_reasoning(self, instance: AERInstance) -> str:
        """Create example reasoning for demonstration"""
        return f"""Event: {instance.target_event}

Options:
A) {instance.option_A}
B) {instance.option_B}
C) {instance.option_C}
D) {instance.option_D}

Reasoning:
Looking at the event and available context, we need to identify what caused this event.
Option {instance.golden_answer} is the most plausible because it directly relates to the event and is supported by the evidence."""
    
    def create_analytical_prompt(self, instance: AERInstance) -> str:
        """
        Create analytical CoT prompt
        
        Args:
            instance: AER instance
            
        Returns:
            Analytical reasoning prompt
        """
        prompt = f"""Analyze this causal reasoning problem systematically:

EVENT: {instance.target_event}

CONTEXT:
{self._format_context(instance.docs)}

POSSIBLE CAUSES:
A) {instance.option_A}
B) {instance.option_B}
C) {instance.option_C}
D) {instance.option_D}

ANALYSIS FRAMEWORK:

1. TEMPORAL REASONING:
   - Which causes could have occurred BEFORE the event?
   - What is the timeline relationship?

2. LOGICAL CONNECTION:
   - Which causes have a direct causal link to the event?
   - Are there any logical contradictions?

3. EVIDENCE SUPPORT:
   - What evidence from the context supports each cause?
   - Which causes lack supporting evidence?

4. PLAUSIBILITY ASSESSMENT:
   - Which causes are most likely given the evidence?
   - Should we consider multiple causes or insufficient information?

CONCLUSION:
Based on the analysis, the most plausible cause(s) is/are:"""
        
        return prompt
    
    def _format_context(self, docs: List[Document]) -> str:
        """Format context documents"""
        if not docs:
            return "No additional context provided."
        
        formatted = []
        for i, doc in enumerate(docs[:3], 1):
            text = doc.get_text(max_length=200)
            formatted.append(f"{i}. {doc.title}: {text}")
        
        return "\n".join(formatted)
    
    def create_comparative_prompt(self, instance: AERInstance) -> str:
        """
        Create comparative analysis prompt
        
        Args:
            instance: AER instance
            
        Returns:
            Comparative reasoning prompt
        """
        prompt = f"""Compare and contrast the possible causes for this event:

EVENT: {instance.target_event}

Let's compare each option:

Option A: {instance.option_A}
Pros: What evidence supports this?
Cons: What evidence contradicts this?
Likelihood: How likely is this to cause the event?

Option B: {instance.option_B}
Pros: What evidence supports this?
Cons: What evidence contradicts this?
Likelihood: How likely is this to cause the event?

Option C: {instance.option_C}
Pros: What evidence supports this?
Cons: What evidence contradicts this?
Likelihood: How likely is this to cause the event?

Option D: {instance.option_D}
Note: This option means there's insufficient information to determine the cause.
Consider: Is the available evidence sufficient to identify the cause?

CONTEXT EVIDENCE:
{self._format_context(instance.docs)}

After comparing all options, the most plausible cause(s) based on evidence and reasoning is/are:"""
        
        return prompt
    
    def create_verification_prompt(
        self,
        instance: AERInstance,
        initial_answer: str
    ) -> str:
        """
        Create verification prompt to check initial answer
        
        Args:
            instance: AER instance
            initial_answer: Initial predicted answer
            
        Returns:
            Verification prompt
        """
        prompt = f"""Verify this causal reasoning:

EVENT: {instance.target_event}

INITIAL ANSWER: {initial_answer}

Let's verify this answer:

1. Does the answer logically explain the event?
2. Is the answer supported by the available context?
3. Are there any contradictions or issues?
4. Should we reconsider other options?

CONTEXT:
{self._format_context(instance.docs)}

OPTIONS:
A) {instance.option_A}
B) {instance.option_B}
C) {instance.option_C}
D) {instance.option_D}

After verification, the final answer is:"""
        
        return prompt


class MultiStepReasoner:
    """Multi-step reasoning with explicit steps"""
    
    def __init__(self):
        """Initialize multi-step reasoner"""
        self.cot = ChainOfThoughtPrompt()
        logger.info("Initialized MultiStepReasoner")
    
    def decompose_problem(self, instance: AERInstance) -> Dict[str, str]:
        """
        Decompose problem into sub-questions
        
        Args:
            instance: AER instance
            
        Returns:
            Dictionary of sub-questions
        """
        return {
            'what_happened': f"What exactly happened in this event: {instance.target_event}?",
            'when': "When did this likely occur? What is the temporal context?",
            'context': "What additional context is provided?",
            'causes': "What are the potential causes listed?",
            'evidence': "What evidence supports each cause?",
            'conclusion': "Which cause(s) is/are most plausible?"
        }
    
    def create_step_by_step_prompt(self, instance: AERInstance) -> str:
        """
        Create explicit step-by-step prompt
        
        Args:
            instance: AER instance
            
        Returns:
            Step-by-step prompt
        """
        sub_questions = self.decompose_problem(instance)
        
        prompt = f"""Solve this step-by-step:

{sub_questions['what_happened']}
Answer: {instance.target_event}

{sub_questions['context']}
Answer: {self.cot._format_context(instance.docs)}

{sub_questions['causes']}
Answer:
- Option A: {instance.option_A}
- Option B: {instance.option_B}
- Option C: {instance.option_C}
- Option D: {instance.option_D}

{sub_questions['evidence']}
Answer: [Analyze evidence for each option]

{sub_questions['conclusion']}
Answer:"""
        
        return prompt


def create_cot_prompt(
    instance: AERInstance,
    style: str = 'standard',
    examples: Optional[List[AERInstance]] = None
) -> str:
    """
    Factory function to create CoT prompts
    
    Args:
        instance: AER instance
        style: CoT style ('standard', 'analytical', 'comparative', 'step_by_step')
        examples: Example instances for few-shot CoT
        
    Returns:
        CoT prompt string
    """
    cot = ChainOfThoughtPrompt()
    
    if style == 'standard':
        if examples:
            return cot.create_with_examples(instance, examples)
        else:
            return cot.create_reasoning_steps(instance)
    
    elif style == 'analytical':
        return cot.create_analytical_prompt(instance)
    
    elif style == 'comparative':
        return cot.create_comparative_prompt(instance)
    
    elif style == 'step_by_step':
        reasoner = MultiStepReasoner()
        return reasoner.create_step_by_step_prompt(instance)
    
    else:
        raise ValueError(f"Unknown CoT style: {style}")