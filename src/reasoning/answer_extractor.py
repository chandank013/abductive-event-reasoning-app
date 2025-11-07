"""
Extract answers from model outputs
"""

import re
from typing import List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AnswerExtractor:
    """Extract structured answers from LLM outputs"""
    
    def __init__(self):
        """Initialize answer extractor"""
        self.valid_options = {'A', 'B', 'C', 'D'}
        logger.debug("Initialized AnswerExtractor")
    
    def extract(self, text: str) -> str:
        """
        Extract answer from text
        
        Args:
            text: Model output text
            
        Returns:
            Answer string (e.g., "A", "B,C", "A,D")
        """
        # Try different extraction methods
        answer = self._extract_explicit(text)
        if answer:
            return answer
        
        answer = self._extract_pattern(text)
        if answer:
            return answer
        
        answer = self._extract_first_mention(text)
        if answer:
            return answer
        
        # Default to D (insufficient information)
        logger.warning(f"Could not extract answer from: {text[:100]}...")
        return "D"
    
    def _extract_explicit(self, text: str) -> Optional[str]:
        """Extract explicitly marked answer"""
        # Look for patterns like "Answer: A,B" or "The answer is A"
        patterns = [
            r'answer:\s*([A-D](?:,\s*[A-D])*)',
            r'answer is\s*([A-D](?:,\s*[A-D])*)',
            r'correct (?:answer|option|cause)s?:\s*([A-D](?:,\s*[A-D])*)',
            r'most plausible (?:cause|option)s?:\s*([A-D](?:,\s*[A-D])*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                answer = match.group(1).upper()
                return self._normalize_answer(answer)
        
        return None
    
    def _extract_pattern(self, text: str) -> Optional[str]:
        """Extract answer using common patterns"""
        # Look for option letters followed by comma or end
        # e.g., "A,B" or "A and B" or "A, B"
        
        # Pattern: Single letter at start
        if re.match(r'^[A-D]$', text.strip()):
            return text.strip().upper()
        
        # Pattern: Multiple letters with commas
        match = re.search(r'\b([A-D](?:,\s*[A-D])+)\b', text)
        if match:
            return self._normalize_answer(match.group(1))
        
        # Pattern: "A and B" format
        match = re.search(r'\b([A-D])\s+and\s+([A-D])\b', text, re.IGNORECASE)
        if match:
            return self._normalize_answer(f"{match.group(1)},{match.group(2)}")
        
        return None
    
    def _extract_first_mention(self, text: str) -> Optional[str]:
        """Extract first mentioned valid options"""
        # Find all option mentions
        options = re.findall(r'\b([A-D])\b', text.upper())
        
        if not options:
            return None
        
        # Get unique options in order
        seen = set()
        unique_options = []
        for opt in options:
            if opt not in seen:
                seen.add(opt)
                unique_options.append(opt)
        
        # Return first 1-3 options
        if len(unique_options) == 0:
            return None
        elif len(unique_options) == 1:
            return unique_options[0]
        else:
            # Take up to 3 options
            return ','.join(sorted(unique_options[:3]))
    
    def _normalize_answer(self, answer: str) -> str:
        """
        Normalize answer format
        
        Args:
            answer: Raw answer string
            
        Returns:
            Normalized answer (e.g., "A,B")
        """
        # Remove whitespace and convert to uppercase
        answer = answer.upper().replace(' ', '')
        
        # Extract valid options
        options = re.findall(r'[A-D]', answer)
        
        if not options:
            return "D"
        
        # Remove duplicates and sort
        unique_options = sorted(set(options))
        
        return ','.join(unique_options)
    
    def extract_batch(self, texts: List[str]) -> List[str]:
        """
        Extract answers from multiple texts
        
        Args:
            texts: List of model outputs
            
        Returns:
            List of extracted answers
        """
        return [self.extract(text) for text in texts]
    
    def validate_answer(self, answer: str) -> bool:
        """
        Validate answer format
        
        Args:
            answer: Answer string
            
        Returns:
            True if valid
        """
        if not answer:
            return False
        
        options = answer.split(',')
        return all(opt.strip() in self.valid_options for opt in options)