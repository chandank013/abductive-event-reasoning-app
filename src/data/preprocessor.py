"""
Text preprocessing for AER data
"""

import re
import unicodedata
from typing import List, Dict, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TextPreprocessor:
    """Text preprocessing utilities"""
    
    def __init__(
        self,
        lowercase: bool = False,
        remove_stopwords: bool = False,
        remove_punctuation: bool = False,
        min_length: int = 0
    ):
        """
        Initialize preprocessor
        
        Args:
            lowercase: Convert text to lowercase
            remove_stopwords: Remove English stopwords
            remove_punctuation: Remove punctuation
            min_length: Minimum text length (characters)
        """
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.min_length = min_length
        
        # Download NLTK data if needed
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            logger.info("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def preprocess(self, text: str) -> str:
        """
        Apply all preprocessing steps
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Clean text
        text = self.clean_text(text)
        
        # Check minimum length
        if len(text) < self.min_length:
            return ""
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = word_tokenize(text)
            tokens = [t for t in tokens if t.lower() not in self.stop_words]
            text = ' '.join(tokens)
        
        return text
    
    def preprocess_document(self, doc: Dict[str, str]) -> Dict[str, str]:
        """
        Preprocess a document dictionary
        
        Args:
            doc: Document with 'title' and 'content' keys
            
        Returns:
            Preprocessed document
        """
        processed_doc = {}
        
        for key, value in doc.items():
            if isinstance(value, str):
                processed_doc[key] = self.preprocess(value)
            else:
                processed_doc[key] = value
        
        return processed_doc
    
    def preprocess_documents(self, docs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Preprocess list of documents
        
        Args:
            docs: List of documents
            
        Returns:
            List of preprocessed documents
        """
        return [self.preprocess_document(doc) for doc in docs]


def create_preprocessor(config: Optional[Dict] = None) -> TextPreprocessor:
    """
    Create preprocessor from config
    
    Args:
        config: Configuration dictionary
        
    Returns:
        TextPreprocessor instance
    """
    if config is None:
        config = {}
    
    return TextPreprocessor(
        lowercase=config.get('lowercase', False),
        remove_stopwords=config.get('remove_stopwords', False),
        remove_punctuation=config.get('remove_punctuation', False),
        min_length=config.get('min_length', 0)
    )