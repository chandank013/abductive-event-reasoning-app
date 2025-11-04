"""
Document embedding using Sentence-BERT and other embedding models
"""

import numpy as np
from typing import List, Optional, Union
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SentenceEmbedder:
    """Sentence/Document embedder using Sentence-BERT"""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize embedder
        
        Args:
            model_name: Hugging Face model name
            device: Device to use ('cuda', 'cpu', or None for auto)
            cache_dir: Directory to cache models
        """
        self.model_name = model_name
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Loading embedding model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load model
        try:
            self.model = SentenceTransformer(
                model_name,
                device=self.device,
                cache_folder=cache_dir
            )
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Embed texts
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            normalize: Normalize embeddings to unit length
            
        Returns:
            numpy array of embeddings (n_texts, embedding_dim)
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        # Filter empty texts
        texts = [t if t else " " for t in texts]
        
        try:
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=normalize
            )
            
            return embeddings
        
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Embed a single query
        
        Args:
            query: Query text
            normalize: Normalize embedding
            
        Returns:
            numpy array of embedding (embedding_dim,)
        """
        embedding = self.embed(query, normalize=normalize)
        return embedding[0]  # Return single embedding
    
    def embed_documents(
        self,
        documents: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Embed multiple documents
        
        Args:
            documents: List of document texts
            batch_size: Batch size
            show_progress: Show progress bar
            
        Returns:
            numpy array of embeddings
        """
        return self.embed(
            documents,
            batch_size=batch_size,
            show_progress=show_progress
        )
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.embedding_dim
    
    def save_model(self, path: str) -> None:
        """Save model to disk"""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model from disk"""
        self.model = SentenceTransformer(path, device=self.device)
        logger.info(f"Model loaded from {path}")


class SimpleEmbedder:
    """Simple embedder using basic methods (fallback)"""
    
    def __init__(self, method: str = "tfidf"):
        """
        Initialize simple embedder
        
        Args:
            method: 'tfidf' or 'count'
        """
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        
        self.method = method
        
        if method == "tfidf":
            self.vectorizer = TfidfVectorizer(max_features=384)
        else:
            self.vectorizer = CountVectorizer(max_features=384)
        
        self.is_fitted = False
        logger.info(f"Initialized simple embedder with method: {method}")
    
    def fit(self, texts: List[str]) -> None:
        """Fit vectorizer on texts"""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        logger.info(f"Vectorizer fitted on {len(texts)} texts")
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Embed texts"""
        if isinstance(texts, str):
            texts = [texts]
        
        if not self.is_fitted:
            logger.warning("Vectorizer not fitted. Fitting on input texts.")
            self.fit(texts)
        
        embeddings = self.vectorizer.transform(texts).toarray()
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.vectorizer.max_features


def create_embedder(
    model_type: str = "sentence-bert",
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    device: Optional[str] = None
):
    """
    Factory function to create embedder
    
    Args:
        model_type: 'sentence-bert' or 'simple'
        model_name: Model name (for sentence-bert)
        device: Device to use
        
    Returns:
        Embedder instance
    """
    if model_type == "sentence-bert":
        return SentenceEmbedder(model_name=model_name, device=device)
    elif model_type == "simple":
        return SimpleEmbedder(method="tfidf")
    else:
        raise ValueError(f"Unknown model type: {model_type}")