"""
Vector storage and similarity search using FAISS
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import pickle
import faiss

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FAISSVectorStore:
    """FAISS-based vector store for efficient similarity search"""
    
    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "flat",
        metric: str = "cosine"
    ):
        """
        Initialize FAISS vector store
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: 'flat' (exact) or 'ivf' (approximate)
            metric: 'cosine' or 'l2'
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric = metric
        
        # Create index
        if metric == "cosine":
            # For cosine similarity, use inner product with normalized vectors
            if index_type == "flat":
                self.index = faiss.IndexFlatIP(embedding_dim)
            else:
                # IVF index for large datasets
                quantizer = faiss.IndexFlatIP(embedding_dim)
                self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, 100)
        else:
            # L2 distance
            if index_type == "flat":
                self.index = faiss.IndexFlatL2(embedding_dim)
            else:
                quantizer = faiss.IndexFlatL2(embedding_dim)
                self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, 100)
        
        self.documents = []  # Store original documents
        self.metadata = []   # Store metadata
        
        logger.info(f"Initialized FAISS index: type={index_type}, metric={metric}, dim={embedding_dim}")
    
    def add(
        self,
        embeddings: np.ndarray,
        documents: Optional[List[str]] = None,
        metadata: Optional[List[Dict]] = None
    ) -> None:
        """
        Add embeddings to the index
        
        Args:
            embeddings: numpy array of shape (n, embedding_dim)
            documents: List of document texts
            metadata: List of metadata dictionaries
        """
        # Ensure embeddings are float32
        embeddings = embeddings.astype('float32')
        
        # Normalize for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(embeddings)
        
        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store documents and metadata
        if documents:
            self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        
        logger.info(f"Added {len(embeddings)} vectors. Total: {self.index.ntotal}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        return_documents: bool = True,
        return_metadata: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]], Optional[List[Dict]]]:
        """
        Search for similar vectors
        
        Args:
            query_embedding: Query embedding (embedding_dim,)
            k: Number of results to return
            return_documents: Return document texts
            return_metadata: Return metadata
            
        Returns:
            Tuple of (distances, indices, documents, metadata)
        """
        # Reshape query
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Normalize for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Get documents and metadata
        docs = None
        meta = None
        
        if return_documents and self.documents:
            docs = [self.documents[i] for i in indices[0] if i < len(self.documents)]
        
        if return_metadata and self.metadata:
            meta = [self.metadata[i] for i in indices[0] if i < len(self.metadata)]
        
        return distances[0], indices[0], docs, meta
    
    def batch_search(
        self,
        query_embeddings: np.ndarray,
        k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch search for multiple queries
        
        Args:
            query_embeddings: Query embeddings (n_queries, embedding_dim)
            k: Number of results per query
            
        Returns:
            Tuple of (distances, indices)
        """
        query_embeddings = query_embeddings.astype('float32')
        
        if self.metric == "cosine":
            faiss.normalize_L2(query_embeddings)
        
        distances, indices = self.index.search(query_embeddings, k)
        return distances, indices
    
    def save(self, path: str) -> None:
        """
        Save index to disk
        
        Args:
            path: Directory to save index
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save documents and metadata
        with open(path / "documents.pkl", 'wb') as f:
            pickle.dump(self.documents, f)
        
        with open(path / "metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # Save config
        config = {
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'metric': self.metric
        }
        with open(path / "config.pkl", 'wb') as f:
            pickle.dump(config, f)
        
        logger.info(f"Vector store saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load index from disk
        
        Args:
            path: Directory containing saved index
        """
        path = Path(path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(path / "index.faiss"))
        
        # Load documents and metadata
        with open(path / "documents.pkl", 'rb') as f:
            self.documents = pickle.load(f)
        
        with open(path / "metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Load config
        with open(path / "config.pkl", 'rb') as f:
            config = pickle.load(f)
            self.embedding_dim = config['embedding_dim']
            self.index_type = config['index_type']
            self.metric = config['metric']
        
        logger.info(f"Vector store loaded from {path}. Total vectors: {self.index.ntotal}")
    
    def get_size(self) -> int:
        """Get number of vectors in index"""
        return self.index.ntotal
    
    def clear(self) -> None:
        """Clear the index"""
        self.index.reset()
        self.documents = []
        self.metadata = []
        logger.info("Vector store cleared")


class SimpleVectorStore:
    """Simple in-memory vector store using numpy (fallback)"""
    
    def __init__(self, embedding_dim: int, metric: str = "cosine"):
        """
        Initialize simple vector store
        
        Args:
            embedding_dim: Embedding dimension
            metric: 'cosine' or 'l2'
        """
        self.embedding_dim = embedding_dim
        self.metric = metric
        self.vectors = None
        self.documents = []
        self.metadata = []
        
        logger.info(f"Initialized simple vector store: dim={embedding_dim}, metric={metric}")
    
    def add(
        self,
        embeddings: np.ndarray,
        documents: Optional[List[str]] = None,
        metadata: Optional[List[Dict]] = None
    ) -> None:
        """Add embeddings"""
        if self.vectors is None:
            self.vectors = embeddings
        else:
            self.vectors = np.vstack([self.vectors, embeddings])
        
        if documents:
            self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        
        logger.info(f"Added {len(embeddings)} vectors. Total: {len(self.vectors)}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        return_documents: bool = True,
        return_metadata: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]], Optional[List[Dict]]]:
        """Search for similar vectors"""
        if self.metric == "cosine":
            # Cosine similarity
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            vectors_norm = self.vectors / np.linalg.norm(self.vectors, axis=1, keepdims=True)
            similarities = np.dot(vectors_norm, query_norm)
            indices = np.argsort(similarities)[::-1][:k]
            distances = similarities[indices]
        else:
            # L2 distance
            distances = np.linalg.norm(self.vectors - query_embedding, axis=1)
            indices = np.argsort(distances)[:k]
            distances = distances[indices]
        
        docs = [self.documents[i] for i in indices] if return_documents else None
        meta = [self.metadata[i] for i in indices] if return_metadata else None
        
        return distances, indices, docs, meta
    
    def save(self, path: str) -> None:
        """Save to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        np.save(path / "vectors.npy", self.vectors)
        with open(path / "documents.pkl", 'wb') as f:
            pickle.dump(self.documents, f)
        with open(path / "metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata, f)
        
        logger.info(f"Vector store saved to {path}")
    
    def load(self, path: str) -> None:
        """Load from disk"""
        path = Path(path)
        
        self.vectors = np.load(path / "vectors.npy")
        with open(path / "documents.pkl", 'rb') as f:
            self.documents = pickle.load(f)
        with open(path / "metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)
        
        logger.info(f"Vector store loaded from {path}")
    
    def get_size(self) -> int:
        """Get size"""
        return len(self.vectors) if self.vectors is not None else 0


def create_vector_store(
    embedding_dim: int,
    store_type: str = "faiss",
    **kwargs
):
    """
    Factory function to create vector store
    
    Args:
        embedding_dim: Embedding dimension
        store_type: 'faiss' or 'simple'
        **kwargs: Additional arguments
        
    Returns:
        Vector store instance
    """
    if store_type == "faiss":
        return FAISSVectorStore(embedding_dim, **kwargs)
    elif store_type == "simple":
        return SimpleVectorStore(embedding_dim, **kwargs)
    else:
        raise ValueError(f"Unknown store type: {store_type}")