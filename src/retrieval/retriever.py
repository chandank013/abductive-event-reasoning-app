"""
Document retrieval system
"""

from typing import List, Dict, Tuple, Optional
import numpy as np

from src.retrieval.embedder import SentenceEmbedder
from src.retrieval.vector_store import FAISSVectorStore
from src.data.loader import Document, AERInstance
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentRetriever:
    """Document retrieval using embeddings and vector search"""
    
    def __init__(
        self,
        embedder: SentenceEmbedder,
        vector_store: Optional[FAISSVectorStore] = None
    ):
        """
        Initialize retriever
        
        Args:
            embedder: Embedding model
            vector_store: Vector store (optional, will create if None)
        """
        self.embedder = embedder
        
        if vector_store is None:
            embedding_dim = embedder.get_embedding_dim()
            self.vector_store = FAISSVectorStore(
                embedding_dim=embedding_dim,
                index_type="flat",
                metric="cosine"
            )
        else:
            self.vector_store = vector_store
        
        logger.info("Document retriever initialized")
    
    def index_documents(
        self,
        documents: List[Document],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> None:
        """
        Index documents for retrieval
        
        Args:
            documents: List of Document objects
            batch_size: Batch size for embedding
            show_progress: Show progress bar
        """
        logger.info(f"Indexing {len(documents)} documents...")
        
        # Extract texts
        texts = [doc.get_text() for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedder.embed_documents(
            texts,
            batch_size=batch_size,
            show_progress=show_progress
        )
        
        # Prepare metadata
        metadata = [
            {
                'title': doc.title,
                'source': doc.source,
                'uuid': doc.uuid,
                'link': doc.link
            }
            for doc in documents
        ]
        
        # Add to vector store
        self.vector_store.add(
            embeddings=embeddings,
            documents=texts,
            metadata=metadata
        )
        
        logger.info(f"Indexed {len(documents)} documents")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> List[Dict]:
        """
        Retrieve relevant documents
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of retrieved documents with scores
        """
        # Embed query
        query_embedding = self.embedder.embed_query(query)
        
        # Search
        distances, indices, docs, metadata = self.vector_store.search(
            query_embedding,
            k=top_k,
            return_documents=True,
            return_metadata=True
        )
        
        # Format results
        results = []
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            if dist >= min_similarity:  # For cosine similarity
                result = {
                    'rank': i + 1,
                    'score': float(dist),
                    'index': int(idx),
                    'content': docs[i] if docs else None,
                    'metadata': metadata[i] if metadata else {}
                }
                results.append(result)
        
        return results
    
    def retrieve_for_instance(
        self,
        instance: AERInstance,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Retrieve documents for an AER instance
        
        Args:
            instance: AER instance
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        # Use target event as query
        query = instance.target_event
        return self.retrieve(query, top_k=top_k)
    
    def save(self, path: str) -> None:
        """Save vector store"""
        self.vector_store.save(path)
        logger.info(f"Retriever saved to {path}")
    
    def load(self, path: str) -> None:
        """Load vector store"""
        self.vector_store.load(path)
        logger.info(f"Retriever loaded from {path}")
    
    def get_stats(self) -> Dict:
        """Get retriever statistics"""
        return {
            'total_documents': self.vector_store.get_size(),
            'embedding_dim': self.embedder.get_embedding_dim(),
            'model_name': self.embedder.model_name
        }


class BM25Retriever:
    """BM25-based retrieval (alternative/complement to dense retrieval)"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever
        
        Args:
            k1: BM25 k1 parameter
            b: BM25 b parameter
        """
        from rank_bm25 import BM25Okapi
        import nltk
        from nltk.tokenize import word_tokenize
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.documents = []
        self.tokenize = word_tokenize
        
        logger.info("BM25 retriever initialized")
    
    def index_documents(self, documents: List[Document]) -> None:
        """Index documents"""
        from rank_bm25 import BM25Okapi
        
        self.documents = documents
        texts = [doc.get_text() for doc in documents]
        
        # Tokenize
        tokenized_corpus = [self.tokenize(text.lower()) for text in texts]
        
        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        
        logger.info(f"Indexed {len(documents)} documents with BM25")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve documents"""
        # Tokenize query
        tokenized_query = self.tokenize(query.lower())
        
        # Get scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Format results
        results = []
        for rank, idx in enumerate(top_indices):
            result = {
                'rank': rank + 1,
                'score': float(scores[idx]),
                'index': int(idx),
                'content': self.documents[idx].get_text(),
                'metadata': {
                    'title': self.documents[idx].title,
                    'source': self.documents[idx].source
                }
            }
            results.append(result)
        
        return results