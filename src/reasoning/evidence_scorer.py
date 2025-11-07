"""
Evidence relevance scoring
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.data.loader import Document
from src.retrieval.embedder import SentenceEmbedder
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EvidenceScorer:
    """Score evidence relevance"""
    
    def __init__(self, embedder: SentenceEmbedder):
        """
        Initialize evidence scorer
        
        Args:
            embedder: Sentence embedder for similarity
        """
        self.embedder = embedder
        logger.info("Initialized EvidenceScorer")
    
    def score_documents(
        self,
        query: str,
        documents: List[Document],
        method: str = 'similarity'
    ) -> List[Tuple[Document, float]]:
        """
        Score documents by relevance to query
        
        Args:
            query: Query text (target event)
            documents: List of documents
            method: Scoring method ('similarity', 'keyword', 'combined')
            
        Returns:
            List of (document, score) tuples, sorted by score
        """
        if not documents:
            return []
        
        if method == 'similarity':
            scores = self._score_by_similarity(query, documents)
        elif method == 'keyword':
            scores = self._score_by_keywords(query, documents)
        elif method == 'combined':
            sim_scores = self._score_by_similarity(query, documents)
            kw_scores = self._score_by_keywords(query, documents)
            # Combine scores (weighted average)
            scores = [0.7 * s + 0.3 * k for s, k in zip(sim_scores, kw_scores)]
        else:
            raise ValueError(f"Unknown scoring method: {method}")
        
        # Pair documents with scores and sort
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return doc_scores
    
    def _score_by_similarity(
        self,
        query: str,
        documents: List[Document]
    ) -> List[float]:
        """Score by semantic similarity"""
        # Embed query
        query_embedding = self.embedder.embed_query(query)
        
        # Embed documents
        doc_texts = [doc.get_text() for doc in documents]
        doc_embeddings = self.embedder.embed(doc_texts)
        
        # Compute cosine similarities
        similarities = np.dot(doc_embeddings, query_embedding)
        
        return similarities.tolist()
    
    def _score_by_keywords(
        self,
        query: str,
        documents: List[Document]
    ) -> List[float]:
        """Score by keyword overlap"""
        import nltk
        from nltk.corpus import stopwords
        
        try:
            stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            stop_words = set(stopwords.words('english'))
        
        # Extract keywords from query
        query_words = set(query.lower().split())
        query_keywords = query_words - stop_words
        
        scores = []
        for doc in documents:
            doc_text = doc.get_text().lower()
            doc_words = set(doc_text.split())
            doc_keywords = doc_words - stop_words
            
            # Compute overlap
            if len(query_keywords) == 0:
                score = 0.0
            else:
                overlap = len(query_keywords & doc_keywords)
                score = overlap / len(query_keywords)
            
            scores.append(score)
        
        return scores
    
    def score_options(
        self,
        target_event: str,
        options: Dict[str, str],
        documents: List[Document]
    ) -> Dict[str, float]:
        """
        Score each option against evidence
        
        Args:
            target_event: Target event text
            options: Dictionary of options (A, B, C, D)
            documents: Context documents
            
        Returns:
            Dictionary of option scores
        """
        scores = {}
        
        for option_key, option_text in options.items():
            # Combine target event and option for query
            query = f"{target_event} {option_text}"
            
            # Score documents
            doc_scores = self.score_documents(query, documents)
            
            # Average top-k document scores
            k = min(3, len(doc_scores))
            if k > 0:
                avg_score = np.mean([score for _, score in doc_scores[:k]])
            else:
                avg_score = 0.0
            
            scores[option_key] = float(avg_score)
        
        return scores
    
    def rank_documents(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Document]:
        """
        Rank documents by relevance
        
        Args:
            query: Query text
            documents: List of documents
            top_k: Return only top k documents
            
        Returns:
            Ranked list of documents
        """
        doc_scores = self.score_documents(query, documents)
        ranked_docs = [doc for doc, _ in doc_scores]
        
        if top_k:
            return ranked_docs[:top_k]
        return ranked_docs