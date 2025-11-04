#!/usr/bin/env python3
"""
Build embeddings for documents
"""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import AERDataLoader
from src.retrieval.embedder import SentenceEmbedder
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.retriever import DocumentRetriever
from src.utils.logger import setup_logger

logger = setup_logger("build_embeddings", log_dir="outputs/logs")


def build_embeddings(data_dir: str, output_dir: str, model_name: str, batch_size: int):
    """
    Build embeddings for a dataset
    
    Args:
        data_dir: Directory containing data
        output_dir: Output directory for embeddings
        model_name: Embedding model name
        batch_size: Batch size for embedding
    """
    logger.info("=" * 70)
    logger.info("Building Document Embeddings")
    logger.info("=" * 70)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model: {model_name}")
    
    # Load data
    logger.info("\n📥 Loading data...")
    loader = AERDataLoader(data_dir)
    instances = loader.load()
    logger.info(f"Loaded {len(instances)} instances")
    
    # Collect all unique documents
    logger.info("\n📄 Collecting documents...")
    doc_map = {}  # uuid -> Document
    
    for inst in instances:
        for doc in inst.docs:
            if doc.uuid and doc.uuid not in doc_map:
                doc_map[doc.uuid] = doc
    
    documents = list(doc_map.values())
    logger.info(f"Found {len(documents)} unique documents")
    
    # Initialize embedder
    logger.info(f"\n🤖 Initializing embedder: {model_name}")
    embedder = SentenceEmbedder(model_name=model_name)
    
    # Initialize retriever
    logger.info("\n🔍 Building retrieval index...")
    retriever = DocumentRetriever(embedder=embedder)
    
    # Index documents
    retriever.index_documents(
        documents,
        batch_size=batch_size,
        show_progress=True
    )
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n💾 Saving embeddings to {output_path}...")
    retriever.save(str(output_path))
    
    # Test retrieval
    logger.info("\n🧪 Testing retrieval...")
    test_query = instances[0].target_event if instances else "test query"
    results = retriever.retrieve(test_query, top_k=3)
    
    logger.info(f"Test query: {test_query}")
    logger.info("Top 3 results:")
    for result in results:
        logger.info(f"  - Score: {result['score']:.4f}, Title: {result['metadata'].get('title', 'N/A')[:50]}...")
    
    # Print stats
    stats = retriever.get_stats()
    logger.info("\n📊 Statistics:")
    logger.info(f"  Total documents indexed: {stats['total_documents']}")
    logger.info(f"  Embedding dimension: {stats['embedding_dim']}")
    logger.info(f"  Model: {stats['model_name']}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ Embeddings built successfully!")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Build document embeddings")
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/train',
        help='Data directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/embeddings/train',
        help='Output directory'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='sentence-transformers/all-mpnet-base-v2',
        help='Embedding model name'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    
    args = parser.parse_args()
    
    try:
        build_embeddings(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model_name=args.model_name,
            batch_size=args.batch_size
        )
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()