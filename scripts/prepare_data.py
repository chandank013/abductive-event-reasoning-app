#!/usr/bin/env python3
"""
Prepare and split AER dataset into train/dev/test sets
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from pathlib import Path
from typing import List, Tuple
import random
from collections import defaultdict

from src.data.loder import AERDataLoader, AERInstance
from src.data.preprocessor import TextPreprocessor
from src.utils.logger import setup_logger
from src.utils.helpers import save_jsonl

logger = setup_logger("prepare_data", log_dir="outputs/logs")


def split_data(
    instances: List[AERInstance],
    train_ratio: float = 0.7,
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[AERInstance], List[AERInstance], List[AERInstance]]:
    """
    Split data into train/dev/test sets
    
    Args:
        instances: List of AER instances
        train_ratio: Ratio for training set
        dev_ratio: Ratio for development set
        test_ratio: Ratio for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train, dev, test) instances
    """
    random.seed(random_seed)
    
    # Shuffle instances
    shuffled = instances.copy()
    random.shuffle(shuffled)
    
    # Calculate split indices
    n = len(shuffled)
    train_end = int(n * train_ratio)
    dev_end = train_end + int(n * dev_ratio)
    
    train = shuffled[:train_end]
    dev = shuffled[train_end:dev_end]
    test = shuffled[dev_end:]
    
    logger.info(f"Split data: Train={len(train)}, Dev={len(dev)}, Test={len(test)}")
    
    return train, dev, test


def save_split(
    instances: List[AERInstance],
    output_dir: Path,
    split_name: str
) -> None:
    """
    Save data split to files
    
    Args:
        instances: List of instances
        output_dir: Output directory
        split_name: Name of split (train/dev/test)
    """
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare questions data
    questions_data = []
    for inst in instances:
        questions_data.append({
            'topic_id': inst.topic_id,
            'question': inst.question,
            'option_A': inst.option_A,
            'option_B': inst.option_B,
            'option_C': inst.option_C,
            'option_D': inst.option_D,
            'answer': inst.answer
        })
    
    # Prepare docs data
    docs_data = []
    for inst in instances:
        docs_data.append({
            'topic_id': inst.topic_id,
            'docs': inst.docs
        })
    
    # Save files
    questions_file = split_dir / "questions.jsonl"
    docs_file = split_dir / "docs.jsonl"
    
    save_jsonl(questions_data, str(questions_file))
    save_jsonl(docs_data, str(docs_file))
    
    logger.info(f"Saved {split_name} split to {split_dir}")
    logger.info(f"  - {questions_file.name}: {len(questions_data)} questions")
    logger.info(f"  - {docs_file.name}: {len(docs_data)} document sets")


def preprocess_instances(
    instances: List[AERInstance],
    preprocessor: TextPreprocessor
) -> List[AERInstance]:
    """
    Apply preprocessing to instances
    
    Args:
        instances: List of instances
        preprocessor: Text preprocessor
        
    Returns:
        List of preprocessed instances
    """
    logger.info("Preprocessing instances...")
    
    processed = []
    for inst in instances:
        # Preprocess question
        question = preprocessor.clean_text(inst.question)
        
        # Preprocess options
        option_A = preprocessor.clean_text(inst.option_A)
        option_B = preprocessor.clean_text(inst.option_B)
        option_C = preprocessor.clean_text(inst.option_C)
        option_D = preprocessor.clean_text(inst.option_D)
        
        # Preprocess documents
        docs = preprocessor.preprocess_documents(inst.docs)
        
        # Create new instance
        processed_inst = AERInstance(
            topic_id=inst.topic_id,
            question=question,
            option_A=option_A,
            option_B=option_B,
            option_C=option_C,
            option_D=option_D,
            answer=inst.answer,
            docs=docs
        )
        processed.append(processed_inst)
    
    return processed


def analyze_splits(
    train: List[AERInstance],
    dev: List[AERInstance],
    test: List[AERInstance]
) -> None:
    """Print analysis of data splits"""
    
    def analyze_split(instances: List[AERInstance], name: str):
        logger.info(f"\n{name} Set Statistics:")
        logger.info(f"  Total instances: {len(instances)}")
        
        # Answer distribution
        answer_dist = defaultdict(int)
        for inst in instances:
            answer_dist[inst.answer] += 1
        
        logger.info("  Answer distribution:")
        for answer, count in sorted(answer_dist.items()):
            pct = (count / len(instances)) * 100
            logger.info(f"    {answer}: {count} ({pct:.1f}%)")
        
        # Document statistics
        total_docs = sum(len(inst.docs) for inst in instances)
        avg_docs = total_docs / len(instances)
        logger.info(f"  Average docs per instance: {avg_docs:.1f}")
    
    analyze_split(train, "Train")
    analyze_split(dev, "Dev")
    analyze_split(test, "Test")


def main():
    parser = argparse.ArgumentParser(description="Prepare AER dataset")
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/raw',
        help='Input directory with raw data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training set ratio'
    )
    parser.add_argument(
        '--dev-ratio',
        type=float,
        default=0.15,
        help='Development set ratio'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Test set ratio'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--preprocess',
        action='store_true',
        help='Apply text preprocessing'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("AER Data Preparation")
    logger.info("=" * 60)
    
    # Load raw data
    logger.info(f"\nLoading data from {args.input_dir}...")
    loader = AERDataLoader(args.input_dir)
    instances = loader.load()
    
    # Get statistics
    stats = loader.get_statistics()
    logger.info("\nRaw Data Statistics:")
    logger.info(f"  Total instances: {stats['total_instances']}")
    logger.info(f"  Total documents: {stats['total_documents']}")
    logger.info(f"  Avg docs per instance: {stats['avg_docs_per_instance']:.1f}")
    
    # Preprocessing
    if args.preprocess:
        preprocessor = TextPreprocessor(
            lowercase=False,
            remove_stopwords=False,
            remove_punctuation=False
        )
        instances = preprocess_instances(instances, preprocessor)
        logger.info("✓ Preprocessing complete")
    
    # Split data
    logger.info("\nSplitting data...")
    train, dev, test = split_data(
        instances,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed
    )
    
    # Analyze splits
    analyze_splits(train, dev, test)
    
    # Save splits
    output_dir = Path(args.output_dir)
    logger.info(f"\nSaving splits to {output_dir}...")
    
    save_split(train, output_dir, "train")
    save_split(dev, output_dir, "dev")
    save_split(test, output_dir, "test")
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Data preparation complete!")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("  1. Run: python scripts/analyze_data.py")
    logger.info("  2. Run: python scripts/build_embeddings.py")


if __name__ == "__main__":
    main()