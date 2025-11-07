"""
Evaluation metrics for AER task
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import Counter

from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_answer(answer: str) -> set:
    """
    Parse answer string to set of options
    
    Args:
        answer: Answer string like "A", "B,C", "A,D"
        
    Returns:
        Set of option letters
    """
    if not answer:
        return set()
    return set(opt.strip() for opt in answer.split(','))


def calculate_score(prediction: str, gold: str) -> float:
    """
    Calculate score for a single prediction
    
    Scoring:
    - Exact match: 1.0
    - Partial match: 0.5
    - Wrong: 0.0
    
    Args:
        prediction: Predicted answer
        gold: Gold answer
        
    Returns:
        Score (0.0, 0.5, or 1.0)
    """
    pred_set = parse_answer(prediction)
    gold_set = parse_answer(gold)
    
    if pred_set == gold_set:
        return 1.0
    elif len(pred_set & gold_set) > 0:
        return 0.5
    else:
        return 0.0


def calculate_accuracy(predictions: List[str], gold_answers: List[str]) -> float:
    """
    Calculate overall accuracy
    
    Args:
        predictions: List of predictions
        gold_answers: List of gold answers
        
    Returns:
        Accuracy score
    """
    if len(predictions) != len(gold_answers):
        raise ValueError("Predictions and gold answers must have same length")
    
    total_score = sum(
        calculate_score(pred, gold)
        for pred, gold in zip(predictions, gold_answers)
    )
    
    return total_score / len(predictions) if predictions else 0.0


def calculate_exact_match(predictions: List[str], gold_answers: List[str]) -> float:
    """
    Calculate exact match accuracy
    
    Args:
        predictions: List of predictions
        gold_answers: List of gold answers
        
    Returns:
        Exact match accuracy
    """
    if len(predictions) != len(gold_answers):
        raise ValueError("Predictions and gold answers must have same length")
    
    exact_matches = sum(
        1 for pred, gold in zip(predictions, gold_answers)
        if parse_answer(pred) == parse_answer(gold)
    )
    
    return exact_matches / len(predictions) if predictions else 0.0


def calculate_partial_match(predictions: List[str], gold_answers: List[str]) -> float:
    """
    Calculate partial match rate
    
    Args:
        predictions: List of predictions
        gold_answers: List of gold answers
        
    Returns:
        Partial match rate
    """
    if len(predictions) != len(gold_answers):
        raise ValueError("Predictions and gold answers must have same length")
    
    partial_matches = 0
    for pred, gold in zip(predictions, gold_answers):
        pred_set = parse_answer(pred)
        gold_set = parse_answer(gold)
        
        # Partial match: some overlap but not exact
        if pred_set != gold_set and len(pred_set & gold_set) > 0:
            partial_matches += 1
    
    return partial_matches / len(predictions) if predictions else 0.0


def calculate_option_metrics(
    predictions: List[str],
    gold_answers: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate per-option precision, recall, F1
    
    Args:
        predictions: List of predictions
        gold_answers: List of gold answers
        
    Returns:
        Dictionary of metrics per option
    """
    options = ['A', 'B', 'C', 'D']
    metrics = {}
    
    for option in options:
        tp = 0  # True positives
        fp = 0  # False positives
        fn = 0  # False negatives
        
        for pred, gold in zip(predictions, gold_answers):
            pred_set = parse_answer(pred)
            gold_set = parse_answer(gold)
            
            if option in pred_set and option in gold_set:
                tp += 1
            elif option in pred_set and option not in gold_set:
                fp += 1
            elif option not in pred_set and option in gold_set:
                fn += 1
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[option] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': tp + fn
        }
    
    return metrics


def calculate_confusion_matrix(
    predictions: List[str],
    gold_answers: List[str]
) -> Dict:
    """
    Calculate confusion matrix for answer types
    
    Args:
        predictions: List of predictions
        gold_answers: List of gold answers
        
    Returns:
        Confusion matrix data
    """
    # Categorize answers
    def categorize(answer: str) -> str:
        opts = parse_answer(answer)
        if not opts:
            return "empty"
        elif len(opts) == 1:
            return f"single_{list(opts)[0]}"
        else:
            return "multiple"
    
    gold_categories = [categorize(g) for g in gold_answers]
    pred_categories = [categorize(p) for p in predictions]
    
    # Count combinations
    matrix = {}
    for gold_cat, pred_cat in zip(gold_categories, pred_categories):
        key = f"{gold_cat}_to_{pred_cat}"
        matrix[key] = matrix.get(key, 0) + 1
    
    return matrix


def calculate_all_metrics(
    predictions: List[str],
    gold_answers: List[str]
) -> Dict:
    """
    Calculate all evaluation metrics
    
    Args:
        predictions: List of predictions
        gold_answers: List of gold answers
        
    Returns:
        Dictionary with all metrics
    """
    logger.info(f"Calculating metrics for {len(predictions)} predictions")
    
    metrics = {
        'num_predictions': len(predictions),
        'accuracy': calculate_accuracy(predictions, gold_answers),
        'exact_match': calculate_exact_match(predictions, gold_answers),
        'partial_match': calculate_partial_match(predictions, gold_answers),
        'per_option': calculate_option_metrics(predictions, gold_answers),
        'confusion_matrix': calculate_confusion_matrix(predictions, gold_answers)
    }
    
    # Calculate answer distribution
    pred_dist = Counter(predictions)
    gold_dist = Counter(gold_answers)
    
    metrics['prediction_distribution'] = dict(pred_dist)
    metrics['gold_distribution'] = dict(gold_dist)
    
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Exact Match: {metrics['exact_match']:.4f}")
    logger.info(f"Partial Match: {metrics['partial_match']:.4f}")
    
    return metrics