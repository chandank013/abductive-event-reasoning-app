"""
Evaluation orchestration
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from src.data.loader import AERDataLoader, AERInstance
from src.evaluation.metrics import calculate_all_metrics
from src.utils.logger import get_logger
from src.utils.helpers import load_jsonl, save_json

logger = get_logger(__name__)


class Evaluator:
    """Evaluate predictions against ground truth"""
    
    def __init__(self):
        """Initialize evaluator"""
        logger.info("Initialized Evaluator")
    
    def evaluate_predictions(
        self,
        predictions: List[Dict],
        ground_truth: List[AERInstance]
    ) -> Dict:
        """
        Evaluate predictions
        
        Args:
            predictions: List of prediction dictionaries with 'uuid' and 'prediction'
            ground_truth: List of ground truth instances
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating {len(predictions)} predictions")
        
        # Create lookup for ground truth
        gt_lookup = {inst.uuid: inst for inst in ground_truth}
        
        # Match predictions to ground truth
        matched_preds = []
        matched_golds = []
        missing_gt = []
        
        for pred in predictions:
            uuid = pred['uuid']
            if uuid in gt_lookup:
                gt_inst = gt_lookup[uuid]
                if gt_inst.has_answer():
                    matched_preds.append(pred['prediction'])
                    matched_golds.append(gt_inst.golden_answer)
            else:
                missing_gt.append(uuid)
        
        if missing_gt:
            logger.warning(f"Missing ground truth for {len(missing_gt)} predictions")
        
        # Calculate metrics
        metrics = calculate_all_metrics(matched_preds, matched_golds)
        
        # Add metadata
        results = {
            'evaluation_date': datetime.now().isoformat(),
            'num_predictions': len(predictions),
            'num_matched': len(matched_preds),
            'num_missing_gt': len(missing_gt),
            'metrics': metrics
        }
        
        return results
    
    def evaluate_from_files(
        self,
        predictions_file: str,
        ground_truth_dir: str
    ) -> Dict:
        """
        Evaluate from prediction file and ground truth directory
        
        Args:
            predictions_file: Path to predictions JSONL file
            ground_truth_dir: Path to ground truth data directory
            
        Returns:
            Evaluation results
        """
        logger.info(f"Loading predictions from {predictions_file}")
        predictions = load_jsonl(predictions_file)
        
        logger.info(f"Loading ground truth from {ground_truth_dir}")
        loader = AERDataLoader(ground_truth_dir)
        ground_truth = loader.load()
        
        return self.evaluate_predictions(predictions, ground_truth)
    
    def save_results(
        self,
        results: Dict,
        output_file: str
    ) -> None:
        """
        Save evaluation results
        
        Args:
            results: Evaluation results
            output_file: Output file path
        """
        save_json(results, output_file)
        logger.info(f"Evaluation results saved to {output_file}")
    
    def print_summary(self, results: Dict) -> None:
        """
        Print evaluation summary
        
        Args:
            results: Evaluation results
        """
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        
        metrics = results['metrics']
        
        print(f"\nPredictions: {results['num_predictions']}")
        print(f"Matched with GT: {results['num_matched']}")
        
        print(f"\n📊 Overall Metrics:")
        print(f"  Accuracy:      {metrics['accuracy']:.4f}")
        print(f"  Exact Match:   {metrics['exact_match']:.4f}")
        print(f"  Partial Match: {metrics['partial_match']:.4f}")
        
        print(f"\n📈 Per-Option Metrics:")
        for option, opt_metrics in metrics['per_option'].items():
            print(f"  Option {option}:")
            print(f"    Precision: {opt_metrics['precision']:.4f}")
            print(f"    Recall:    {opt_metrics['recall']:.4f}")
            print(f"    F1:        {opt_metrics['f1']:.4f}")
            print(f"    Support:   {opt_metrics['support']}")
        
        print(f"\n📋 Prediction Distribution:")
        for answer, count in sorted(metrics['prediction_distribution'].items()):
            pct = count / results['num_matched'] * 100
            print(f"  {answer}: {count} ({pct:.1f}%)")
        
        print("=" * 70)