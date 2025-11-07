#!/usr/bin/env python3
"""
Run baseline experiments for AER task
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
from typing import Optional, Dict, List, Any

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import AERDataLoader
from src.models.llm_wrapper import create_llm
from src.reasoning.abductive import AbductiveReasoner, ChainOfThoughtReasoner, FewShotReasoner
from src.prompting.prompt_builder import create_prompt_builder
from src.evaluation.metrics import calculate_score
from src.utils.logger import setup_logger
from src.utils.helpers import save_jsonl

logger = setup_logger("baseline", log_dir="outputs/logs")


class BaselineExperiment:
    """Baseline experiment runner"""
    
    def __init__(
        self,
        model_name: str = 'mock',
        strategy: str = 'zero_shot',
        data_dir: str = 'data/dev'
    ):
        """
        Initialize experiment
        
        Args:
            model_name: Model to use
            strategy: Reasoning strategy
            data_dir: Data directory
        """
        self.model_name = model_name
        self.strategy = strategy
        self.data_dir = data_dir
        
        # Load data
        logger.info(f"Loading data from {data_dir}")
        loader = AERDataLoader(data_dir)
        self.instances = loader.load()
        logger.info(f"Loaded {len(self.instances)} instances")
        
        # Create model
        logger.info(f"Creating model: {model_name}")
        self.model = create_llm(model_name, cache_responses=True)
        
        # Create reasoner
        logger.info(f"Creating reasoner: {strategy}")
        self.reasoner = self._create_reasoner(strategy)
    
    def _create_reasoner(self, strategy: str):
        """Create reasoner based on strategy"""
        if strategy == 'zero_shot':
            prompt_builder = create_prompt_builder('zero_shot')
            return AbductiveReasoner(self.model, prompt_builder)
        
        elif strategy == 'cot':
            return ChainOfThoughtReasoner(self.model)
        
        elif strategy == 'few_shot':
            # Load training data for examples
            train_loader = AERDataLoader('data/train')
            train_instances = train_loader.load()
            return FewShotReasoner(
                self.model,
                training_instances=train_instances,
                num_examples=3
            )
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def run(self, max_instances: Optional[int] = None) -> Dict:
        """
        Run experiment
        
        Args:
            max_instances: Limit number of instances (for testing)
            
        Returns:
            Results dictionary
        """
        logger.info("=" * 70)
        logger.info(f"Running Baseline Experiment")
        logger.info(f"Model: {self.model_name}, Strategy: {self.strategy}")
        logger.info("=" * 70)
        
        # Limit instances if requested
        instances = self.instances[:max_instances] if max_instances else self.instances
        logger.info(f"Processing {len(instances)} instances")
        
        # Run reasoning
        logger.info("Generating predictions...")
        results = self.reasoner.reason_batch(
            instances,
            include_documents=True,
            show_progress=True
        )
        
        # Evaluate
        logger.info("Evaluating predictions...")
        score, metrics = self._evaluate(instances, results)
        
        # Compile results
        experiment_results = {
            'model': self.model_name,
            'strategy': self.strategy,
            'data_dir': self.data_dir,
            'num_instances': len(instances),
            'score': score,
            'metrics': metrics,
            'predictions': results,
            'timestamp': datetime.now().isoformat()
        }
        
        return experiment_results
    
    def _evaluate(self, instances, results):
        """Evaluate predictions"""
        correct = 0
        partial = 0
        wrong = 0
        total_score = 0.0
        
        for instance, result in zip(instances, results):
            if not instance.has_answer():
                continue
            
            pred = result['prediction']
            gold = instance.golden_answer
            
            score = calculate_score(pred, gold)
            total_score += score
            
            if score == 1.0:
                correct += 1
            elif score == 0.5:
                partial += 1
            else:
                wrong += 1
        
        total = correct + partial + wrong
        
        metrics = {
            'total_instances': len(instances),
            'instances_with_answers': total,
            'correct': correct,
            'partial': partial,
            'wrong': wrong,
            'accuracy': total_score / total if total > 0 else 0,
            'exact_match': correct / total if total > 0 else 0,
            'partial_match': partial / total if total > 0 else 0
        }
        
        return total_score / total if total > 0 else 0, metrics
    
    def save_results(self, results: Dict, output_dir: str = 'outputs/results'):
        """Save experiment results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save full results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = output_path / f"baseline_{self.strategy}_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        
        # Save predictions in JSONL format
        predictions_file = output_path / f"predictions_{self.strategy}_{timestamp}.jsonl"
        save_jsonl(results['predictions'], str(predictions_file))
        logger.info(f"Predictions saved to {predictions_file}")
        
        return results_file


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run baseline experiments")
    parser.add_argument(
        '--model',
        type=str,
        default='mock',
        help='Model to use (mock, gpt-3.5-turbo, gpt-4, etc.)'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='zero_shot',
        choices=['zero_shot', 'cot', 'few_shot'],
        help='Reasoning strategy'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/dev',
        help='Data directory'
    )
    parser.add_argument(
        '--max-instances',
        type=int,
        default=None,
        help='Maximum instances to process (for testing)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/results',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    try:
        # Create experiment
        experiment = BaselineExperiment(
            model_name=args.model,
            strategy=args.strategy,
            data_dir=args.data_dir
        )
        
        # Run experiment
        results = experiment.run(max_instances=args.max_instances)
        
        # Print results
        print("\n" + "=" * 70)
        print("EXPERIMENT RESULTS")
        print("=" * 70)
        print(f"Model: {results['model']}")
        print(f"Strategy: {results['strategy']}")
        print(f"Instances: {results['num_instances']}")
        print(f"\nScore: {results['score']:.4f}")
        print(f"Exact Match: {results['metrics']['exact_match']:.4f}")
        print(f"Partial Match: {results['metrics']['partial_match']:.4f}")
        print(f"Correct: {results['metrics']['correct']}")
        print(f"Partial: {results['metrics']['partial']}")
        print(f"Wrong: {results['metrics']['wrong']}")
        print("=" * 70)
        
        # Save results
        output_file = experiment.save_results(results, args.output_dir)
        
        print(f"\n✅ Experiment complete!")
        print(f"Results saved to: {output_file}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error running experiment: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())