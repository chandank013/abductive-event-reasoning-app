#!/usr/bin/env python3
"""
Complete pipeline: Data → Model → Evaluation
Demonstrates end-to-end workflow
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import AERDataLoader
from src.models.llm_wrapper import create_llm
from src.reasoning.abductive import AbductiveReasoner, ChainOfThoughtReasoner
from src.evaluation.evaluator import Evaluator
from src.evaluation.error_analysis import ErrorAnalyzer
from src.utils.logger import setup_logger
from src.utils.helpers import save_jsonl, save_json

logger = setup_logger("pipeline", log_dir="outputs/logs")


def run_pipeline(
    data_dir: str = 'data/sample',
    model_name: str = 'mock',
    strategy: str = 'zero_shot',
    max_instances: int = None,
    output_dir: str = 'outputs/pipeline'
):
    """
    Run complete pipeline
    
    Args:
        data_dir: Data directory
        model_name: Model to use
        strategy: Reasoning strategy
        max_instances: Limit instances
        output_dir: Output directory
    """
    print("=" * 70)
    print("AER COMPLETE PIPELINE")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Data: {data_dir}")
    print(f"  Model: {model_name}")
    print(f"  Strategy: {strategy}")
    print(f"  Max Instances: {max_instances or 'All'}")
    print(f"  Output: {output_dir}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load Data
    print("\n" + "=" * 70)
    print("STEP 1: Loading Data")
    print("=" * 70)
    
    loader = AERDataLoader(data_dir)
    instances = loader.load()
    print(f"✓ Loaded {len(instances)} instances")
    
    if max_instances:
        instances = instances[:max_instances]
        print(f"✓ Limited to {len(instances)} instances")
    
    # Get statistics
    stats = loader.get_statistics()
    print(f"✓ Instances with answers: {stats['instances_with_answers']}")
    
    # Step 2: Initialize Model
    print("\n" + "=" * 70)
    print("STEP 2: Initializing Model")
    print("=" * 70)
    
    model = create_llm(model_name, cache_responses=True)
    print(f"✓ Model initialized: {model.model_name}")
    print(f"✓ Model available: {model.available}")
    
    # Step 3: Create Reasoner
    print("\n" + "=" * 70)
    print("STEP 3: Creating Reasoner")
    print("=" * 70)
    
    if strategy == 'cot':
        reasoner = ChainOfThoughtReasoner(model)
    else:
        reasoner = AbductiveReasoner(model)
    
    print(f"✓ Reasoner created: {strategy}")
    
    # Step 4: Generate Predictions
    print("\n" + "=" * 70)
    print("STEP 4: Generating Predictions")
    print("=" * 70)
    
    start_time = datetime.now()
    predictions = reasoner.reason_batch(
        instances,
        include_documents=True,
        show_progress=True
    )
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"✓ Generated {len(predictions)} predictions")
    print(f"✓ Time: {elapsed:.2f}s ({elapsed/len(predictions):.2f}s per instance)")
    
    # Save predictions
    pred_file = output_path / f'predictions_{strategy}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'
    save_jsonl(predictions, str(pred_file))
    print(f"✓ Predictions saved: {pred_file}")
    
    # Step 5: Evaluate
    print("\n" + "=" * 70)
    print("STEP 5: Evaluating Predictions")
    print("=" * 70)
    
    evaluator = Evaluator()
    eval_results = evaluator.evaluate_predictions(predictions, instances)
    
    print(f"✓ Evaluated {eval_results['num_matched']} predictions")
    
    # Save evaluation results
    eval_file = output_path / f'evaluation_{strategy}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    save_json(eval_results, str(eval_file))
    print(f"✓ Evaluation saved: {eval_file}")
    
    # Print summary
    evaluator.print_summary(eval_results)
    
    # Step 6: Error Analysis
    print("\n" + "=" * 70)
    print("STEP 6: Error Analysis")
    print("=" * 70)
    
    analyzer = ErrorAnalyzer()
    error_analysis = analyzer.analyze(instances, predictions)
    
    # Save error analysis
    error_file = output_path / f'error_analysis_{strategy}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    save_json(error_analysis, str(error_file))
    print(f"✓ Error analysis saved: {error_file}")
    
    # Print error analysis
    analyzer.print_analysis(error_analysis)
    
    # Step 7: Model Statistics
    print("\n" + "=" * 70)
    print("STEP 7: Model Statistics")
    print("=" * 70)
    
    model_stats = model.get_usage_stats()
    print(f"Total Requests: {model_stats['total_requests']}")
    print(f"Cache Hits: {model_stats['cache_hits']}")
    print(f"Cache Misses: {model_stats['cache_misses']}")
    print(f"Cache Hit Rate: {model_stats['cache_hit_rate']:.2%}")
    
    # Final Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'data_dir': data_dir,
            'model': model_name,
            'strategy': strategy,
            'num_instances': len(instances)
        },
        'performance': {
            'accuracy': eval_results['metrics']['accuracy'],
            'exact_match': eval_results['metrics']['exact_match'],
            'partial_match': eval_results['metrics']['partial_match'],
            'time_per_instance': elapsed / len(predictions)
        },
        'model_stats': model_stats,
        'error_summary': {
            'num_correct': error_analysis['num_correct'],
            'num_partial': error_analysis['num_partial'],
            'num_wrong': error_analysis['num_wrong']
        },
        'files': {
            'predictions': str(pred_file),
            'evaluation': str(eval_file),
            'error_analysis': str(error_file)
        }
    }
    
    summary_file = output_path / f'summary_{strategy}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    save_json(summary, str(summary_file))
    
    print(f"\n✅ Pipeline completed successfully!")
    print(f"📁 All results saved to: {output_path}")
    print(f"📊 Summary: {summary_file}")
    
    return summary


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run complete AER pipeline")
    parser.add_argument('--data-dir', type=str, default='data/sample', help='Data directory')
    parser.add_argument('--model', type=str, default='mock', help='Model name')
    parser.add_argument('--strategy', type=str, default='zero_shot', choices=['zero_shot', 'cot'], help='Strategy')
    parser.add_argument('--max-instances', type=int, default=None, help='Max instances')
    parser.add_argument('--output-dir', type=str, default='outputs/pipeline', help='Output directory')
    
    args = parser.parse_args()
    
    try:
        summary = run_pipeline(
            data_dir=args.data_dir,
            model_name=args.model,
            strategy=args.strategy,
            max_instances=args.max_instances,
            output_dir=args.output_dir
        )
        return 0
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\n❌ Pipeline failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())