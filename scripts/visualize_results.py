#!/usr/bin/env python3
"""
Visualize baseline experiment results
"""

import sys
from pathlib import Path
import json
import argparse

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger

logger = setup_logger("visualize", log_dir="outputs/logs")


def load_results(results_dir: str):
    """Load all results from directory"""
    results_path = Path(results_dir)
    
    all_results = []
    for result_file in results_path.glob("*.json"):
        # Skip the report and other non-result files
        if result_file.name in ['baseline_report.md', 'test_models_config.json']:
            continue
        
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                # Only include if it has the expected structure
                if 'strategy' in data or 'model' in data:
                    all_results.append(data)
        except Exception as e:
            logger.warning(f"Skipping file {result_file.name}: {e}")
    
    return all_results


def print_comparison_table(results):
    """Print comparison table"""
    print("\n" + "=" * 100)
    print("BASELINE RESULTS COMPARISON")
    print("=" * 100)
    
    # Header
    print(f"{'Strategy':<20} {'Model':<20} {'Accuracy':<12} {'Exact':<12} {'Partial':<12} {'Instances':<12}")
    print("-" * 100)
    
    # Sort by accuracy
    results_sorted = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
    
    for result in results_sorted:
        strategy = result.get('strategy', 'N/A')
        model = result.get('model', 'N/A')
        
        # Handle both old and new result formats
        if 'metrics' in result:
            accuracy = result['metrics'].get('accuracy', 0)
            exact = result['metrics'].get('exact_match', 0)
            partial = result['metrics'].get('partial_match', 0)
        else:
            accuracy = result.get('score', 0)
            exact = result.get('exact_match', 0)
            partial = result.get('partial_match', 0)
        
        instances = result.get('num_instances', 0)
        
        print(f"{strategy:<20} {model:<20} {accuracy:<12.4f} {exact:<12.4f} {partial:<12.4f} {instances:<12}")
    
    print("=" * 100)


def create_visualizations(results):
    """Create visualizations using matplotlib"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        if not results:
            print("   ⚠️  No results to visualize")
            return
        
        # Prepare data
        strategies = [r.get('strategy', 'N/A') for r in results]
        
        # Extract metrics (handle both formats)
        accuracies = []
        exact_matches = []
        partial_matches = []
        
        for r in results:
            if 'metrics' in r:
                accuracies.append(r['metrics'].get('accuracy', 0))
                exact_matches.append(r['metrics'].get('exact_match', 0))
                partial_matches.append(r['metrics'].get('partial_match', 0))
            else:
                accuracies.append(r.get('score', 0))
                exact_matches.append(r.get('exact_match', 0))
                partial_matches.append(r.get('partial_match', 0))
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Baseline Experiment Results', fontsize=16, fontweight='bold')
        
        # 1. Accuracy comparison
        ax1 = axes[0, 0]
        colors = plt.cm.viridis(np.linspace(0, 1, len(strategies)))
        bars = ax1.bar(range(len(strategies)), accuracies, color=colors)
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Overall Accuracy by Strategy')
        ax1.set_xticks(range(len(strategies)))
        ax1.set_xticklabels(strategies, rotation=45, ha='right')
        ax1.set_ylim([0, 1])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        # 2. Match type breakdown
        ax2 = axes[0, 1]
        x = np.arange(len(strategies))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, exact_matches, width, label='Exact Match', color='#2ecc71')
        bars2 = ax2.bar(x + width/2, partial_matches, width, label='Partial Match', color='#f39c12')
        
        ax2.set_ylabel('Rate')
        ax2.set_title('Match Type Distribution')
        ax2.set_xticks(x)
        ax2.set_xticklabels(strategies, rotation=45, ha='right')
        ax2.legend()
        ax2.set_ylim([0, 1])
        
        # 3. Correct/Partial/Wrong distribution
        ax3 = axes[1, 0]
        correct = []
        partial = []
        wrong = []
        
        for r in results:
            metrics = r.get('metrics', r)
            correct.append(metrics.get('correct', 0))
            partial.append(metrics.get('partial', 0))
            wrong.append(metrics.get('wrong', 0))
        
        x = np.arange(len(strategies))
        width = 0.25
        
        ax3.bar(x - width, correct, width, label='Correct', color='#2ecc71')
        ax3.bar(x, partial, width, label='Partial', color='#f39c12')
        ax3.bar(x + width, wrong, width, label='Wrong', color='#e74c3c')
        
        ax3.set_ylabel('Count')
        ax3.set_title('Prediction Distribution')
        ax3.set_xticks(x)
        ax3.set_xticklabels(strategies, rotation=45, ha='right')
        ax3.legend()
        
        # 4. Per-option F1 scores (if available)
        ax4 = axes[1, 1]
        
        # Check if per_option data exists
        has_per_option = any('per_option' in r.get('metrics', {}) for r in results)
        
        if has_per_option:
            options = ['A', 'B', 'C', 'D']
            f1_scores = {opt: [] for opt in options}
            
            for result in results:
                metrics = result.get('metrics', {})
                per_option = metrics.get('per_option', {})
                for opt in options:
                    if opt in per_option:
                        f1_scores[opt].append(per_option[opt].get('f1', 0))
                    else:
                        f1_scores[opt].append(0)
            
            x = np.arange(len(strategies))
            width = 0.2
            
            for i, opt in enumerate(options):
                offset = (i - 1.5) * width
                ax4.bar(x + offset, f1_scores[opt], width, label=f'Option {opt}')
            
            ax4.set_ylabel('F1 Score')
            ax4.set_title('Per-Option F1 Scores')
            ax4.set_xticks(x)
            ax4.set_xticklabels(strategies, rotation=45, ha='right')
            ax4.legend()
            ax4.set_ylim([0, 1])
        else:
            # Show a message instead
            ax4.text(0.5, 0.5, 'Per-option metrics not available',
                    ha='center', va='center', fontsize=12)
            ax4.set_title('Per-Option F1 Scores')
            ax4.axis('off')
        
        plt.tight_layout()
        
        # Save figure
        output_path = Path('outputs/results/baseline_comparison.png')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualization saved to: {output_path}")
        
    except ImportError:
        print("\n⚠️  Matplotlib not available. Skipping visualizations.")
        print("   Install with: pip install matplotlib")
    except Exception as e:
        print(f"\n⚠️  Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()


def generate_report(results, output_file: str):
    """Generate markdown report"""
    with open(output_file, 'w') as f:
        f.write("# Baseline Experiment Results\n\n")
        
        if results:
            timestamp = results[0].get('timestamp', 'N/A')
            f.write(f"*Generated: {timestamp}*\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- Total Experiments: {len(results)}\n")
        
        strategies = set(r.get('strategy', 'N/A') for r in results)
        models = set(r.get('model', 'N/A') for r in results)
        
        f.write(f"- Strategies Tested: {', '.join(strategies)}\n")
        f.write(f"- Models Used: {', '.join(models)}\n\n")
        
        f.write("## Results Table\n\n")
        f.write("| Strategy | Model | Accuracy | Exact Match | Partial Match | Instances |\n")
        f.write("|----------|-------|----------|-------------|---------------|----------|\n")
        
        for result in sorted(results, key=lambda x: x.get('score', x.get('metrics', {}).get('accuracy', 0)), reverse=True):
            strategy = result.get('strategy', 'N/A')
            model = result.get('model', 'N/A')
            
            # Handle both formats
            if 'metrics' in result:
                accuracy = result['metrics'].get('accuracy', 0)
                exact = result['metrics'].get('exact_match', 0)
                partial = result['metrics'].get('partial_match', 0)
            else:
                accuracy = result.get('score', 0)
                exact = result.get('exact_match', 0)
                partial = result.get('partial_match', 0)
            
            instances = result.get('num_instances', 0)
            
            f.write(f"| {strategy} | {model} | "
                   f"{accuracy:.4f} | "
                   f"{exact:.4f} | "
                   f"{partial:.4f} | "
                   f"{instances} |\n")
        
        f.write("\n## Detailed Metrics\n\n")
        
        for result in results:
            strategy = result.get('strategy', 'N/A')
            model = result.get('model', 'N/A')
            
            f.write(f"### {strategy} - {model}\n\n")
            
            # Handle both formats
            if 'metrics' in result:
                metrics = result['metrics']
                accuracy = metrics.get('accuracy', 0)
                exact_match = metrics.get('exact_match', 0)
                partial_match = metrics.get('partial_match', 0)
                correct = metrics.get('correct', 0)
                partial = metrics.get('partial', 0)
                wrong = metrics.get('wrong', 0)
            else:
                accuracy = result.get('score', 0)
                exact_match = result.get('exact_match', 0)
                partial_match = result.get('partial_match', 0)
                correct = result.get('correct', 0)
                partial = result.get('partial', 0)
                wrong = result.get('wrong', 0)
            
            f.write(f"- **Accuracy**: {accuracy:.4f}\n")
            f.write(f"- **Exact Match**: {exact_match:.4f}\n")
            f.write(f"- **Partial Match**: {partial_match:.4f}\n")
            f.write(f"- **Correct**: {correct}\n")
            f.write(f"- **Partial**: {partial}\n")
            f.write(f"- **Wrong**: {wrong}\n\n")
            
            # Per-option metrics if available
            if 'metrics' in result and 'per_option' in result['metrics']:
                f.write("**Per-Option Performance:**\n\n")
                f.write("| Option | Precision | Recall | F1 | Support |\n")
                f.write("|--------|-----------|--------|----|---------|\n")
                
                per_option = result['metrics']['per_option']
                for opt in ['A', 'B', 'C', 'D']:
                    if opt in per_option:
                        opt_metrics = per_option[opt]
                        f.write(f"| {opt} | {opt_metrics.get('precision', 0):.4f} | "
                               f"{opt_metrics.get('recall', 0):.4f} | "
                               f"{opt_metrics.get('f1', 0):.4f} | "
                               f"{opt_metrics.get('support', 0)} |\n")
            
            f.write("\n")
    
    print(f"\n📄 Report saved to: {output_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Visualize baseline results")
    parser.add_argument(
        '--results-dir',
        type=str,
        default='outputs/results',
        help='Directory containing result JSON files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/results',
        help='Directory to save visualizations'
    )
    
    args = parser.parse_args()
    
    try:
        print("=" * 70)
        print("Baseline Results Visualization")
        print("=" * 70)
        
        # Load results
        print(f"\n📥 Loading results from {args.results_dir}...")
        results = load_results(args.results_dir)
        
        if not results:
            print(f"❌ No results found in {args.results_dir}")
            print("   Run experiments first:")
            print("   python experiments/baseline/run_baseline.py")
            print("   or")
            print("   python scripts/complete_pipeline.py --data-dir data/sample --model mock")
            return 1
        
        print(f"   Found {len(results)} result files")
        
        # Remove duplicates (same strategy+model)
        unique_results = []
        seen = set()
        for r in results:
            key = (r.get('strategy'), r.get('model'))
            if key not in seen:
                seen.add(key)
                unique_results.append(r)
        
        if len(unique_results) < len(results):
            print(f"   Note: Removed {len(results) - len(unique_results)} duplicate results")
            results = unique_results
        
        # Print comparison table
        print_comparison_table(results)
        
        # Create visualizations
        print("\n📊 Creating visualizations...")
        create_visualizations(results)
        
        # Generate report
        report_file = Path(args.output_dir) / 'baseline_report.md'
        print("\n📄 Generating report...")
        generate_report(results, str(report_file))
        
        print("\n" + "=" * 70)
        print("✅ Visualization Complete!")
        print("=" * 70)
        print(f"\nOutput files:")
        print(f"  - Visualization: {args.output_dir}/baseline_comparison.png")
        print(f"  - Report: {report_file}")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())