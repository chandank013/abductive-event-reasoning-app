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
    for result_file in results_path.glob("baseline_*.json"):
        with open(result_file, 'r') as f:
            data = json.load(f)
            all_results.append(data)
    
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
    results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
    
    for result in results_sorted:
        strategy = result['strategy']
        model = result['model']
        accuracy = result['score']
        exact = result['metrics']['exact_match']
        partial = result['metrics']['partial_match']
        instances = result['num_instances']
        
        print(f"{strategy:<20} {model:<20} {accuracy:<12.4f} {exact:<12.4f} {partial:<12.4f} {instances:<12}")
    
    print("=" * 100)


def create_visualizations(results):
    """Create visualizations using matplotlib"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Prepare data
        strategies = [r['strategy'] for r in results]
        accuracies = [r['score'] for r in results]
        exact_matches = [r['metrics']['exact_match'] for r in results]
        partial_matches = [r['metrics']['partial_match'] for r in results]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Baseline Experiment Results', fontsize=16, fontweight='bold')
        
        # 1. Accuracy comparison
        ax1 = axes[0, 0]
        bars = ax1.bar(strategies, accuracies, color=['#3498db', '#e74c3c', '#2ecc71'])
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Overall Accuracy by Strategy')
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
        ax2.set_xticklabels(strategies)
        ax2.legend()
        ax2.set_ylim([0, 1])
        
        # 3. Correct/Partial/Wrong distribution
        ax3 = axes[1, 0]
        correct = [r['metrics']['correct'] for r in results]
        partial = [r['metrics']['partial'] for r in results]
        wrong = [r['metrics']['wrong'] for r in results]
        
        x = np.arange(len(strategies))
        width = 0.25
        
        ax3.bar(x - width, correct, width, label='Correct', color='#2ecc71')
        ax3.bar(x, partial, width, label='Partial', color='#f39c12')
        ax3.bar(x + width, wrong, width, label='Wrong', color='#e74c3c')
        
        ax3.set_ylabel('Count')
        ax3.set_title('Prediction Distribution')
        ax3.set_xticks(x)
        ax3.set_xticklabels(strategies)
        ax3.legend()
        
        # 4. Per-option F1 scores
        ax4 = axes[1, 1]
        
        # Collect F1 scores for each option
        options = ['A', 'B', 'C', 'D']
        f1_scores = {opt: [] for opt in options}
        
        for result in results:
            for opt in options:
                f1_scores[opt].append(result['metrics']['per_option'][opt]['f1'])
        
        x = np.arange(len(strategies))
        width = 0.2
        
        for i, opt in enumerate(options):
            offset = (i - 1.5) * width
            ax4.bar(x + offset, f1_scores[opt], width, label=f'Option {opt}')
        
        ax4.set_ylabel('F1 Score')
        ax4.set_title('Per-Option F1 Scores')
        ax4.set_xticks(x)
        ax4.set_xticklabels(strategies)
        ax4.legend()
        ax4.set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Save figure
        output_path = Path('outputs/results/baseline_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualization saved to: {output_path}")
        
        # Show if interactive
        # plt.show()
        
    except ImportError:
        print("\n⚠️  Matplotlib not available. Skipping visualizations.")
        print("   Install with: pip install matplotlib")
    except Exception as e:
        print(f"\n⚠️  Error creating visualizations: {e}")


def generate_report(results, output_file: str):
    """Generate markdown report"""
    with open(output_file, 'w') as f:
        f.write("# Baseline Experiment Results\n\n")
        f.write(f"*Generated: {results[0].get('timestamp', 'N/A')}*\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- Total Experiments: {len(results)}\n")
        f.write(f"- Strategies Tested: {', '.join(set(r['strategy'] for r in results))}\n")
        f.write(f"- Models Used: {', '.join(set(r['model'] for r in results))}\n\n")
        
        f.write("## Results Table\n\n")
        f.write("| Strategy | Model | Accuracy | Exact Match | Partial Match | Instances |\n")
        f.write("|----------|-------|----------|-------------|---------------|----------|\n")
        
        for result in sorted(results, key=lambda x: x['score'], reverse=True):
            f.write(f"| {result['strategy']} | {result['model']} | "
                   f"{result['score']:.4f} | "
                   f"{result['metrics']['exact_match']:.4f} | "
                   f"{result['metrics']['partial_match']:.4f} | "
                   f"{result['num_instances']} |\n")
        
        f.write("\n## Detailed Metrics\n\n")
        
        for result in results:
            f.write(f"### {result['strategy']} - {result['model']}\n\n")
            
            metrics = result['metrics']
            f.write(f"- **Accuracy**: {result['score']:.4f}\n")
            f.write(f"- **Exact Match**: {metrics['exact_match']:.4f}\n")
            f.write(f"- **Partial Match**: {metrics['partial_match']:.4f}\n")
            f.write(f"- **Correct**: {metrics['correct']}\n")
            f.write(f"- **Partial**: {metrics['partial']}\n")
            f.write(f"- **Wrong**: {metrics['wrong']}\n\n")
            
            f.write("**Per-Option Performance:**\n\n")
            f.write("| Option | Precision | Recall | F1 | Support |\n")
            f.write("|--------|-----------|--------|----|---------|\n")
            
            for opt in ['A', 'B', 'C', 'D']:
                opt_metrics = metrics['per_option'][opt]
                f.write(f"| {opt} | {opt_metrics['precision']:.4f} | "
                       f"{opt_metrics['recall']:.4f} | "
                       f"{opt_metrics['f1']:.4f} | "
                       f"{opt_metrics['support']} |\n")
            
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
            print("   Run experiments first: python experiments/baseline/run_baseline.py")
            return 1
        
        print(f"   Found {len(results)} result files")
        
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
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())