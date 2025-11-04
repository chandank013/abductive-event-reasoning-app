#!/usr/bin/env python3
"""
Analyze AER dataset and generate statistics
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict, Counter
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import AERDataLoader
from src.utils.logger import setup_logger
from src.utils.helpers import save_json

logger = setup_logger("analyze_data", log_dir="outputs/logs")


class DataAnalyzer:
    """Analyze AER dataset"""
    
    def __init__(self, data_dir: str):
        """
        Initialize analyzer
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.loader = AERDataLoader(str(data_dir))
        self.instances = None
    
    def load_data(self):
        """Load data"""
        logger.info(f"Loading data from {self.data_dir}...")
        self.instances = self.loader.load()
        logger.info(f"Loaded {len(self.instances)} instances")
    
    def analyze_all(self) -> dict:
        """
        Perform complete analysis
        
        Returns:
            Dictionary with all statistics
        """
        if self.instances is None:
            self.load_data()
        
        logger.info("Performing complete data analysis...")
        
        analysis = {
            'basic_stats': self.get_basic_stats(),
            'answer_distribution': self.get_answer_distribution(),
            'option_analysis': self.get_option_analysis(),
            'document_stats': self.get_document_stats(),
            'text_length_stats': self.get_text_length_stats(),
            'complexity_analysis': self.get_complexity_analysis(),
            'topic_analysis': self.get_topic_analysis()
        }
        
        return analysis
    
    def get_basic_stats(self) -> dict:
        """Get basic dataset statistics"""
        logger.info("Computing basic statistics...")
        
        total_instances = len(self.instances)
        total_docs = sum(len(inst.docs) for inst in self.instances)
        instances_with_answers = sum(1 for inst in self.instances if inst.has_answer())
        
        stats = {
            'total_instances': total_instances,
            'instances_with_answers': instances_with_answers,
            'instances_without_answers': total_instances - instances_with_answers,
            'total_documents': total_docs,
            'avg_docs_per_instance': total_docs / total_instances if total_instances > 0 else 0,
            'unique_topics': len(set(inst.topic_id for inst in self.instances))
        }
        
        return stats
    
    def get_answer_distribution(self) -> dict:
        """Analyze answer distribution"""
        logger.info("Analyzing answer distribution...")
        
        answer_counts = Counter(
            inst.golden_answer for inst in self.instances 
            if inst.has_answer()
        )
        
        # Count single vs multiple answers
        single_answer = sum(1 for ans in answer_counts.keys() if ',' not in ans)
        multiple_answer = len(answer_counts) - single_answer
        
        # Count insufficient information answers (typically D)
        insufficient_count = sum(
            count for ans, count in answer_counts.items() 
            if ans == 'D' or 'D' in ans.split(',')
        )
        
        distribution = {
            'answer_counts': dict(answer_counts),
            'single_answer_instances': single_answer,
            'multiple_answer_instances': multiple_answer,
            'insufficient_info_count': insufficient_count,
            'most_common_answers': answer_counts.most_common(10)
        }
        
        return distribution
    
    def get_option_analysis(self) -> dict:
        """Analyze individual option statistics"""
        logger.info("Analyzing option statistics...")
        
        option_in_answer = defaultdict(int)
        
        for inst in self.instances:
            if inst.has_answer():
                answers = inst.get_answer_list()
                for opt in answers:
                    option_in_answer[opt] += 1
        
        total_with_answers = sum(1 for inst in self.instances if inst.has_answer())
        
        analysis = {
            'option_frequencies': dict(option_in_answer),
            'option_percentages': {
                opt: (count / total_with_answers * 100) if total_with_answers > 0 else 0
                for opt, count in option_in_answer.items()
            }
        }
        
        return analysis
    
    def get_document_stats(self) -> dict:
        """Analyze document statistics"""
        logger.info("Analyzing document statistics...")
        
        doc_counts = [len(inst.docs) for inst in self.instances]
        doc_lengths = []
        docs_with_content = 0
        
        for inst in self.instances:
            for doc in inst.docs:
                if doc.content:
                    docs_with_content += 1
                    doc_lengths.append(len(doc.content))
        
        stats = {
            'min_docs_per_instance': min(doc_counts) if doc_counts else 0,
            'max_docs_per_instance': max(doc_counts) if doc_counts else 0,
            'avg_docs_per_instance': sum(doc_counts) / len(doc_counts) if doc_counts else 0,
            'docs_with_content': docs_with_content,
            'docs_without_content': sum(len(inst.docs) for inst in self.instances) - docs_with_content,
            'min_doc_length': min(doc_lengths) if doc_lengths else 0,
            'max_doc_length': max(doc_lengths) if doc_lengths else 0,
            'avg_doc_length': sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0,
        }
        
        return stats
    
    def get_text_length_stats(self) -> dict:
        """Analyze text length statistics"""
        logger.info("Analyzing text lengths...")
        
        target_event_lengths = [len(inst.target_event) for inst in self.instances]
        option_lengths = []
        
        for inst in self.instances:
            for opt in inst.get_options_list():
                option_lengths.append(len(opt))
        
        stats = {
            'target_event_lengths': {
                'min': min(target_event_lengths) if target_event_lengths else 0,
                'max': max(target_event_lengths) if target_event_lengths else 0,
                'avg': sum(target_event_lengths) / len(target_event_lengths) if target_event_lengths else 0
            },
            'option_lengths': {
                'min': min(option_lengths) if option_lengths else 0,
                'max': max(option_lengths) if option_lengths else 0,
                'avg': sum(option_lengths) / len(option_lengths) if option_lengths else 0
            }
        }
        
        return stats
    
    def get_complexity_analysis(self) -> dict:
        """Analyze dataset complexity"""
        logger.info("Analyzing complexity...")
        
        # Count instances by number of correct answers
        answer_complexity = defaultdict(int)
        for inst in self.instances:
            if inst.has_answer():
                num_answers = len(inst.get_answer_list())
                answer_complexity[num_answers] += 1
        
        # Instances with option D (insufficient info)
        has_option_d = sum(1 for inst in self.instances if inst.has_answer() and 'D' in inst.answer)
        
        complexity = {
            'answers_per_instance': dict(answer_complexity),
            'instances_with_insufficient_info': has_option_d,
            'percentage_insufficient_info': (has_option_d / len(self.instances) * 100) if self.instances else 0
        }
        
        return complexity
    
    def get_topic_analysis(self) -> dict:
        """Analyze topic distribution"""
        logger.info("Analyzing topics...")
        
        topic_counts = Counter(inst.topic_id for inst in self.instances)
        topic_names = {}
        
        for inst in self.instances:
            if inst.topic_id not in topic_names:
                topic_names[inst.topic_id] = inst.topic
        
        analysis = {
            'total_topics': len(topic_counts),
            'instances_per_topic': dict(topic_counts),
            'topic_names': topic_names,
            'avg_instances_per_topic': sum(topic_counts.values()) / len(topic_counts) if topic_counts else 0
        }
        
        return analysis
    
    def print_summary(self, analysis: dict) -> None:
        """Print analysis summary"""
        logger.info("\n" + "=" * 70)
        logger.info("DATASET ANALYSIS SUMMARY")
        logger.info("=" * 70)
        
        # Basic stats
        logger.info("\n📊 Basic Statistics:")
        basic = analysis['basic_stats']
        logger.info(f"  Total Instances: {basic['total_instances']}")
        logger.info(f"  Instances with Answers: {basic['instances_with_answers']}")
        logger.info(f"  Instances without Answers: {basic['instances_without_answers']}")
        logger.info(f"  Total Documents: {basic['total_documents']}")
        logger.info(f"  Avg Documents/Instance: {basic['avg_docs_per_instance']:.2f}")
        logger.info(f"  Unique Topics: {basic['unique_topics']}")
        
        # Answer distribution
        logger.info("\n📋 Answer Distribution:")
        answer_dist = analysis['answer_distribution']
        logger.info(f"  Single Answer Instances: {answer_dist['single_answer_instances']}")
        logger.info(f"  Multiple Answer Instances: {answer_dist['multiple_answer_instances']}")
        logger.info(f"  Insufficient Info (D): {answer_dist['insufficient_info_count']}")
        
        logger.info("\n  Most Common Answers:")
        for ans, count in answer_dist['most_common_answers'][:5]:
            pct = (count / basic['instances_with_answers']) * 100 if basic['instances_with_answers'] > 0 else 0
            logger.info(f"    {ans}: {count} ({pct:.1f}%)")
        
        # Option analysis
        logger.info("\n🔤 Option Analysis:")
        option_analysis = analysis['option_analysis']
        for opt in ['A', 'B', 'C', 'D']:
            freq = option_analysis['option_frequencies'].get(opt, 0)
            pct = option_analysis['option_percentages'].get(opt, 0)
            logger.info(f"  Option {opt}: {freq} times ({pct:.1f}%)")
        
        # Document stats
        logger.info("\n📄 Document Statistics:")
        doc_stats = analysis['document_stats']
        logger.info(f"  Docs per Instance: {doc_stats['min_docs_per_instance']}-{doc_stats['max_docs_per_instance']} (avg: {doc_stats['avg_docs_per_instance']:.1f})")
        logger.info(f"  Docs with Content: {doc_stats['docs_with_content']}")
        logger.info(f"  Doc Length (chars): {doc_stats['min_doc_length']}-{doc_stats['max_doc_length']} (avg: {doc_stats['avg_doc_length']:.0f})")
        
        # Text lengths
        logger.info("\n📏 Text Length Statistics:")
        text_stats = analysis['text_length_stats']
        logger.info(f"  Target Event Length (chars): {text_stats['target_event_lengths']['min']}-{text_stats['target_event_lengths']['max']} (avg: {text_stats['target_event_lengths']['avg']:.0f})")
        logger.info(f"  Option Length (chars): {text_stats['option_lengths']['min']}-{text_stats['option_lengths']['max']} (avg: {text_stats['option_lengths']['avg']:.0f})")
        
        # Complexity
        logger.info("\n🧩 Complexity Analysis:")
        complexity = analysis['complexity_analysis']
        logger.info("  Answers per Instance:")
        for num_ans, count in sorted(complexity['answers_per_instance'].items()):
            logger.info(f"    {num_ans} answer(s): {count} instances")
        logger.info(f"  Insufficient Info Rate: {complexity['percentage_insufficient_info']:.1f}%")
        
        # Topics
        logger.info("\n🏷️  Topic Analysis:")
        topic_analysis = analysis['topic_analysis']
        logger.info(f"  Total Topics: {topic_analysis['total_topics']}")
        logger.info(f"  Avg Instances/Topic: {topic_analysis['avg_instances_per_topic']:.1f}")
        
        logger.info("\n" + "=" * 70)


def visualize_analysis(analysis: dict, output_dir: str) -> None:
    """
    Generate visualizations (requires matplotlib)
    
    Args:
        analysis: Analysis results
        output_dir: Directory to save plots
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        sns.set_style("whitegrid")
        
        # 1. Answer distribution
        answer_dist = analysis['answer_distribution']
        top_answers = dict(answer_dist['most_common_answers'][:10])
        
        plt.figure(figsize=(12, 6))
        plt.bar(top_answers.keys(), top_answers.values())
        plt.xlabel('Answer Combination')
        plt.ylabel('Frequency')
        plt.title('Top 10 Answer Distributions')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path / 'answer_distribution.png', dpi=300)
        plt.close()
        
        # 2. Option frequency
        option_analysis = analysis['option_analysis']
        options = ['A', 'B', 'C', 'D']
        frequencies = [option_analysis['option_frequencies'].get(opt, 0) for opt in options]
        
        plt.figure(figsize=(8, 6))
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        bars = plt.bar(options, frequencies, color=colors)
        plt.xlabel('Option')
        plt.ylabel('Frequency')
        plt.title('Option Frequency in Correct Answers')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path / 'option_frequency.png', dpi=300)
        plt.close()
        
        logger.info(f"Visualizations saved to {output_path}")
        
    except ImportError:
        logger.warning("Matplotlib not available. Skipping visualizations.")
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")


def main():
    parser = argparse.ArgumentParser(description="Analyze AER dataset")
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing data files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/reports',
        help='Directory to save analysis results'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualizations'
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("=" * 70)
        logger.info("AER Dataset Analysis")
        logger.info("=" * 70)
        
        # Analyze data
        analyzer = DataAnalyzer(args.data_dir)
        analysis = analyzer.analyze_all()
        
        # Print summary
        analyzer.print_summary(analysis)
        
        # Save analysis
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        data_dir_name = Path(args.data_dir).name
        analysis_file = output_dir / f'{data_dir_name}_analysis.json'
        save_json(analysis, str(analysis_file))
        logger.info(f"\n✓ Analysis saved to {analysis_file}")
        
        # Generate visualizations
        if args.visualize:
            viz_dir = output_dir / 'visualizations' / data_dir_name
            visualize_analysis(analysis, str(viz_dir))
        
        logger.info("\n" + "=" * 70)
        logger.info("✓ Analysis complete!")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()