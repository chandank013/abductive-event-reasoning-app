"""
Error analysis for predictions
"""

from typing import List, Dict, Tuple
from collections import defaultdict, Counter

from src.data.loader import AERInstance
from src.evaluation.metrics import parse_answer, calculate_score
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ErrorAnalyzer:
    """Analyze prediction errors"""
    
    def __init__(self):
        """Initialize error analyzer"""
        logger.info("Initialized ErrorAnalyzer")
    
    def analyze(
        self,
        instances: List[AERInstance],
        predictions: List[Dict]
    ) -> Dict:
        """
        Analyze errors in predictions
        
        Args:
            instances: Ground truth instances
            predictions: Prediction dictionaries
            
        Returns:
            Error analysis results
        """
        logger.info(f"Analyzing errors for {len(predictions)} predictions")
        
        # Create lookup
        inst_lookup = {inst.uuid: inst for inst in instances}
        
        # Categorize predictions
        correct = []
        partial = []
        wrong = []
        
        for pred in predictions:
            uuid = pred['uuid']
            if uuid not in inst_lookup:
                continue
            
            instance = inst_lookup[uuid]
            if not instance.has_answer():
                continue
            
            score = calculate_score(pred['prediction'], instance.golden_answer)
            
            entry = {
                'uuid': uuid,
                'instance': instance,
                'prediction': pred['prediction'],
                'gold': instance.golden_answer,
                'score': score
            }
            
            if score == 1.0:
                correct.append(entry)
            elif score == 0.5:
                partial.append(entry)
            else:
                wrong.append(entry)
        
        # Analyze error patterns
        error_patterns = self._analyze_error_patterns(wrong, partial)
        
        # Analyze by answer type
        by_answer_type = self._analyze_by_answer_type(correct, partial, wrong)
        
        # Analyze by topic
        by_topic = self._analyze_by_topic(correct, partial, wrong)
        
        results = {
            'num_correct': len(correct),
            'num_partial': len(partial),
            'num_wrong': len(wrong),
            'error_patterns': error_patterns,
            'by_answer_type': by_answer_type,
            'by_topic': by_topic,
            'sample_errors': wrong[:5]  # Sample of errors
        }
        
        return results
    
    def _analyze_error_patterns(
        self,
        wrong: List[Dict],
        partial: List[Dict]
    ) -> Dict:
        """Analyze common error patterns"""
        patterns = {
            'predicted_D_when_not_D': 0,
            'missed_D': 0,
            'over_predicted': 0,
            'under_predicted': 0,
            'option_confusion': Counter()
        }
        
        for entry in wrong + partial:
            pred_set = parse_answer(entry['prediction'])
            gold_set = parse_answer(entry['gold'])
            
            # Predicted D incorrectly
            if 'D' in pred_set and 'D' not in gold_set:
                patterns['predicted_D_when_not_D'] += 1
            
            # Missed D
            if 'D' in gold_set and 'D' not in pred_set:
                patterns['missed_D'] += 1
            
            # Over/under prediction
            if len(pred_set) > len(gold_set):
                patterns['over_predicted'] += 1
            elif len(pred_set) < len(gold_set):
                patterns['under_predicted'] += 1
            
            # Option confusion (predicted X when should be Y)
            for opt in pred_set - gold_set:
                for gold_opt in gold_set:
                    patterns['option_confusion'][f"{opt}_for_{gold_opt}"] += 1
        
        return {
            'predicted_D_when_not_D': patterns['predicted_D_when_not_D'],
            'missed_D': patterns['missed_D'],
            'over_predicted': patterns['over_predicted'],
            'under_predicted': patterns['under_predicted'],
            'most_common_confusions': patterns['option_confusion'].most_common(5)
        }
    
    def _analyze_by_answer_type(
        self,
        correct: List[Dict],
        partial: List[Dict],
        wrong: List[Dict]
    ) -> Dict:
        """Analyze performance by answer type"""
        def get_type(answer: str) -> str:
            opts = parse_answer(answer)
            if 'D' in opts:
                return 'insufficient_info'
            elif len(opts) == 1:
                return 'single'
            else:
                return 'multiple'
        
        types = defaultdict(lambda: {'correct': 0, 'partial': 0, 'wrong': 0})
        
        for entry in correct:
            answer_type = get_type(entry['gold'])
            types[answer_type]['correct'] += 1
        
        for entry in partial:
            answer_type = get_type(entry['gold'])
            types[answer_type]['partial'] += 1
        
        for entry in wrong:
            answer_type = get_type(entry['gold'])
            types[answer_type]['wrong'] += 1
        
        # Calculate accuracy per type
        results = {}
        for answer_type, counts in types.items():
            total = sum(counts.values())
            accuracy = (counts['correct'] + 0.5 * counts['partial']) / total if total > 0 else 0
            results[answer_type] = {
                **counts,
                'total': total,
                'accuracy': accuracy
            }
        
        return results
    
    def _analyze_by_topic(
        self,
        correct: List[Dict],
        partial: List[Dict],
        wrong: List[Dict]
    ) -> Dict:
        """Analyze performance by topic"""
        topics = defaultdict(lambda: {'correct': 0, 'partial': 0, 'wrong': 0})
        
        for entry in correct:
            topic_id = entry['instance'].topic_id
            topics[topic_id]['correct'] += 1
        
        for entry in partial:
            topic_id = entry['instance'].topic_id
            topics[topic_id]['partial'] += 1
        
        for entry in wrong:
            topic_id = entry['instance'].topic_id
            topics[topic_id]['wrong'] += 1
        
        # Calculate accuracy per topic
        results = {}
        for topic_id, counts in topics.items():
            total = sum(counts.values())
            accuracy = (counts['correct'] + 0.5 * counts['partial']) / total if total > 0 else 0
            results[topic_id] = {
                **counts,
                'total': total,
                'accuracy': accuracy
            }
        
        return results
    
    def print_analysis(self, analysis: Dict) -> None:
        """Print error analysis"""
        print("\n" + "=" * 70)
        print("ERROR ANALYSIS")
        print("=" * 70)
        
        print(f"\n📊 Summary:")
        print(f"  Correct: {analysis['num_correct']}")
        print(f"  Partial: {analysis['num_partial']}")
        print(f"  Wrong:   {analysis['num_wrong']}")
        
        print(f"\n🔍 Error Patterns:")
        patterns = analysis['error_patterns']
        print(f"  Predicted D incorrectly: {patterns['predicted_D_when_not_D']}")
        print(f"  Missed D: {patterns['missed_D']}")
        print(f"  Over-predicted: {patterns['over_predicted']}")
        print(f"  Under-predicted: {patterns['under_predicted']}")
        
        print(f"\n  Most common confusions:")
        for confusion, count in patterns['most_common_confusions']:
            print(f"    {confusion}: {count}")
        
        print(f"\n📈 Performance by Answer Type:")
        for answer_type, metrics in analysis['by_answer_type'].items():
            print(f"  {answer_type}:")
            print(f"    Accuracy: {metrics['accuracy']:.4f}")
            print(f"    Correct: {metrics['correct']}, Partial: {metrics['partial']}, Wrong: {metrics['wrong']}")
        
        print("=" * 70)