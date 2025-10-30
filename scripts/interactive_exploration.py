#!/usr/bin/env python3
"""
Interactive data exploration script
Alternative to Jupyter notebook
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter, defaultdict

from src.data.loder import AERDataLoader
from src.utils.logger import setup_logger

logger = setup_logger("exploration", log_dir="outputs/logs")

def main():
    # Load data
    logger.info("Loading dataset...")
    loader = AERDataLoader('data/raw')
    instances = loader.load()
    
    logger.info(f"Loaded {len(instances)} instances")
    
    # Create output directory
    from pathlib import Path
    output_dir = Path('outputs/reports/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all visualizations
    logger.info("Generating visualizations...")
    
    # 1. Answer distribution
    answer_counts = Counter(inst.answer for inst in instances if inst.answer)
    top_15 = dict(answer_counts.most_common(15))
    
    fig = go.Figure([go.Bar(x=list(top_15.keys()), y=list(top_15.values()))])
    fig.update_layout(title='Top 15 Answer Distributions')
    fig.write_html(str(output_dir / 'answer_distribution.html'))
    fig.write_image(str(output_dir / 'answer_distribution.png'))
    
    # 2. Option frequency
    option_counts = defaultdict(int)
    for inst in instances:
        if inst.answer:
            for opt in inst.answer.split(','):
                option_counts[opt.strip()] += 1
    
    fig = go.Figure([
        go.Bar(
            x=['A', 'B', 'C', 'D'],
            y=[option_counts[opt] for opt in ['A', 'B', 'C', 'D']],
            marker_color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        )
    ])
    fig.update_layout(title='Option Frequency in Correct Answers')
    fig.write_html(str(output_dir / 'option_frequency.html'))
    
    # 3. Document distribution
    doc_counts = [len(inst.docs) for inst in instances]
    fig = go.Figure([go.Histogram(x=doc_counts)])
    fig.update_layout(title='Document Count Distribution')
    fig.write_html(str(output_dir / 'document_distribution.html'))
    
    logger.info(f"Visualizations saved to {output_dir}")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("DATASET SUMMARY")
    logger.info("="*80)
    logger.info(f"Total Instances: {len(instances)}")
    logger.info(f"Total Documents: {sum(len(inst.docs) for inst in instances)}")
    logger.info(f"Avg Docs/Instance: {np.mean(doc_counts):.2f}")
    logger.info("="*80)

if __name__ == "__main__":
    main()