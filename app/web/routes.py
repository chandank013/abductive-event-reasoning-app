"""
Web interface routes
"""

from flask import Blueprint, render_template, request, redirect, url_for
from src.utils.logger import get_logger

logger = get_logger("web")

# Create blueprint
web_bp = Blueprint('web', __name__)


@web_bp.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@web_bp.route('/dashboard')
def dashboard():
    """Dashboard page"""
    # TODO: Get actual statistics
    stats = {
        'total_instances': 1000,
        'total_predictions': 250,
        'accuracy': 0.82,
        'models_available': 3
    }
    return render_template('dashboard.html', stats=stats)


@web_bp.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction interface"""
    if request.method == 'POST':
        # Handle prediction request
        target_event = request.form.get('target_event')
        option_a = request.form.get('option_a')
        option_b = request.form.get('option_b')
        option_c = request.form.get('option_c')
        option_d = request.form.get('option_d')
        
        # TODO: Implement actual prediction
        result = {
            'prediction': 'A',
            'confidence': {'A': 0.75, 'B': 0.15, 'C': 0.05, 'D': 0.05},
            'reasoning': 'Based on the evidence...'
        }
        
        return render_template('prediction.html', result=result)
    
    return render_template('prediction.html', result=None)


@web_bp.route('/data')
def data_explorer():
    """Data exploration page"""
    return render_template('data_explorer.html')


@web_bp.route('/evaluation')
def evaluation():
    """Evaluation results page"""
    return render_template('evaluation.html')


@web_bp.route('/about')
def about():
    """About page"""
    return render_template('about.html')