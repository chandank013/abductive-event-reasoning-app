"""
Abductive Event Reasoning System - Flask Backend
"""

from flask import Flask, render_template, request, jsonify
import pickle
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import numpy as np
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'myflasksecret123'

# MODEL DEFINITION (Must match training)

class ImprovedBERTClassifier(nn.Module):
    """Improved BERT with multi-head attention and residual connections"""
    
    def __init__(self, model_name='bert-base-uncased', num_labels=4, dropout=0.3, hidden_size=512):
        super(ImprovedBERTClassifier, self).__init__()
        
        # Use RoBERTa for better performance
        if 'roberta' in model_name:
            from transformers import RobertaModel
            self.bert = RobertaModel.from_pretrained(model_name)
        else:
            self.bert = BertModel.from_pretrained(model_name)
        
        bert_hidden_size = self.bert.config.hidden_size
        
        # Multi-layer classifier with residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(bert_hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        
        self.dropout2 = nn.Dropout(dropout * 0.5)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        
        self.dropout3 = nn.Dropout(dropout * 0.3)
        self.fc3 = nn.Linear(hidden_size // 2, num_labels)
    
    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use both [CLS] token and mean pooling
        cls_output = outputs.last_hidden_state[:, 0, :]
        mean_output = torch.mean(outputs.last_hidden_state, dim=1)
        pooled_output = (cls_output + mean_output) / 2
        
        # Multi-layer processing
        x = self.dropout1(pooled_output)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.dropout3(x)
        logits = self.fc3(x)
        
        return torch.sigmoid(logits)


# LOAD MODEL

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    """Load trained model"""
    try:
        with open(r'E:\GitHub\Academics\SEM03\Gen AI\Aer-Project_02\models\RoBERTa.pkl', 'rb') as f:  # adjust path if needed
            model_package = pickle.load(f)

        # Extract hyperparameters and initialize model
        dropout = model_package.get('best_params', {}).get('dropout', 0.3)
        model = ImprovedBERTClassifier(
            model_name='roberta-base',
            num_labels=4,
            dropout=dropout
        )
        model.load_state_dict(model_package['model_state_dict'])
        model.to(device)
        model.eval()

        tokenizer = model_package['tokenizer']

        print("✅ Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None


model, tokenizer = load_model()


# PREDICTION FUNCTION

def predict_cause(event, option_a, option_b, option_c, option_d, model_type='Baseline'):
    """Make prediction for given event and options"""
    
    if model is None or tokenizer is None:
        return {
            'error': 'Model not loaded',
            'predictions': {},
            'recommended': 'Unable to predict'
        }
    
    try:
        # Prepare input text
        text = f"[CLS] Question: {event} [SEP] "
        text += f"A: {option_a} [SEP] "
        text += f"B: {option_b} [SEP] "
        text += f"C: {option_c} [SEP] "
        text += f"D: {option_d} [SEP]"
        
        # Tokenize
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probabilities = outputs.cpu().numpy()[0]
        
        # Format results
        options = ['A', 'B', 'C', 'D']
        predictions = {
            opt: float(prob) for opt, prob in zip(options, probabilities)
        }
        
        # Determine recommended answer(s)
        threshold = 0.5
        recommended = [opt for opt, prob in predictions.items() if prob > threshold]
        
        if not recommended:
            # If no option exceeds threshold, recommend the highest
            recommended = [max(predictions, key=predictions.get)]
        
        recommended_str = ','.join(sorted(recommended))
        
        return {
            'predictions': predictions,
            'recommended': recommended_str,
            'confidence': {
                opt: f"{prob*100:.1f}%" for opt, prob in predictions.items()
            }
        }
    
    except Exception as e:
        return {
            'error': str(e),
            'predictions': {},
            'recommended': 'Error in prediction'
        }

# ROUTES

@app.route('/')
def home():
    """Landing page"""
    return render_template('landing.html')

@app.route('/predict')
def predict_page():
    """Prediction page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        event = data.get('event', '')
        option_a = data.get('option_a', '')
        option_b = data.get('option_b', '')
        option_c = data.get('option_c', '')
        option_d = data.get('option_d', 'Insufficient information')
        model_type = data.get('model', 'Baseline')
        
        # Validate inputs
        if not event:
            return jsonify({
                'success': False,
                'error': 'Event description is required'
            }), 400
        
        if not option_a or not option_b or not option_c:
            return jsonify({
                'success': False,
                'error': 'Please fill in options A, B, and C'
            }), 400
        
        # Make prediction
        result = predict_cause(event, option_a, option_b, option_c, option_d, model_type)
        
        if 'error' in result:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        with open('models/random_forest_new2.pkl', 'rb') as f:
            model_package = pickle.load(f)
        
        test_metrics = model_package.get('test_metrics', {})
        
        return jsonify({
            'success': True,
            'model_type': model_package.get('model_type', 'BERT'),
            'metrics': {
                'exact_match': f"{test_metrics.get('exact_match', 0)*100:.2f}%",
                'macro_f1': f"{test_metrics.get('macro_f1', 0)*100:.2f}%",
                'macro_precision': f"{test_metrics.get('macro_precision', 0)*100:.2f}%",
                'macro_recall': f"{test_metrics.get('macro_recall', 0)*100:.2f}%",
            },
            'best_params': model_package.get('best_params', {})
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ERROR HANDLERS

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500


# RUN APP

if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs('static/images', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)