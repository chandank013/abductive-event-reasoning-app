# Abductive Event Reasoning (AER) System

A comprehensive framework for identifying plausible causes of real-world events using large language models and retrieval-augmented generation. Built for SemEval 2026 Task 12.


## 🎯 Project Overview

The Abductive Event Reasoning task is framed as a multiple-choice question answering problem, evaluating large language models' ability to identify the most plausible direct cause of a real-world event based on textual evidence.

### Task Definition

Each instance consists of:
- **Event**: A short description of an observed real-world event
- **Context**: Retrieved documents related to the event (including distractor documents)
- **Options (A-D)**: Four candidate explanations, where:
  - One or more may be correct
  - Option D is typically: "The information provided is insufficient to determine the cause"

### Evaluation Metric

- ✅ Full match with correct answers → 1 point
- ⚠️ Partial match → 0.5 point
- ❌ Wrong or invalid selection → 0 points

## 📁 Project Structure

```
project/
│
├── abstract/                # Abstract and documentation
│   ├── abstract.md          # This abstract md file
│   └── abstract.docs          # This abstract file
├── .ebextensions/          # AWS Elastic Beanstalk config (if deploying)
├── .git/                   # Git repository
├── dataset/                # Dataset files
│   ├── validation
│   │    ├── question.jsonl
│   │    └── docs.jsonl
│   ├── question.jsonl
│   └── docs.jsonl
│
├── models/                 # Trained models
│   └── best_baseline_model.pkl
│
├── notebooks/              # Jupyter notebooks
│   ├── EDA.ipynb          # Exploratory Data Analysis
│   └── Model_Training.ipynb  # Model training and tuning
│
├── static/                 # Static files
│   └── images/            # Images and plots
│       ├── logo.jpg
│       ├── preview1.png
│       └── preview2.png
│
├── templates/              # HTML templates
│   ├── landing.html       # Landing page
│   ├── index.html         # Prediction page
│
├── .gitignore             # Git ignore file
├── abstract.md            # Project abstract
├── COMPLETE_SYSTEM_SUMMARY.md           # Complete system summary of project
├── config.py               # Configuration file
├── quick_start.py          # Quick start script
├── quick_start.py          # Quick start script
├── application.py          # Flask application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── septup.py              # Setup script
└── SYSTEM_ARCHITECTURE.txt   # System architecture description
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 8GB+ RAM (for BERT models)
- CUDA-compatible GPU (optional, for faster training)

### Step 1: Clone Repository

```bash
git clone https//github.com/chandank013/abductive-event-reasoning.git
cd abductive-event-reasoning
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download BERT Model (First Time Only)

The BERT model will be automatically downloaded when you first run the training notebook or application.

## 📊 Data Preparation

### 1. Prepare Your Data

Ensure your data files are in the correct format:

**question.jsonl:**
```json
{
    "topic_id": 1,
    "question": "The Iranian government issued an intercity travel ban...",
    "option_A": "U.S. port closures",
    "option_B": "COVID-19 lockdowns",
    "option_C": "Economic sanctions",
    "option_D": "Insufficient information",
    "answer": "B"
}
```

**docs.jsonl:**
```json
{
    "topic_id": 1,
    "docs": [
        {
            "title": "Document Title",
            "content": "Document content here..."
        }
    ]
}
```

### 2. Run Preprocessing

Open and run `notebooks/EDA.ipynb` to:
- Load and explore data
- Preprocess text
- Generate visualizations
- Save cleaned data

## 🔬 Model Training

### Multi-Algorithm Approach

The system trains and compares **6 different architectures**:

1. **BERT Baseline** - Standard BERT with feedforward classifier
2. **RoBERTa Baseline** - Improved pre-training, better tokenization
3. **RoBERTa + BiLSTM + Attention** - Sequential modeling with attention ⭐
4. **Longformer** - Handles long contexts up to 4096 tokens
5. **DistilBERT** - Fast and lightweight (60% faster inference)
6. **Hierarchical Attention** - Dual-level attention mechanism ⭐⭐

### Hyperparameter Tuning

Each model is trained with automatic hyperparameter tuning:

- **Learning Rate**: [1e-5, 2e-5, 3e-5]
- **Dropout**: [0.2, 0.3, 0.4]
- **Hidden Sizes**: [256, 384, 512, 768]
- **Batch Sizes**: [4, 8, 16, 32]
- **~4 trials per model** = 24 total trainings

The system automatically:
- Trains each model with different hyperparameters
- Evaluates on validation set
- Selects best configuration
- Retrains best model on combined train+val data
- Finds optimal threshold
- Saves best performing model

### 1. Run Training Notebook

Open and run `notebooks/Model_Training.ipynb` to:
- Train all 6 model architectures
- Perform hyperparameter tuning for each
- Compare model performance
- Automatically select and save best model
- Generate comprehensive visualizations

### Training Output

The training process will:
- Train ~24 model variants (6 architectures × 4 hyperparameter sets)
- Take 2-4 hours on GPU / 8-12 hours on CPU
- Generate comparison visualizations
- Save best model to `models/longformer.pkl`
- Create detailed results CSV

## 🌐 Running the Application

### Development Mode

```bash
python application.py
```

The application will be available at `http://localhost:5000`

### Production Mode

```bash
gunicorn -w 4 -b 0.0.0.0:8000 application:app
```

## 🖥️ Using the Web Interface

### 1. Landing Page
- Overview of the AER system
- Key features and capabilities
- Navigation to different sections

### 2. Prediction Page
- Enter target event description
- Provide 4 possible causes (A, B, C, D)
- Select model type
- Get predictions with confidence scores


## 📈 Model Performance

### Best Model Results

```
**Best Performing Model:** Longformer Baseline

| Metric | Performance |
|--------|-------------|
| Validation Macro F1 | 61.52% |
| Exact Match Accuracy | 25.00% |
| Optimal Threshold | 45.0% |
```

## 🔧 API Usage

### Prediction Endpoint

```python
import requests

url = "http://localhost:5000/api/predict"

data = {
    "event": "The Iranian government issued a travel ban",
    "option_a": "U.S. port closures",
    "option_b": "COVID-19 lockdowns",
    "option_c": "Economic sanctions",
    "option_d": "Insufficient information",
    "model": "Baseline"
}

response = requests.post(url, json=data)
result = response.json()

print(result)
# Output:
# {
#     "success": True,
#     "result": {
#         "predictions": {
#             "A": 0.23,
#             "B": 0.87,
#             "C": 0.45,
#             "D": 0.12
#         },
#         "recommended": "B",
#         "confidence": {
#             "A": "23.0%",
#             "B": "87.0%",
#             "C": "45.0%",
#             "D": "12.0%"
#         }
#     }
# }
```

### Model Info Endpoint

```python
response = requests.get("http://localhost:5000/api/model-info")
info = response.json()
print(info)
```

## 🛠️ Customization

### Training Custom Models

Edit `notebooks/Model_Training.ipynb` to:
- Adjust hyperparameters
- Add new model architectures
- Modify training loop
- Change evaluation metrics

### Modifying UI

Edit templates in `templates/` folder:
- `landing.html` - Landing page
- `index.html` - Prediction interface
- Custom CSS in `<style>` tags

### Adding New Features

1. Add route in `application.py`:
```python
@app.route('/predict')
def new_feature():
    return render_template('landing.html')
```

2. Create template in `templates/landing.html`

## 📝 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Reduce batch size in training
BATCH_SIZE = 4  # or 2
```

**2. Model Not Loading**
```bash
# Ensure model file exists
ls models/best_baseline_model.pkl

# Retrain if needed
jupyter notebook notebooks/Model_Training.ipynb
```

**3. Port Already in Use**
```bash
# Change port in application.py
app.run(port=5001)
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request


## 🙏 Acknowledgments

- SemEval 2026 Task 12 organizers
- Hugging Face Transformers library
- PyTorch team
- Flask framework

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the development team.

---

**Built with ❤️ for SemEval 2026 Task 12**