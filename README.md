# 🧠 Abductive Event Reasoning (AER) System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> A comprehensive web-based framework for abductive event reasoning using large language models, retrieval-augmented generation, and advanced prompting techniques. Built for **SemEval 2026 Task 12**.

## 🌟 Overview

The **Abductive Event Reasoning (AER)** system implements a state-of-the-art solution for identifying plausible causes of real-world events using natural language understanding and causal reasoning. This framework combines:

- 🌐 **Full-Stack Web Application** with Flask backend and responsive HTML/CSS/JS frontend
- 🤖 **Multiple LLM Integration** (GPT-4, Claude, Llama 3)
- 🔍 **Retrieval-Augmented Generation (RAG)** with FAISS/ChromaDB
- 🧩 **Chain-of-Thought Reasoning** and few-shot learning
- 📊 **Interactive Web Dashboard** for visualization and analysis
- 🎯 **Ensemble Methods** for improved accuracy
- 📈 **Comprehensive Evaluation** with partial match scoring
- 🚀 **RESTful API** for programmatic access

---

## 🚀 Key Features

### Web Application
- 🌐 **Modern Web Interface**: Responsive design with Bootstrap
- 📊 **Real-time Dashboard**: Live metrics and visualizations
- 🔮 **Interactive Prediction**: Single and batch prediction interfaces
- 📈 **Performance Analytics**: Charts and detailed evaluation reports
- 🎨 **Data Exploration**: Interactive dataset browser
- 👤 **User Management**: Authentication and session handling

### Core Capabilities
- ✅ Multi-choice causal reasoning with evidence-based inference
- ✅ Document retrieval and relevance scoring
- ✅ Dynamic prompt engineering with template management
- ✅ Multiple reasoning strategies (zero-shot, few-shot, CoT)
- ✅ Model ensemble and confidence calibration
- ✅ RESTful API with comprehensive endpoints

### Technical Highlights
- 🏗️ Modular, extensible architecture
- ⚡ Efficient caching and embeddings storage
- 📝 Comprehensive logging and monitoring
- 🧪 Full test coverage with pytest
- 🐳 Docker support (optional)
- 🔒 Secure API with authentication
- 📊 Real-time progress tracking

---

## 📋 Task Definition

The AER task evaluates language models' ability to identify the most plausible direct cause of a real-world event from multiple-choice options based on textual evidence.

**Input:**
- **Target Event**: The observation that needs explanation (O₂)
- **Context Documents**: Retrieved articles with evidence (may include distractors)
- **4 Candidate Hypotheses**: Options A-D (possible causes)

**Output:**
- One or more correct options (e.g., "A", "B,C", "A,D")
- Option D typically represents "Insufficient information"

**Scoring:**
- ✅ Full match: 1.0 point
- ⚠️ Partial match: 0.5 points
- ❌ Wrong answer: 0 points

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Backend** | Flask 2.3+, Python 3.9+ |
| **Frontend** | HTML5, CSS3, JavaScript (ES6+), Bootstrap 5 |
| **ML/AI** | PyTorch, Transformers, LangChain |
| **LLMs** | OpenAI GPT-4, Anthropic Claude, Llama 3 |
| **Retrieval** | FAISS, ChromaDB, Sentence-BERT |
| **Visualization** | Chart.js, Plotly.js, D3.js |
| **Database** | SQLite (dev), PostgreSQL (prod) |
| **Caching** | Redis (optional) |
| **API** | RESTful, JSON |
| **Testing** | Pytest, Selenium |
| **Deployment** | Gunicorn, Nginx, Docker |

---

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- pip or conda
- Git
- (Optional) Redis for caching
- (Optional) Docker for containerized deployment

### Quick Start
```bash
# 1. Clone repository
git clone https://github.com/chandank013/aer-project.git
cd aer-project

# 2. Create virtual environment
python -m venv venv

# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup environment variables
cp .env.example .env
# Edit .env with your API keys and configuration

# 5. Run complete setup (first time only)
python run.py --mode setup

# 6. Start the application
python run.py
```

### Configuration

Edit `.env` file with your settings:
```bash
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
HUGGINGFACE_TOKEN=your_hf_token_here

# Flask Configuration
FLASK_APP=app
FLASK_ENV=development  # or production
SECRET_KEY=your_secret_key_here

# Server Configuration
HOST=0.0.0.0
PORT=5000
DEBUG=False

# Model Configuration
DEFAULT_MODEL=gpt-4
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
MAX_TOKENS=1000
TEMPERATURE=0.3

# Database
DATABASE_URL=sqlite:///aer.db  # or postgresql://...

# Redis (optional)
REDIS_URL=redis://localhost:6379/0
```

---

## 🎮 Usage

### Starting the Application
```bash
# Full web application (default)
python run.py

# With custom port
python run.py --port 8080

# Debug mode
python run.py --debug

# API server only (no web interface)
python run.py --mode api --port 5001

# Training mode
python run.py --mode train

# Setup mode (first time)
python run.py --mode setup
```

### Accessing the Application

Once running, access:

- **🌐 Web Interface**: http://localhost:5000
- **📊 Dashboard**: http://localhost:5000/dashboard
- **🔮 Prediction**: http://localhost:5000/predict
- **📈 Evaluation**: http://localhost:5000/evaluation
- **🔍 Data Explorer**: http://localhost:5000/data
- **📚 API Documentation**: http://localhost:5000/api/docs

### Using the Web Interface

1. **Dashboard**: View overall system statistics and performance metrics
2. **Single Prediction**: 
   - Enter target event
   - Select or paste context documents
   - Choose options A-D
   - Get instant prediction with confidence scores
3. **Batch Prediction**: Upload JSONL file for bulk predictions
4. **Evaluation**: View detailed performance analysis on dev/test sets
5. **Data Explorer**: Browse and filter dataset instances

### API Usage
```python
import requests

# Single prediction
response = requests.post('http://localhost:5000/api/predict', json={
    "target_event": "The Iranian government issued a travel ban.",
    "options": {
        "A": "U.S. mandated port closures",
        "B": "COVID-19 forced lockdowns",
        "C": "COVID-19 forced lockdowns",
        "D": "Virus was identified"
    },
    "docs": [
        {
            "title": "COVID-19 Pandemic",
            "content": "Countries worldwide implemented lockdowns..."
        }
    ]
})

print(response.json())
# Output: {"prediction": "B,C", "confidence": {"A": 0.15, "B": 0.85, ...}}
```

### Command Line Tools
```bash
# Analyze dataset
python scripts/analyze_data.py --data-dir data/train --visualize

# Build embeddings
python scripts/build_embeddings.py --data-dir data/train

# Train model
python scripts/train_model.py --config config/training_config.yaml

# Evaluate predictions
python src/evaluation/evaluator.py \
    --predictions outputs/predictions/dev_predictions.jsonl \
    --ground_truth data/dev/questions.jsonl

# Run tests
pytest tests/ -v --cov=src --cov=app
```

---

## 📁 Project Structure
```
aer-project/
├── run.py                       # 🚀 Main entry point - Run this!
├── wsgi.py                      # Production WSGI server
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
├── README.md                    # This file
│
├── app/                         # Flask application
│   ├── __init__.py             # App factory
│   ├── api/                    # REST API endpoints
│   ├── services/               # Business logic
│   ├── models/                 # Request/Response models
│   └── middleware/             # Authentication, CORS, etc.
│
├── frontend/                    # Web interface
│   ├── templates/              # HTML templates
│   │   ├── base.html
│   │   ├── dashboard.html
│   │   ├── prediction.html
│   │   └── evaluation.html
│   └── static/                 # CSS, JS, images
│       ├── css/
│       ├── js/
│       └── img/
│
├── src/                        # Core ML/NLP code
│   ├── data/                  # Data loading
│   ├── retrieval/             # Document retrieval
│   ├── models/                # LLM wrappers
│   ├── prompting/             # Prompt engineering
│   ├── reasoning/             # Abductive reasoning
│   └── evaluation/            # Metrics & evaluation
│
├── config/                     # Configuration files
├── data/                       # Datasets
├── scripts/                    # Utility scripts
├── experiments/                # Model experiments
├── tests/                      # Test suite
├── outputs/                    # Results, logs, models
└── docs/                       # Documentation
```

---

## 🧪 Experiments & Results

### Baseline Results (Dev Set)

| Model | Strategy | Score | Exact Match | Partial Match | Inference Time |
|-------|----------|-------|-------------|---------------|----------------|
| GPT-4 | Zero-shot | 0.623 | 45.2% | 34.2% | 1.2s |
| GPT-4 | Few-shot (5) | 0.712 | 53.8% | 31.6% | 1.8s |
| GPT-4 | CoT | 0.758 | 61.4% | 28.8% | 2.3s |
| GPT-4 + RAG | CoT + Retrieval | 0.824 | 72.1% | 20.6% | 2.8s |
| **Ensemble** | **Multi-model voting** | **0.847** | **75.8%** | **17.8%** | **4.1s** |

*(Results are illustrative - actual performance depends on implementation and dataset)*

---

## 📊 Web Dashboard Features

### 1. Real-Time Dashboard
- System health monitoring
- Live prediction statistics
- Model performance metrics
- Resource utilization graphs

### 2. Prediction Interface
- **Single Prediction**: Interactive form with instant results
- **Batch Prediction**: Upload JSONL files for bulk processing
- **Confidence Visualization**: Bar charts showing option probabilities
- **Reasoning Explanation**: Step-by-step reasoning chain display

### 3. Evaluation Module
- Performance metrics (accuracy, F1, precision, recall)
- Confusion matrix visualization
- Error analysis by category
- Comparison across models

### 4. Data Explorer
- Browse dataset instances
- Filter by topic, answer type, difficulty
- Search functionality
- Export selected instances

### 5. Model Management
- Switch between different models
- Configure model parameters
- View model information
- Monitor API usage and costs

---

## 🔧 API Endpoints

### Prediction Endpoints
```bash
# Single prediction
POST /api/predict
Body: {
  "target_event": "string",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "docs": [{"title": "...", "content": "..."}],
  "model": "gpt-4"  # optional
}

# Batch prediction
POST /api/predict/batch
Body: {
  "instances": [...],
  "model": "gpt-4"
}

# Get prediction status
GET /api/predict/status/<job_id>
```

### Evaluation Endpoints
```bash
# Evaluate predictions
POST /api/evaluate
Body: {
  "predictions": [...],
  "ground_truth": [...]
}

# Get evaluation results
GET /api/evaluate/results/<eval_id>
```

### Data Endpoints
```bash
# Get dataset statistics
GET /api/data/stats?dataset=train

# Get specific instance
GET /api/data/instance/<uuid>

# Search instances
POST /api/data/search
Body: {
  "query": "string",
  "filters": {...}
}
```

### Health & Monitoring
```bash
# Health check
GET /api/health

# System metrics
GET /api/metrics

# API usage statistics
GET /api/stats
```

---

## 🧩 Code Examples

### Using the Data Loader
```python
from src.data.loader import AERDataLoader

# Load training data
loader = AERDataLoader('data/train')
instances = loader.load()

# Access an instance
inst = instances[0]
print(f"Event: {inst.target_event}")
print(f"Options: {inst.get_options_dict()}")
print(f"Answer: {inst.golden_answer}")
print(f"Documents: {len(inst.docs)}")
```

### Document Retrieval
```python
from src.retrieval.retriever import DocumentRetriever
from src.retrieval.embedder import SentenceEmbedder

# Initialize
embedder = SentenceEmbedder()
retriever = DocumentRetriever(embedder)

# Retrieve relevant documents
query = "Why did Iran issue a travel ban?"
relevant_docs = retriever.retrieve(query, top_k=5)
```

### Making Predictions
```python
from src.models.llm_wrapper import LLMWrapper
from src.reasoning.abductive import AbductiveReasoner

# Initialize
model = LLMWrapper(model_name="gpt-4")
reasoner = AbductiveReasoner(model)

# Predict
prediction = reasoner.reason(
    target_event="Iran issued travel ban",
    options={"A": "...", "B": "...", "C": "...", "D": "..."},
    docs=[...]
)
print(f"Prediction: {prediction}")
```

### Flask Service Integration
```python
from app.services.prediction_service import PredictionService

# In your Flask route
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    service = PredictionService()
    result = service.predict(
        target_event=data['target_event'],
        options=data['options'],
        docs=data.get('docs', [])
    )
    return jsonify(result)
```

---

## 📈 Performance Optimization

- **Caching Strategy**: 
  - Embeddings cached to disk (FAISS index)
  - API responses cached in Redis (optional)
  - LRU cache for frequent queries
  
- **Batch Processing**: 
  - Parallel processing with ThreadPoolExecutor
  - Batch API calls to reduce overhead
  
- **Efficient Retrieval**: 
  - FAISS indexing for O(log n) search
  - Pre-computed document embeddings
  
- **API Cost Optimization**: 
  - Smart prompt truncation
  - Response caching
  - Token usage monitoring

---

## 🧪 Testing
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov=app --cov-report=html

# Run specific test module
pytest tests/test_api.py -v

# Run integration tests only
pytest tests/ -m integration

# Run with parallel execution
pytest tests/ -n auto
```

---

## 🐳 Docker Deployment
```bash
# Build image
docker build -t aer-system .

# Run container
docker run -p 5000:5000 --env-file .env aer-system

# Using docker-compose
docker-compose up -d
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Write tests for new features (aim for >80% coverage)
- Update documentation
- Run `black` and `pylint` before committing
- Use meaningful commit messages

---

## 📝 Citation

If you use this project in your research, please cite:
```bibtex
@misc{aer-project-2025,
  title={Abductive Event Reasoning: A Web-Based Framework for Causal Inference},
  author={Chandan Kumar},
  year={2025},
  publisher={GitHub},
  url={https://github.com/chandank013/aer-project},
  note={SemEval 2026 Task 12}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **SemEval 2026 Task 12** organizers for the dataset and task definition
- **Hugging Face** for Transformers library and model hosting
- **OpenAI** and **Anthropic** for LLM API access
- **Flask** community for the excellent web framework
- **Bootstrap** for responsive UI components
- All open-source contributors whose libraries made this possible

---

## 📧 Contact

**Chandan Kumar**  
- GitHub: [@chandank013](https://github.com/chandank013)
- Email: chandank013@gmail.com
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/chandank013)

**Project Link**: [https://github.com/chandank013/aer-project](https://github.com/chandank013/aer-project)

---

## 🔗 Related Resources

- [SemEval 2026 Task 12 Dataset](https://github.com/sooo66/semeval2026-task12-dataset)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Documentation](https://faiss.ai/)
- [Bootstrap Documentation](https://getbootstrap.com/)

---

## 🏷️ Repository Topics
```
nlp machine-learning deep-learning llm gpt-4 claude flask web-application 
retrieval-augmented-generation rag chain-of-thought abductive-reasoning 
causal-inference semeval python pytorch transformers langchain faiss 
prompt-engineering few-shot-learning ensemble-learning rest-api 
html-css-javascript bootstrap data-visualization
```

---

## 📊 Project Statistics

![GitHub stars](https://img.shields.io/github/stars/chandank013/aer-project?style=social)
![GitHub forks](https://img.shields.io/github/forks/chandank013/aer-project?style=social)
![GitHub issues](https://img.shields.io/github/issues/chandank013/aer-project)
![GitHub pull requests](https://img.shields.io/github/issues-pr/chandank013/aer-project)
![GitHub last commit](https://img.shields.io/github/last-commit/chandank013/aer-project)

---

<div align="center">
  <strong>⭐ Star this repository if you find it helpful!</strong>
  <br><br>
  Made with ❤️ for SemEval 2026 Task 12
</div>