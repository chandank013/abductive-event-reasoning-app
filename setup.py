"""
Project Setup Script
Creates the complete directory structure for the AER System
"""

import os
import json

def create_directories():
    """Create all necessary directories"""
    directories = [
        '.ebextensions',
        'dataset',
        'models',
        'notebooks',
        'static',
        'static/images',
        'templates'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def create_sample_data():
    """Create sample data files"""
    
    # Sample questions
    sample_questions = [
        {
            "topic_id": 1,
            "question": "The Iranian government issued an intercity travel ban and closed schools in several provinces.",
            "option_A": "U.S. port closures affected shipping",
            "option_B": "COVID-19 pandemic lockdown measures",
            "option_C": "Economic sanctions implementation",
            "option_D": "Insufficient information",
            "answer": "B"
        },
        {
            "topic_id": 2,
            "question": "Major airlines cancelled hundreds of flights across Europe.",
            "option_A": "Severe weather conditions",
            "option_B": "Air traffic control strike",
            "option_C": "Volcanic ash cloud",
            "option_D": "Insufficient information",
            "answer": "A,B"
        }
    ]
    
    # Sample documents
    sample_docs = [
        {
            "topic_id": 1,
            "docs": [
                {
                    "title": "Iran COVID-19 Response",
                    "content": "Iran implemented strict lockdown measures including travel bans and school closures to combat the spread of COVID-19. The government announced these measures would affect multiple provinces."
                },
                {
                    "title": "Economic Situation",
                    "content": "Economic sanctions have impacted Iran's economy significantly in recent years."
                }
            ]
        },
        {
            "topic_id": 2,
            "docs": [
                {
                    "title": "European Weather Alert",
                    "content": "Severe storms across Europe caused major disruptions to air travel with hundreds of flights cancelled."
                },
                {
                    "title": "Labor Disputes",
                    "content": "Air traffic controllers in several European countries have announced strike action."
                }
            ]
        }
    ]
    
    # Save as JSONL
    with open('dataset/question.jsonl', 'w', encoding='utf-8') as f:
        for question in sample_questions:
            f.write(json.dumps(question) + '\n')
    print("✓ Created sample questions: dataset/question.jsonl")
    
    with open('dataset/docs.jsonl', 'w', encoding='utf-8') as f:
        for doc in sample_docs:
            f.write(json.dumps(doc) + '\n')
    print("✓ Created sample documents: dataset/docs.jsonl")

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# PyTorch
*.pth
*.ckpt

# Flask
instance/
.webassets-cache

# Environment variables
.env
.env.local

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Models (large files)
models/*.pth
models/*.bin

# Data
dataset/*.csv
dataset/*.pkl

# Logs
*.log
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    print("✓ Created .gitignore")

def create_config_file():
    """Create configuration file"""
    config_content = """# AER System Configuration

# Flask Settings
DEBUG = True
SECRET_KEY = 'your-secret-key-change-in-production'

# Model Settings
MODEL_PATH = 'models/random_forest_new2.pkl'
MAX_LENGTH = 512
BATCH_SIZE = 8

# Training Settings
LEARNING_RATE = 2e-5
EPOCHS = 4
DROPOUT = 0.3

# API Settings
API_RATE_LIMIT = 100  # requests per hour
"""
    
    with open('config.py', 'w') as f:
        f.write(config_content)
    print("✓ Created config.py")

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*70)
    print("🎉 PROJECT SETUP COMPLETE!")
    print("="*70)
    print("\nNext Steps:")
    print("\n1. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("\n2. Add your actual data:")
    print("   - Replace dataset/question.jsonl with your data")
    print("   - Replace dataset/docs.jsonl with your documents")
    print("\n3. Run preprocessing:")
    print("   jupyter notebook notebooks/EDA.ipynb")
    print("\n4. Train models:")
    print("   jupyter notebook notebooks/Model_Training.ipynb")
    print("\n5. Start the application:")
    print("   python application.py")
    print("\n6. Open browser:")
    print("   http://localhost:5000")
    print("\n" + "="*70)
    print("\n📚 Check README.md for detailed documentation")
    print("="*70 + "\n")

def main():
    """Main setup function"""
    print("\n" + "="*70)
    print("🚀 Setting up Abductive Event Reasoning System")
    print("="*70 + "\n")
    
    # Create directories
    print("Creating directory structure...")
    create_directories()
    print()
    
    # Create sample data
    print("Creating sample data files...")
    create_sample_data()
    print()
    
    # Create .gitignore
    print("Creating configuration files...")
    create_gitignore()
    create_config_file()
    print()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()