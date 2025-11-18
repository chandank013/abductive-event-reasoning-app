"""
Quick Start Script for AER Multi-Algorithm System
Automated setup and training launcher
"""

import os
import sys
import subprocess
import json

def print_banner(text):
    """Print formatted banner"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current:", f"{version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_requirements():
    """Check if requirements are installed"""
    try:
        import torch
        import transformers
        import sklearn
        import flask
        print("✅ Core packages installed")
        return True
    except ImportError as e:
        print(f"❌ Missing packages: {e}")
        return False

def install_requirements():
    """Install requirements"""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def create_directories():
    """Create necessary directories"""
    dirs = ['dataset', 'models', 'notebooks', 'static', 'static/images', 'templates']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✅ Directories created")

def check_data():
    """Check if data files exist"""
    required_files = ['dataset/question.jsonl', 'dataset/docs.jsonl']
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        print(f"⚠️  Missing data files: {missing}")
        print("   Please add your data files before training")
        return False
    
    # Count instances
    with open('dataset/question.jsonl', 'r') as f:
        n_questions = sum(1 for _ in f)
    
    print(f"✅ Data files found ({n_questions} instances)")
    return True

def run_preprocessing():
    """Run data preprocessing"""
    print("\nRunning preprocessing...")
    print("Opening EDA.ipynb - please execute all cells")
    
    try:
        subprocess.Popen(["jupyter", "notebook", "notebooks/EDA.ipynb"])
        print("✅ Jupyter notebook opened")
        return True
    except FileNotFoundError:
        print("❌ Jupyter not found. Install with: pip install jupyter")
        return False

def run_training():
    """Run model training"""
    print("\nRunning multi-algorithm training...")
    print("Opening Model_Training.ipynb - please execute all cells")
    
    try:
        subprocess.Popen(["jupyter", "notebook", "notebooks/Model_Training.ipynb"])
        print("✅ Training notebook opened")
        return True
    except FileNotFoundError:
        print("❌ Jupyter not found")
        return False

def start_app():
    """Start Flask application"""
    print("\nStarting Flask application...")
    try:
        subprocess.Popen([sys.executable, "application.py"])
        print("✅ Application starting at http://localhost:5000")
        return True
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        return False

def show_menu():
    """Show interactive menu"""
    print("\n" + "="*70)
    print("  🧠 AER MULTI-ALGORITHM SYSTEM - QUICK START")
    print("="*70)
    print("\nWhat would you like to do?")
    print("\n1. 🔧 Setup (Install requirements & create directories)")
    print("2. 📊 Check data files")
    print("3. 🔬 Run preprocessing (EDA)")
    print("4. 🚀 Train models (Multi-algorithm + Hyperparameter tuning)")
    print("5. 🌐 Start Flask application")
    print("6. 📚 Full workflow (Setup → Preprocessing → Training → Launch)")
    print("7. 📖 Show documentation")
    print("8. ❌ Exit")
    print("\n" + "="*70)
    
    choice = input("\nEnter your choice (1-8): ").strip()
    return choice

def show_documentation():
    """Show documentation guide"""
    print("\n" + "="*70)
    print("  📖 DOCUMENTATION")
    print("="*70)
    
    print("\n📁 Project Structure:")
    print("""
    project/
    ├── dataset/
    │   ├── question.jsonl          # Your questions
    │   └── docs.jsonl              # Your documents
    ├── models/
    │   └── random_forest_new2.pkl  # Trained model (after training)
    ├── notebooks/
    │   ├── EDA.ipynb               # Data preprocessing
    │   └── Model_Training.ipynb     # Multi-algorithm training
    ├── static/images/              # Generated visualizations
    ├── templates/                  # HTML templates
    ├── application.py              # Flask backend
    ├── requirements.txt            # Dependencies
    └── README.md                   # Full documentation
    """)
    
    print("\n🎯 6 Models Trained:")
    print("   1. BERT Baseline")
    print("   2. RoBERTa Baseline")
    print("   3. RoBERTa + BiLSTM + Attention ⭐")
    print("   4. Longformer (for long contexts)")
    print("   5. DistilBERT (fast & lightweight)")
    print("   6. Hierarchical Attention ⭐⭐")
    
    print("\n🔧 Hyperparameter Tuning:")
    print("   - Learning Rate: [1e-5, 2e-5, 3e-5]")
    print("   - Dropout: [0.2, 0.3, 0.4]")
    print("   - Hidden Size: [256, 384, 512, 768]")
    print("   - Batch Size: [4, 8, 16, 32]")
    print("   - 4 trials per model = ~24 total trainings")
    
    print("\n⏱️  Expected Time:")
    print("   - Preprocessing: 5-10 minutes")
    print("   - Training: 2-4 hours (GPU) / 8-12 hours (CPU)")
    print("   - Total: ~3-5 hours")
    
    print("\n📊 Expected Results:")
    print("   - Best Model Macro F1: 85-89%")
    print("   - Exact Match: 75-82%")
    print("   - Hamming Accuracy: 90-94%")
    
    print("\n💡 Tips:")
    print("   - Ensure GPU is available for faster training")
    print("   - Start with DistilBERT for quick experiments")
    print("   - Monitor training in Jupyter notebook")
    print("   - Check visualizations in static/images/")
    
    print("\n📚 More Info:")
    print("   - README.md: Complete documentation")
    print("   - Multi-Algorithm Guide: notebooks/Model_Training.ipynb")
    
    input("\nPress Enter to continue...")

def full_workflow():
    """Run complete workflow"""
    print_banner("FULL WORKFLOW")
    
    print("This will:")
    print("  1. Install requirements")
    print("  2. Create directories")
    print("  3. Check data")
    print("  4. Open preprocessing notebook")
    print("  5. Open training notebook")
    print("\nYou will need to:")
    print("  - Execute cells in EDA.ipynb")
    print("  - Execute cells in Model_Training.ipynb (2-4 hours)")
    print("  - Start Flask app when training completes")
    
    confirm = input("\nContinue? (y/n): ").strip().lower()
    if confirm != 'y':
        return
    
    # Step 1: Setup
    print_banner("Step 1: Setup")
    create_directories()
    
    if not check_requirements():
        if not install_requirements():
            return
    
    # Step 2: Check data
    print_banner("Step 2: Check Data")
    if not check_data():
        print("\n⚠️  Please add your data files:")
        print("   - dataset/question.jsonl")
        print("   - dataset/docs.jsonl")
        print("\nThen run this script again.")
        return
    
    # Step 3: Preprocessing
    print_banner("Step 3: Preprocessing")
    print("Opening EDA notebook...")
    print("\n📋 TODO: Execute all cells in the notebook")
    print("         This will preprocess and analyze your data")
    input("\nPress Enter to open notebook...")
    run_preprocessing()
    
    input("\n⏸️  Press Enter once preprocessing is complete...")
    
    # Step 4: Training
    print_banner("Step 4: Multi-Algorithm Training")
    print("Opening training notebook...")
    print("\n📋 TODO: Execute all cells in the notebook")
    print("         This will train 6 models with hyperparameter tuning")
    print("         ⏱️  Expected time: 2-4 hours")
    input("\nPress Enter to open notebook...")
    run_training()
    
    input("\n⏸️  Press Enter once training is complete...")
    
    # Step 5: Launch
    print_banner("Step 5: Launch Application")
    confirm = input("Start Flask application? (y/n): ").strip().lower()
    if confirm == 'y':
        start_app()
        print("\n✅ Application running at http://localhost:5000")
        print("   Press Ctrl+C to stop")
    
    print_banner("WORKFLOW COMPLETE!")
    print("Check your results:")
    print("  - Model: models/random_forest_new2.pkl")
    print("  - Comparison: models/model_comparison_results.csv")
    print("  - Visualizations: static/images/multi_model_comparison.png")

def main():
    """Main function"""
    print("\n" + "="*70)
    print("  🧠 ABDUCTIVE EVENT REASONING - MULTI-ALGORITHM SYSTEM")
    print("  📚 SemEval 2026 Task 12")
    print("="*70)
    
    # Initial checks
    if not check_python_version():
        return
    
    while True:
        choice = show_menu()
        
        if choice == '1':
            print_banner("SETUP")
            create_directories()
            if not check_requirements():
                install_requirements()
        
        elif choice == '2':
            print_banner("DATA CHECK")
            check_data()
        
        elif choice == '3':
            print_banner("PREPROCESSING")
            if not check_data():
                print("⚠️  Please add data files first")
                continue
            run_preprocessing()
        
        elif choice == '4':
            print_banner("TRAINING")
            if not os.path.exists('dataset/preprocessed_data.pkl'):
                print("⚠️  Please run preprocessing first")
                continue
            run_training()
        
        elif choice == '5':
            print_banner("START APPLICATION")
            if not os.path.exists('models/random_forest_new2.pkl'):
                print("⚠️  Please train models first")
                continue
            start_app()
        
        elif choice == '6':
            full_workflow()
        
        elif choice == '7':
            show_documentation()
        
        elif choice == '8':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()