#!/usr/bin/env python3
"""
AER Project - Main Entry Point
Run this file to start the complete application

Usage:
    python run.py                    # Start web application
    python run.py --mode setup       # Setup mode (first time)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_project():
    """Run complete project setup"""
    print("=" * 70)
    print("AER Project - Initial Setup")
    print("=" * 70)
    print()
    
    # Step 1: Create directory structure
    print("📁 Step 1/5: Creating directory structure...")
    directories = [
        'config',
        'data/sample', 'data/train', 'data/dev', 'data/test', 'data/embeddings',
        'src/data', 'src/retrieval', 'src/models', 'src/prompting', 
        'src/reasoning', 'src/evaluation', 'src/utils',
        'app/api', 'app/services', 'app/models', 'app/middleware',
        'frontend/templates', 'frontend/static/css', 'frontend/static/js', 
        'frontend/static/img', 'frontend/static/lib',
        'scripts', 'experiments/baseline', 'experiments/retrieval_aug',
        'experiments/chain_of_thought', 'experiments/ensemble',
        'notebooks', 'tests',
        'outputs/predictions', 'outputs/logs', 'outputs/models',
        'outputs/results', 'outputs/cache',
        'docs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        # Create __init__.py for Python packages
        if directory.startswith('src/') or directory.startswith('app/') or directory == 'tests':
            init_file = Path(directory) / '__init__.py'
            if not init_file.exists():
                init_file.touch()
    
    print("   ✓ Directory structure created")
    
    # Step 2: Create .gitkeep files
    print("\n📝 Step 2/5: Creating .gitkeep files...")
    gitkeep_dirs = [
        'data/sample', 'data/train', 'data/dev', 'data/test', 'data/embeddings',
        'outputs/predictions', 'outputs/logs', 'outputs/models',
        'outputs/results', 'outputs/cache',
        'frontend/static/img', 'frontend/static/lib'
    ]
    
    for directory in gitkeep_dirs:
        gitkeep = Path(directory) / '.gitkeep'
        gitkeep.touch()
    
    print("   ✓ .gitkeep files created")
    
    # Step 3: Create .env file
    print("\n🔧 Step 3/5: Setting up environment variables...")
    env_file = Path('.env')
    env_example = Path('.env.example')
    
    if not env_file.exists():
        if env_example.exists():
            import shutil
            shutil.copy(env_example, env_file)
            print("   ✓ .env file created from template")
            print("   ⚠️  Please edit .env file with your API keys!")
        else:
            print("   ⚠️  .env.example not found. Please create .env manually")
    else:
        print("   ✓ .env file already exists")
    
    # Step 4: Install dependencies
    print("\n📦 Step 4/5: Installing dependencies...")
    print("   This may take several minutes...")
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
            check=True,
            capture_output=True
        )
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
            check=True
        )
        print("   ✓ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error installing dependencies: {e}")
        return False
    
    # Step 5: Download NLTK data
    print("\n📥 Step 5/5: Downloading NLTK data...")
    try:
        import nltk
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        print("   ✓ NLTK data downloaded")
    except Exception as e:
        print(f"   ⚠️  NLTK download warning: {e}")
    
    # Test imports
    print("\n🧪 Testing basic imports...")
    try:
        import flask
        import torch
        import transformers
        import pandas
        import numpy
        print("   ✓ All core libraries imported successfully")
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✅ Setup Complete!")
    print("=" * 70)
    print("\n📋 Next Steps:")
    print("   1. Edit .env file with your API keys")
    print("   2. Copy your data files to appropriate directories:")
    print("      - sample_data/ → data/sample/")
    print("      - train_data/ → data/train/")
    print("      - dev_data/ → data/dev/")
    print("   3. Run: python scripts/setup_data.sh")
    print("   4. Run: python scripts/analyze_data.py --data-dir data/train")
    print()
    
    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='AER Project - Abductive Event Reasoning System'
    )
    
    parser.add_argument(
        '--mode',
        choices=['setup', 'web', 'api', 'train'],
        default='setup',
        help='Application mode'
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'setup':
            success = setup_project()
            sys.exit(0 if success else 1)
        else:
            print("Other modes will be implemented in later phases.")
            print("For now, run: python run.py --mode setup")
    
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

    #!/usr/bin/env python3
"""
AER Project - Main Entry Point (Updated for Flask)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_web_app(host='0.0.0.0', port=5000, debug=False):
    """Run Flask web application"""
    from src.utils.logger import setup_logger
    logger = setup_logger('main', log_dir='outputs/logs')
    
    logger.info("=" * 70)
    logger.info("Starting AER Web Application")
    logger.info("=" * 70)
    
    # Import app
    from app import create_app
    app = create_app(config_name='development' if debug else 'production')
    
    logger.info(f"Server starting on http://{host}:{port}")
    logger.info("Press CTRL+C to stop")
    logger.info("=" * 70)
    
    # Run app
    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True
    )


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='AER Project - Abductive Event Reasoning System'
    )
    
    parser.add_argument(
        '--mode',
        choices=['setup', 'web', 'api', 'train'],
        default='web',
        help='Application mode (default: web)'
    )
    
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host address (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port number (default: 5000)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'setup':
            from run import setup_project
            success = setup_project()
            sys.exit(0 if success else 1)
        
        elif args.mode in ['web', 'api']:
            run_web_app(
                host=args.host,
                port=args.port,
                debug=args.debug
            )
        
        elif args.mode == 'train':
            print("Training mode will be implemented in later phases")
        
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        sys.exit(0)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()