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