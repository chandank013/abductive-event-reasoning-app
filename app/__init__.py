"""
Flask application factory
"""

from flask import Flask
from flask_cors import CORS
from pathlib import Path
import os
import logging

from src.utils.logger import setup_logger

logger = setup_logger("flask_app", log_dir="outputs/logs")


def create_app(config_name: str = 'development', api_only: bool = False):
    """
    Create and configure Flask application
    
    Args:
        config_name: Configuration name ('development', 'production', 'testing')
        api_only: If True, only setup API routes (no web interface)
        
    Returns:
        Flask app instance
    """
    # Create Flask app
    app = Flask(
        __name__,
        template_folder='../frontend/templates',
        static_folder='../frontend/static'
    )
    
    # Load configuration
    app.config.from_object(f'config.settings.{config_name.capitalize()}Config')
    
    # Or load from environment
    app.config.update(
        SECRET_KEY=os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production'),
        DEBUG=os.getenv('DEBUG', 'False').lower() == 'true',
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
    )
    
    # Enable CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Setup logging
    if not app.debug:
        app.logger.setLevel(logging.INFO)
    
    logger.info(f"Creating Flask app in {config_name} mode")
    
    # Register blueprints
    register_blueprints(app, api_only=api_only)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Register template filters
    register_template_filters(app)
    
    # Initialize extensions
    initialize_extensions(app)
    
    logger.info("Flask app created successfully")
    
    return app


def register_blueprints(app: Flask, api_only: bool = False):
    """Register Flask blueprints"""
    from app.api.routes import api_bp
    
    # Always register API
    app.register_blueprint(api_bp, url_prefix='/api')
    logger.info("Registered API blueprint")
    
    # Register web routes if not API-only
    if not api_only:
        from app.web.routes import web_bp
        app.register_blueprint(web_bp)
        logger.info("Registered web blueprint")


def register_error_handlers(app: Flask):
    """Register error handlers"""
    from flask import jsonify, render_template, request
    
    @app.errorhandler(404)
    def not_found(error):
        if request.path.startswith('/api/'):
            return jsonify({'error': 'Not found'}), 404
        return render_template('error.html', error_code=404, error_message='Page not found'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal error: {error}")
        if request.path.startswith('/api/'):
            return jsonify({'error': 'Internal server error'}), 500
        return render_template('error.html', error_code=500, error_message='Internal server error'), 500
    
    @app.errorhandler(413)
    def request_entity_too_large(error):
        if request.path.startswith('/api/'):
            return jsonify({'error': 'File too large'}), 413
        return render_template('error.html', error_code=413, error_message='File too large'), 413
    
    logger.info("Registered error handlers")


def register_template_filters(app: Flask):
    """Register custom template filters"""
    
    @app.template_filter('datetime')
    def format_datetime(value, format='%Y-%m-%d %H:%M'):
        """Format datetime object"""
        if value is None:
            return ""
        return value.strftime(format)
    
    @app.template_filter('truncate_text')
    def truncate_text(text, length=100):
        """Truncate text to specified length"""
        if len(text) <= length:
            return text
        return text[:length] + "..."
    
    logger.info("Registered template filters")


def initialize_extensions(app: Flask):
    """Initialize Flask extensions"""
    # Add any extensions here (e.g., Flask-SQLAlchemy, Flask-Login, etc.)
    logger.info("Extensions initialized")


# Create default app instance
app = create_app()