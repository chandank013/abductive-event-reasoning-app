"""
Model management and selection
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import json

from src.models.llm_wrapper import create_llm, BaseLLM
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelManager:
    """Manage multiple LLM models"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize model manager
        
        Args:
            config_path: Path to model configuration file
        """
        self.models: Dict[str, BaseLLM] = {}
        self.default_model: Optional[str] = None
        self.config_path = config_path
        
        if config_path and Path(config_path).exists():
            self.load_config(config_path)
        
        logger.info("Model manager initialized")
    
    def register_model(
        self,
        name: str,
        model: BaseLLM,
        set_as_default: bool = False
    ) -> None:
        """
        Register a model
        
        Args:
            name: Model identifier
            model: LLM instance
            set_as_default: Set as default model
        """
        self.models[name] = model
        logger.info(f"Registered model: {name}")
        
        if set_as_default or self.default_model is None:
            self.default_model = name
            logger.info(f"Set default model: {name}")
    
    def create_and_register(
        self,
        name: str,
        model_name: str,
        provider: Optional[str] = None,
        set_as_default: bool = False,
        **kwargs
    ) -> BaseLLM:
        """
        Create and register a model
        
        Args:
            name: Model identifier
            model_name: Provider model name
            provider: Provider ('openai', 'anthropic', etc.)
            set_as_default: Set as default model
            **kwargs: Additional model arguments
            
        Returns:
            Created model instance
        """
        logger.info(f"Creating model: {name} ({model_name})")
        
        try:
            model = create_llm(model_name, provider, **kwargs)
            self.register_model(name, model, set_as_default)
            return model
        except Exception as e:
            logger.error(f"Failed to create model {name}: {e}")
            raise
    
    def get_model(self, name: Optional[str] = None) -> BaseLLM:
        """
        Get a model by name
        
        Args:
            name: Model identifier (uses default if None)
            
        Returns:
            LLM instance
        """
        if name is None:
            name = self.default_model
        
        if name is None:
            raise ValueError("No model specified and no default model set")
        
        if name not in self.models:
            raise ValueError(f"Model not found: {name}")
        
        return self.models[name]
    
    def list_models(self) -> List[str]:
        """List all registered models"""
        return list(self.models.keys())
    
    def get_model_info(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get model information
        
        Args:
            name: Model identifier
            
        Returns:
            Model information dictionary
        """
        model = self.get_model(name)
        
        return {
            'name': name or self.default_model,
            'model_name': model.model_name,
            'temperature': model.temperature,
            'max_tokens': model.max_tokens,
            'available': model.available,
            'usage_stats': model.get_usage_stats()
        }
    
    def get_all_models_info(self) -> List[Dict[str, Any]]:
        """Get information for all models"""
        return [
            self.get_model_info(name)
            for name in self.list_models()
        ]
    
    def load_config(self, config_path: str) -> None:
        """
        Load models from configuration file
        
        Args:
            config_path: Path to JSON configuration file
        """
        logger.info(f"Loading model configuration from {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        for model_config in config.get('models', []):
            try:
                name = model_config['name']
                model_name = model_config['model_name']
                provider = model_config.get('provider')
                is_default = model_config.get('default', False)
                
                # Extract model parameters
                params = {
                    'temperature': model_config.get('temperature', 0.3),
                    'max_tokens': model_config.get('max_tokens', 1000),
                    'cache_responses': model_config.get('cache_responses', True)
                }
                
                self.create_and_register(
                    name=name,
                    model_name=model_name,
                    provider=provider,
                    set_as_default=is_default,
                    **params
                )
            
            except Exception as e:
                logger.error(f"Failed to load model from config: {e}")
    
    def save_config(self, config_path: str) -> None:
        """
        Save current models to configuration file
        
        Args:
            config_path: Path to save JSON configuration
        """
        config = {
            'models': [
                {
                    'name': name,
                    'model_name': model.model_name,
                    'temperature': model.temperature,
                    'max_tokens': model.max_tokens,
                    'cache_responses': model.cache_responses,
                    'default': name == self.default_model
                }
                for name, model in self.models.items()
            ]
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model configuration saved to {config_path}")


# Global model manager instance
_model_manager = None


def get_model_manager() -> ModelManager:
    """Get global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
        
        # Register default models
        try:
            _model_manager.create_and_register(
                name='baseline',
                model_name='mock',
                provider='mock',
                set_as_default=True
            )
        except Exception as e:
            logger.error(f"Failed to register baseline model: {e}")
    
    return _model_manager


def initialize_models(config_path: Optional[str] = None) -> ModelManager:
    """
    Initialize model manager with configuration
    
    Args:
        config_path: Path to model configuration file
        
    Returns:
        ModelManager instance
    """
    global _model_manager
    _model_manager = ModelManager(config_path)
    return _model_manager