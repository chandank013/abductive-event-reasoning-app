"""
LLM wrapper for multiple model providers
Supports OpenAI, Anthropic, and HuggingFace models
"""

import os
import time
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import json

from src.utils.logger import get_logger
from src.utils.cache import ResponseCache

logger = get_logger(__name__)


class BaseLLM(ABC):
    """Base class for LLM wrappers"""
    
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.3,
        max_tokens: int = 1000,
        cache_responses: bool = True
    ):
        """
        Initialize LLM
        
        Args:
            model_name: Name of the model
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            cache_responses: Whether to cache responses
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_responses = cache_responses
        
        if cache_responses:
            self.cache = ResponseCache()
        else:
            self.cache = None
        
        self._total_requests = 0
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info(f"Initialized {self.__class__.__name__} with model: {model_name}")
    
    @abstractmethod
    def _generate_uncached(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text without cache (to be implemented by subclasses)"""
        pass
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate text from prompt (with caching)
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        self._total_requests += 1
        
        # Use default values if not provided
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        # Check cache
        if self.cache_responses:
            cache_key = self._get_cache_key(prompt, temperature=temperature, max_tokens=max_tokens, **kwargs)
            cached_response = self.cache.get(cache_key)
            if cached_response:
                self._cache_hits += 1
                logger.debug("Cache hit")
                return cached_response
            self._cache_misses += 1
            logger.debug("Cache miss")
        
        # Generate response
        response = self._generate_uncached(prompt, temperature, max_tokens, **kwargs)
        
        # Cache response
        if self.cache_responses:
            self.cache.set(cache_key, response)
        
        return response
    
    def _get_cache_key(self, prompt: str, **kwargs) -> str:
        """Generate cache key for prompt and parameters"""
        import hashlib
        # Include model name in cache key
        key_str = f"{self.model_name}:{prompt}:{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_usage_stats(self) -> Dict[str, int]:
        """Get usage statistics"""
        total = self._total_requests
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            'total_requests': total,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate': round(hit_rate, 2)
        }


class OpenAIWrapper(BaseLLM):
    """Wrapper for OpenAI API (GPT models)"""
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OpenAI wrapper
        
        Args:
            model_name: OpenAI model name (gpt-3.5-turbo, gpt-4, etc.)
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
        """
        super().__init__(model_name, **kwargs)
        
        # Set API key
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            logger.warning("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            self.available = False
        else:
            try:
                import openai
                openai.api_key = self.api_key
                self.openai = openai
                self.available = True
                logger.info("OpenAI API configured successfully")
            except ImportError:
                logger.error("openai package not installed. Install with: pip install openai")
                self.available = False
    
    def _generate_uncached(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text using OpenAI API"""
        if not self.available:
            raise ValueError("OpenAI API not configured. Please set OPENAI_API_KEY environment variable.")
        
        try:
            logger.debug(f"Making OpenAI API call to {self.model_name}")
            start_time = time.time()
            
            response = self.openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"OpenAI API call completed in {elapsed_time:.2f}s")
            
            # Extract response
            generated_text = response.choices[0].message.content.strip()
            return generated_text
        
        except Exception as e:
            logger.error(f"Error in OpenAI API call: {e}")
            raise


class AnthropicWrapper(BaseLLM):
    """Wrapper for Anthropic API (Claude models)"""
    
    def __init__(
        self,
        model_name: str = "claude-3-sonnet-20240229",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Anthropic wrapper
        
        Args:
            model_name: Claude model name
            api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
        """
        super().__init__(model_name, **kwargs)
        
        # Set API key
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            logger.warning("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
            self.available = False
        else:
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)
                self.available = True
                logger.info("Anthropic API configured successfully")
            except ImportError:
                logger.error("anthropic package not installed. Install with: pip install anthropic")
                self.available = False
    
    def _generate_uncached(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text using Anthropic API"""
        if not self.available:
            raise ValueError("Anthropic API not configured. Please set ANTHROPIC_API_KEY environment variable.")
        
        try:
            logger.debug(f"Making Anthropic API call to {self.model_name}")
            start_time = time.time()
            
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                **kwargs
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"Anthropic API call completed in {elapsed_time:.2f}s")
            
            # Extract response
            generated_text = message.content[0].text.strip()
            return generated_text
        
        except Exception as e:
            logger.error(f"Error in Anthropic API call: {e}")
            raise


class HuggingFaceWrapper(BaseLLM):
    """Wrapper for HuggingFace models (local or API)"""
    
    def __init__(
        self,
        model_name: str = "facebook/opt-1.3b",
        use_api: bool = False,
        api_key: Optional[str] = None,
        device: str = "cpu",
        **kwargs
    ):
        """
        Initialize HuggingFace wrapper
        
        Args:
            model_name: HuggingFace model name
            use_api: Use HuggingFace Inference API instead of local model
            api_key: HuggingFace API key (if using API)
            device: Device to run model on ('cpu' or 'cuda')
        """
        super().__init__(model_name, **kwargs)
        
        self.use_api = use_api
        self.device = device
        
        if use_api:
            self.api_key = api_key or os.getenv('HUGGINGFACE_TOKEN')
            if not self.api_key:
                logger.warning("HuggingFace API key not found.")
                self.available = False
            else:
                self.available = True
                logger.info("HuggingFace API configured")
        else:
            # Load local model
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                import torch
                
                logger.info(f"Loading HuggingFace model: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                
                if device == "cuda" and torch.cuda.is_available():
                    self.model = self.model.cuda()
                
                self.available = True
                logger.info("HuggingFace model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load HuggingFace model: {e}")
                self.available = False
    
    def _generate_uncached(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text using HuggingFace model"""
        if not self.available:
            raise ValueError("HuggingFace model not configured")
        
        try:
            logger.debug(f"Generating with HuggingFace model: {self.model_name}")
            start_time = time.time()
            
            if self.use_api:
                # Use API
                import requests
                API_URL = f"https://api-inference.huggingface.co/models/{self.model_name}"
                headers = {"Authorization": f"Bearer {self.api_key}"}
                
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "temperature": temperature,
                        "max_new_tokens": max_tokens
                    }
                }
                
                response = requests.post(API_URL, headers=headers, json=payload)
                response.raise_for_status()
                
                result = response.json()
                generated_text = result[0]['generated_text'] if isinstance(result, list) else result['generated_text']
            else:
                # Use local model
                inputs = self.tokenizer(prompt, return_tensors="pt")
                if self.device == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    **kwargs
                )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove the input prompt from generated text
                generated_text = generated_text[len(prompt):].strip()
            
            elapsed_time = time.time() - start_time
            logger.info(f"HuggingFace generation completed in {elapsed_time:.2f}s")
            
            return generated_text
        
        except Exception as e:
            logger.error(f"Error in HuggingFace generation: {e}")
            raise


class MockLLM(BaseLLM):
    """Mock LLM for testing without API keys"""
    
    def __init__(self, model_name: str = "mock-model", **kwargs):
        """Initialize mock LLM"""
        super().__init__(model_name, **kwargs)
        self.available = True
        logger.info("Initialized Mock LLM (for testing)")
    
    def _generate_uncached(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate mock response
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (ignored)
            max_tokens: Maximum tokens (ignored)
            
        Returns:
            Mock response
        """
        logger.debug("Generating mock response")
        
        # Simple mock response based on prompt content
        prompt_lower = prompt.lower()
        
        if "why" in prompt_lower or "cause" in prompt_lower:
            return "Based on the provided evidence, option A appears to be the most plausible cause."
        elif "what" in prompt_lower and "option" in prompt_lower:
            return "A"
        elif "explain" in prompt_lower or "describe" in prompt_lower:
            return "This is a detailed explanation of the concept based on the given context."
        else:
            return "This is a mock response from the test model. In production, this would be replaced with actual LLM output."


def create_llm(
    model_name: str,
    provider: Optional[str] = None,
    **kwargs
) -> BaseLLM:
    """
    Factory function to create LLM instance
    
    Args:
        model_name: Name of the model
        provider: Provider name ('openai', 'anthropic', 'huggingface', 'mock')
                 If None, auto-detect from model name
        **kwargs: Additional arguments for the LLM
        
    Returns:
        LLM instance
    """
    # Auto-detect provider if not specified
    if provider is None:
        if model_name.startswith('gpt'):
            provider = 'openai'
        elif model_name.startswith('claude'):
            provider = 'anthropic'
        elif model_name == 'mock' or model_name == 'baseline':
            provider = 'mock'
        else:
            provider = 'huggingface'
    
    # Create appropriate wrapper
    provider = provider.lower()
    
    if provider == 'openai':
        return OpenAIWrapper(model_name=model_name, **kwargs)
    elif provider == 'anthropic':
        return AnthropicWrapper(model_name=model_name, **kwargs)
    elif provider == 'huggingface':
        return HuggingFaceWrapper(model_name=model_name, **kwargs)
    elif provider == 'mock':
        return MockLLM(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")