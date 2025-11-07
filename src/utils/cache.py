"""
Response caching utilities
"""

import json
import pickle
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timedelta

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ResponseCache:
    """Simple file-based cache for LLM responses"""
    
    def __init__(
        self,
        cache_dir: str = "outputs/cache",
        ttl_hours: int = 24
    ):
        """
        Initialize cache
        
        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time-to-live in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        
        # In-memory cache
        self._memory_cache = {}
        
        logger.debug(f"Initialized cache at {self.cache_dir}")
    
    def get(self, key: str) -> Optional[str]:
        """
        Get cached response
        
        Args:
            key: Cache key
            
        Returns:
            Cached response or None if not found/expired
        """
        # Check memory cache first
        if key in self._memory_cache:
            data, timestamp = self._memory_cache[key]
            if datetime.now() - timestamp < self.ttl:
                logger.debug(f"Memory cache hit for key: {key[:16]}...")
                return data
            else:
                logger.debug(f"Memory cache expired for key: {key[:16]}...")
                del self._memory_cache[key]
        
        # Check file cache
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data, timestamp = pickle.load(f)
                
                # Check if expired
                if datetime.now() - timestamp < self.ttl:
                    # Add to memory cache
                    self._memory_cache[key] = (data, timestamp)
                    logger.debug(f"File cache hit for key: {key[:16]}...")
                    return data
                else:
                    # Remove expired cache
                    cache_file.unlink()
                    logger.debug(f"File cache expired for key: {key[:16]}...")
            except Exception as e:
                logger.error(f"Error reading cache: {e}")
        
        return None
    
    def set(self, key: str, value: str) -> None:
        """
        Set cached response
        
        Args:
            key: Cache key
            value: Response to cache
        """
        timestamp = datetime.now()
        
        # Store in memory
        self._memory_cache[key] = (value, timestamp)
        logger.debug(f"Cached in memory: {key[:16]}...")
        
        # Store in file
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump((value, timestamp), f)
            logger.debug(f"Cached to file: {key[:16]}...")
        except Exception as e:
            logger.error(f"Error writing cache: {e}")
    
    def clear(self) -> None:
        """Clear all cache"""
        self._memory_cache.clear()
        
        # Clear file cache
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        
        logger.info("Cache cleared")
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        
        return {
            'memory_cache_size': len(self._memory_cache),
            'file_cache_size': len(cache_files),
            'cache_dir': str(self.cache_dir)
        }