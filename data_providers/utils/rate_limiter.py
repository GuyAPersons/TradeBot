import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import threading

class RateLimiter:
    """
    Rate limiter to prevent exceeding API rate limits.
    Implements token bucket algorithm.
    """
    
    def __init__(self, max_tokens: int, refill_rate: float, initial_tokens: Optional[int] = None):
        """
        Initialize the rate limiter.
        
        Args:
            max_tokens: Maximum number of tokens in the bucket
            refill_rate: Tokens added per second
            initial_tokens: Initial number of tokens (defaults to max_tokens)
        """
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self.tokens = initial_tokens if initial_tokens is not None else max_tokens
        self.last_refill_time = time.time()
        self.lock = threading.Lock()
    
    def check(self) -> bool:
        """
        Check if a request can be made.
        
        Returns:
            True if request can proceed, False otherwise
        """
        with self.lock:
            self._refill()
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill_time
        new_tokens = elapsed * self.refill_rate
        
        if new_tokens > 0:
            self.tokens = min(self.max_tokens, self.tokens + new_tokens)
            self.last_refill_time = now
    
    def wait_for_token(self) -> None:
        """Wait until a token is available."""
        while not self.check():
            time.sleep(0.01)


class MultiRateLimiter:
    """
    Manages multiple rate limiters for different API endpoints.
    """
    
    def __init__(self):
        """Initialize the multi-rate limiter."""
        self.limiters = {}
        self.lock = threading.Lock()
    
    def add_limiter(self, name: str, max_tokens: int, refill_rate: float) -> None:
        """
        Add a new rate limiter.
        
        Args:
            name: Name of the limiter (e.g., 'market_data', 'orders')
            max_tokens: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        with self.lock:
            self.limiters[name] = RateLimiter(max_tokens, refill_rate)
    
    def check(self, name: str) -> bool:
        """
        Check if a request can be made for a specific limiter.
        
        Args:
            name: Name of the limiter
            
        Returns:
            True if request can proceed, False otherwise
        """
        with self.lock:
            if name in self.limiters:
                return self.limiters[name].check()
            return True  # If limiter doesn't exist, allow the request
    
    def wait_for_token(self, name: str) -> None:
        """
        Wait until a token is available for a specific limiter.
        
        Args:
            name: Name of the limiter
        """
        while not self.check(name):
            time.sleep(0.01)
