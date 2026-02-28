"""Circuit breaker pattern for LLM API resilience."""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

import anthropic
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service is failing, requests are rejected immediately
    - HALF_OPEN: Testing if service has recovered
    """

    name: str
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close from half-open
    timeout: float = 60.0  # Seconds before trying half-open
    _state: CircuitState = field(default=CircuitState.CLOSED)
    _failure_count: int = field(default=0)
    _success_count: int = field(default=0)
    _last_failure_time: float = field(default=0.0)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def state(self) -> CircuitState:
        """Get current state, transitioning from OPEN to HALF_OPEN if timeout expired."""
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.timeout:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                logger.info(f"Circuit {self.name} transitioning to HALF_OPEN")
        return self._state

    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute a function through the circuit breaker."""
        async with self._lock:
            state = self.state

            if state == CircuitState.OPEN:
                logger.warning(f"Circuit {self.name} is OPEN, rejecting call")
                raise CircuitOpenError(f"Circuit breaker {self.name} is open")

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure(e)
            raise

    async def _on_success(self) -> None:
        """Handle successful call."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info(f"Circuit {self.name} CLOSED after recovery")
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    async def _on_failure(self, error: Exception) -> None:
        """Handle failed call."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit {self.name} back to OPEN after failure in half-open")
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(
                        f"Circuit {self.name} OPENED after {self._failure_count} failures",
                        error=str(error),
                    )

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        logger.info(f"Circuit {self.name} manually reset")


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


# Global circuit breakers for different services
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, **kwargs) -> CircuitBreaker:
    """Get or create a circuit breaker by name."""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name=name, **kwargs)
    return _circuit_breakers[name]


# LLM-specific retry and circuit breaker configuration
LLM_RETRYABLE_ERRORS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
    anthropic.APIStatusError,
    TimeoutError,
    ConnectionError,
)


def llm_retry():
    """Create a retry decorator for LLM calls."""
    return retry(
        retry=retry_if_exception_type(LLM_RETRYABLE_ERRORS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )


class ResilientLLMClient:
    """Wrapper around Anthropic client with circuit breaker and retry logic."""

    def __init__(
        self,
        client: Optional[anthropic.AsyncAnthropic] = None,
        circuit_name: str = "anthropic",
    ):
        """Initialize the resilient LLM client.

        Args:
            client: Anthropic client (created if not provided)
            circuit_name: Name for the circuit breaker
        """
        self._client = client
        self._circuit = get_circuit_breaker(
            circuit_name,
            failure_threshold=5,
            success_threshold=2,
            timeout=60.0,
        )

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        """Get or create the Anthropic client."""
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(
                api_key=settings.anthropic_api_key.get_secret_value()
            )
        return self._client

    async def create_message(
        self,
        model: str,
        max_tokens: int,
        messages: list[dict],
        temperature: float = 0.0,
        system: Optional[str] = None,
        fallback_response: Optional[str] = None,
    ) -> anthropic.types.Message:
        """Create a message with circuit breaker and retry logic.

        Args:
            model: Model name
            max_tokens: Maximum tokens
            messages: Message list
            temperature: Sampling temperature
            system: System prompt
            fallback_response: Response to return if circuit is open

        Returns:
            Message response

        Raises:
            CircuitOpenError: If circuit is open and no fallback provided
        """
        @llm_retry()
        async def _call():
            kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": messages,
                "temperature": temperature,
            }
            if system:
                kwargs["system"] = system
            return await self.client.messages.create(**kwargs)

        try:
            return await self._circuit.call(_call)
        except CircuitOpenError:
            if fallback_response is not None:
                logger.warning(
                    "Circuit open, using fallback response",
                    circuit=self._circuit.name,
                )
                # Return a mock response structure
                return self._create_fallback_message(fallback_response)
            raise

    def _create_fallback_message(self, content: str) -> anthropic.types.Message:
        """Create a fallback message object."""
        # Create a minimal mock response
        from unittest.mock import MagicMock
        msg = MagicMock(spec=anthropic.types.Message)
        msg.content = [MagicMock(text=content)]
        msg.usage = MagicMock(input_tokens=0, output_tokens=0)
        return msg

    @property
    def circuit_state(self) -> CircuitState:
        """Get current circuit state."""
        return self._circuit.state

    def reset_circuit(self) -> None:
        """Reset the circuit breaker."""
        self._circuit.reset()


# Global resilient client instance
_resilient_client: Optional[ResilientLLMClient] = None


def get_resilient_llm_client() -> ResilientLLMClient:
    """Get the global resilient LLM client."""
    global _resilient_client
    if _resilient_client is None:
        _resilient_client = ResilientLLMClient()
    return _resilient_client
