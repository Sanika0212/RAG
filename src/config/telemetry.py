"""OpenTelemetry instrumentation for observability."""

import os
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

import structlog

from src.config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

# Try to import opentelemetry packages - make them optional
TELEMETRY_AVAILABLE = False
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace import Status, StatusCode
    TELEMETRY_AVAILABLE = True
except ImportError:
    logger.warning("OpenTelemetry packages not available - telemetry disabled")
    # Create stub classes/objects for when telemetry isn't available
    trace = None
    StatusCode = None

T = TypeVar("T")

# Global tracer
_tracer: Optional[trace.Tracer] = None


def setup_telemetry(
    app=None,
    service_name: str = "self-healing-rag",
    otlp_endpoint: Optional[str] = None,
    enable_console: bool = False,
) -> None:
    """Initialize OpenTelemetry instrumentation.

    Args:
        app: FastAPI application instance
        service_name: Name of the service for traces
        otlp_endpoint: OTLP exporter endpoint (e.g., "http://localhost:4317")
        enable_console: Whether to also export traces to console
    """
    global _tracer

    if not TELEMETRY_AVAILABLE:
        logger.info("Telemetry disabled - OpenTelemetry packages not installed")
        return

    # Check if already initialized
    if _tracer is not None:
        return

    # Get endpoint from environment if not provided
    otlp_endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

    # Create resource
    resource = Resource.create({
        "service.name": service_name,
        "service.version": settings.app_version,
        "deployment.environment": settings.environment,
    })

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Add exporters
    if otlp_endpoint:
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        logger.info("OTLP exporter configured", endpoint=otlp_endpoint)

    if enable_console or settings.debug:
        console_exporter = ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(console_exporter))
        logger.info("Console exporter enabled")

    # Set as global tracer provider
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)

    # Instrument FastAPI
    if app is not None:
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI instrumented")

    # Instrument HTTP client (for Anthropic API calls)
    HTTPXClientInstrumentor().instrument()
    logger.info("HTTPX client instrumented")

    logger.info("OpenTelemetry setup complete", service=service_name)


def instrument_sqlalchemy(engine) -> None:
    """Instrument SQLAlchemy engine for database tracing.

    Args:
        engine: SQLAlchemy engine instance
    """
    if not TELEMETRY_AVAILABLE:
        return
    SQLAlchemyInstrumentor().instrument(engine=engine)
    logger.info("SQLAlchemy instrumented")


class NoOpTracer:
    """No-op tracer for when telemetry is disabled."""

    def start_as_current_span(self, name, **kwargs):
        return NoOpSpan()


class NoOpSpan:
    """No-op span context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_attribute(self, key, value):
        pass

    def set_status(self, status):
        pass

    def record_exception(self, exception):
        pass

    def add_event(self, name, attributes=None):
        pass


_noop_tracer = NoOpTracer()


def get_tracer():
    """Get the global tracer instance."""
    global _tracer
    if not TELEMETRY_AVAILABLE:
        return _noop_tracer
    if _tracer is None:
        # Create a no-op tracer if not initialized
        _tracer = trace.get_tracer("self-healing-rag")
    return _tracer


@contextmanager
def trace_span(
    name: str,
    attributes: Optional[dict[str, Any]] = None,
):
    """Context manager for creating a trace span.

    Args:
        name: Span name
        attributes: Optional attributes to add to the span

    Yields:
        The span object
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value) if not isinstance(value, (str, int, float, bool)) else value)
        try:
            yield span
        except Exception as e:
            if TELEMETRY_AVAILABLE:
                span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def traced(
    name: Optional[str] = None,
    attributes: Optional[dict[str, Any]] = None,
):
    """Decorator for tracing async functions.

    Args:
        name: Custom span name (defaults to function name)
        attributes: Optional static attributes to add

    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        span_name = name or func.__name__

        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            tracer = get_tracer()
            with tracer.start_as_current_span(span_name) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    if TELEMETRY_AVAILABLE:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper
    return decorator


def traced_sync(
    name: Optional[str] = None,
    attributes: Optional[dict[str, Any]] = None,
):
    """Decorator for tracing synchronous functions.

    Args:
        name: Custom span name (defaults to function name)
        attributes: Optional static attributes to add

    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        span_name = name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            tracer = get_tracer()
            with tracer.start_as_current_span(span_name) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    if TELEMETRY_AVAILABLE:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper
    return decorator


def add_span_attributes(**attributes) -> None:
    """Add attributes to the current span.

    Args:
        **attributes: Key-value pairs to add
    """
    if not TELEMETRY_AVAILABLE:
        return
    span = trace.get_current_span()
    if span:
        for key, value in attributes.items():
            span.set_attribute(key, str(value) if not isinstance(value, (str, int, float, bool)) else value)


def add_span_event(name: str, attributes: Optional[dict[str, Any]] = None) -> None:
    """Add an event to the current span.

    Args:
        name: Event name
        attributes: Optional event attributes
    """
    if not TELEMETRY_AVAILABLE:
        return
    span = trace.get_current_span()
    if span:
        span.add_event(name, attributes=attributes or {})


def record_exception(exception: Exception) -> None:
    """Record an exception on the current span.

    Args:
        exception: The exception to record
    """
    if not TELEMETRY_AVAILABLE:
        return
    span = trace.get_current_span()
    if span:
        span.record_exception(exception)
        span.set_status(Status(StatusCode.ERROR, str(exception)))
