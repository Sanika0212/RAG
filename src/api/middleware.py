"""Security middleware for the RAG API."""

import re
from typing import Optional

import structlog
from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)


# Prompt injection patterns to detect
INJECTION_PATTERNS = [
    # Direct instruction overrides
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"forget\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"override\s+(all\s+)?instructions?",

    # Role manipulation
    r"you\s+are\s+(now|actually)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"act\s+as\s+(if|a|an)",
    r"switch\s+(to|into)\s+\w+\s+mode",

    # System prompt extraction
    r"(show|reveal|display|print|output)\s+(your\s+)?(system|initial)\s+prompt",
    r"what\s+(is|are)\s+your\s+(system\s+)?instructions?",
    r"repeat\s+(your\s+)?(system\s+)?prompt",

    # Code/format injection
    r"```\s*system",
    r"<\s*system\s*>",
    r"<\s*admin\s*>",
    r"\[\s*SYSTEM\s*\]",

    # Jailbreak attempts
    r"dan\s+mode",
    r"developer\s+mode",
    r"jailbreak",
    r"bypass\s+(safety|content\s+)?filter",

    # Dangerous operations
    r"(execute|run|eval)\s+(this\s+)?(code|script|command)",
    r"(delete|drop|truncate)\s+(all\s+)?(data|database|table)",
]

# Compiled patterns for efficiency
_compiled_patterns = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def detect_prompt_injection(text: str) -> tuple[bool, Optional[str]]:
    """Detect potential prompt injection attacks in text.

    Args:
        text: Input text to check

    Returns:
        Tuple of (is_suspicious, matched_pattern)
    """
    for pattern in _compiled_patterns:
        match = pattern.search(text)
        if match:
            return True, match.group(0)
    return False, None


def sanitize_input(text: str) -> str:
    """Sanitize user input by removing potentially dangerous content.

    Args:
        text: Input text to sanitize

    Returns:
        Sanitized text
    """
    # Remove control characters
    sanitized = "".join(char for char in text if ord(char) >= 32 or char in "\n\t")

    # Limit consecutive whitespace
    sanitized = re.sub(r"\s{10,}", " " * 9, sanitized)

    # Remove null bytes
    sanitized = sanitized.replace("\x00", "")

    return sanitized


class PromptInjectionMiddleware(BaseHTTPMiddleware):
    """Middleware to detect and block prompt injection attempts."""

    def __init__(
        self,
        app,
        block_on_detection: bool = True,
        log_only: bool = False,
    ):
        """Initialize the middleware.

        Args:
            app: FastAPI application
            block_on_detection: Whether to block requests with detected injections
            log_only: If True, only log detections without blocking
        """
        super().__init__(app)
        self.block_on_detection = block_on_detection
        self.log_only = log_only

    async def dispatch(self, request: Request, call_next):
        """Process the request and check for injections."""
        # Only check POST/PUT requests with JSON bodies
        if request.method in ["POST", "PUT"]:
            content_type = request.headers.get("content-type", "")

            if "application/json" in content_type:
                try:
                    # Read and cache the body
                    body = await request.body()

                    # Check for injection patterns in the raw body
                    body_text = body.decode("utf-8", errors="ignore")
                    is_suspicious, matched = detect_prompt_injection(body_text)

                    if is_suspicious:
                        logger.warning(
                            "Potential prompt injection detected",
                            path=request.url.path,
                            matched_pattern=matched,
                            client_ip=request.client.host if request.client else "unknown",
                        )

                        if self.block_on_detection and not self.log_only:
                            raise HTTPException(
                                status_code=400,
                                detail="Request blocked: potentially malicious content detected",
                            )

                except Exception as e:
                    if isinstance(e, HTTPException):
                        raise
                    logger.error("Error in injection detection", error=str(e))

        response = await call_next(request)
        return response


class TenantMiddleware(BaseHTTPMiddleware):
    """Middleware to extract and validate tenant ID from requests."""

    TENANT_HEADER = "X-Tenant-ID"

    async def dispatch(self, request: Request, call_next):
        """Extract tenant ID and add to request state."""
        tenant_id = request.headers.get(self.TENANT_HEADER)

        if tenant_id:
            # Validate tenant ID format (UUID)
            uuid_pattern = re.compile(
                r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
                re.IGNORECASE,
            )
            if not uuid_pattern.match(tenant_id):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid tenant ID format",
                )

        # Store tenant ID in request state
        request.state.tenant_id = tenant_id

        response = await call_next(request)
        return response


def get_tenant_id(request: Request) -> Optional[str]:
    """Get the tenant ID from request state.

    Args:
        request: FastAPI request object

    Returns:
        Tenant ID or None
    """
    return getattr(request.state, "tenant_id", None)
