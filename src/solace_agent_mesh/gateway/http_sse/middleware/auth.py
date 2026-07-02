"""
HTTP SSE Gateway Bearer Token Authentication Middleware.

This middleware provides **opt-in** Bearer token authentication for the
HTTP/SSE gateway.  When enabled it validates every incoming request against a
shared secret *before* the SSE stream (or any other handler) is opened,
returning a 401 immediately on failure.

Configuration
-------------
The middleware is intentionally disabled by default so that existing
deployments remain completely unaffected.  To enable it, pass a non-empty
``auth_token`` when constructing the middleware **or** set the
``SOLACE_GATEWAY_AUTH_TOKEN`` environment variable.

YAML gateway config example::

    gateway_config:
      # … existing keys …
      auth_token: "${SOLACE_GATEWAY_AUTH_TOKEN}"   # or a literal value

Code integration example::

    import os
    from solace_agent_mesh.gateway.http_sse.middleware.auth import (
        GatewayAuthMiddleware,
    )

    auth_token = gateway_config.get("auth_token") or os.getenv(
        "SOLACE_GATEWAY_AUTH_TOKEN", ""
    )

    # Only wrap the app when a token is configured – zero overhead otherwise.
    if auth_token:
        fastapi_app.add_middleware(GatewayAuthMiddleware, auth_token=auth_token)

Security notes
--------------
* The comparison uses :func:`hmac.compare_digest` to avoid timing-based
  side-channel attacks.
* The ``Authorization`` header value is **never** logged or included in error
  responses to prevent accidental token leakage.
* The middleware short-circuits *before* route handlers execute, so an SSE
  connection is never established for unauthenticated requests.

Skipped paths
-------------
The ``/health`` and ``/metrics`` endpoints are intentionally exempt from
authentication so that infrastructure probes continue to work without
requiring credentials.
"""

from __future__ import annotations

import hmac
import logging
import os
from typing import Sequence

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

log = logging.getLogger(__name__)
LOG_IDENTIFIER = "[GatewayAuth]"

# Paths that are always allowed without authentication.
_DEFAULT_SKIP_PATHS: frozenset[str] = frozenset({"/health", "/metrics"})

# Expected prefix in the Authorization header value.
_BEARER_PREFIX = "Bearer "


class GatewayAuthMiddleware(BaseHTTPMiddleware):
    """Starlette/FastAPI middleware that enforces Bearer token authentication.

    The middleware is deliberately a no-op when ``auth_token`` resolves to an
    empty string (or ``None``), which preserves backwards compatibility with
    deployments that have not configured a token.

    Args:
        app:
            The ASGI application to wrap.
        auth_token:
            The expected secret Bearer token.  When *falsy* the middleware
            becomes a transparent pass-through.  Defaults to the value of the
            ``SOLACE_GATEWAY_AUTH_TOKEN`` environment variable so that the
            token can be supplied at runtime without modifying source code.
        skip_paths:
            An optional collection of exact path strings that bypass
            authentication.  Defaults to ``{"/health", "/metrics"}``.
    """

    def __init__(
        self,
        app,
        *,
        auth_token: str | None = None,
        skip_paths: Sequence[str] | None = None,
    ) -> None:
        super().__init__(app)

        # Resolve the token: explicit argument → env var → empty (disabled).
        resolved = auth_token or os.getenv("SOLACE_GATEWAY_AUTH_TOKEN", "")
        self._auth_token: str = resolved or ""
        self._enabled: bool = bool(self._auth_token)

        self._skip_paths: frozenset[str] = (
            frozenset(skip_paths) if skip_paths is not None else _DEFAULT_SKIP_PATHS
        )

        if self._enabled:
            log.info(
                "%s Bearer token authentication is ENABLED (skip_paths=%s).",
                LOG_IDENTIFIER,
                sorted(self._skip_paths),
            )
        else:
            log.info(
                "%s Bearer token authentication is DISABLED "
                "(no auth_token configured).",
                LOG_IDENTIFIER,
            )

    # ------------------------------------------------------------------
    # BaseHTTPMiddleware interface
    # ------------------------------------------------------------------

    async def dispatch(self, request: Request, call_next) -> Response:
        """Validate the Authorization header before forwarding the request.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware / route handler in the chain.

        Returns:
            A 401 :class:`~starlette.responses.JSONResponse` when
            authentication fails, otherwise the response from the downstream
            handler.
        """
        # Fast-path: middleware disabled → transparent pass-through.
        if not self._enabled:
            return await call_next(request)

        # Fast-path: skip health / metrics probes.
        if request.url.path in self._skip_paths:
            return await call_next(request)

        # Validate the Authorization header.
        failure_reason = self._validate(request)
        if failure_reason is not None:
            log.warning(
                "%s Rejected %s %s – %s (client=%s).",
                LOG_IDENTIFIER,
                request.method,
                request.url.path,
                failure_reason,
                request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "error_description": "A valid Bearer token is required.",
                },
                headers={"WWW-Authenticate": 'Bearer realm="solace-agent-mesh"'},
            )

        return await call_next(request)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate(self, request: Request) -> str | None:
        """Check the Authorization header value.

        Returns:
            ``None`` when the request is authenticated, or a short human-readable
            reason string when it is not (used only in server-side log messages).
        """
        auth_header: str | None = request.headers.get("Authorization")

        if not auth_header:
            return "missing Authorization header"

        if not auth_header.startswith(_BEARER_PREFIX):
            return "Authorization header does not use Bearer scheme"

        provided_token = auth_header[len(_BEARER_PREFIX):]

        # Constant-time comparison to prevent timing attacks.
        if not hmac.compare_digest(
            provided_token.encode("utf-8"),
            self._auth_token.encode("utf-8"),
        ):
            return "invalid token"

        return None
