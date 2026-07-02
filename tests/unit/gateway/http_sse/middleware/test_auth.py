"""
Unit tests for GatewayAuthMiddleware.

Test strategy
-------------
* All tests exercise the middleware in isolation via FastAPI + Starlette's
  TestClient – no real network, no SAC app, no database.
* A single protected route (GET /api/v1/resource) and the two always-exempt
  routes (/health, /metrics) are registered for each scenario.
* Parametrize over interesting header values so edge cases are explicit.

Coverage targets
----------------
1. Middleware DISABLED (no token configured) → all requests pass through.
2. Missing Authorization header → 401.
3. Wrong scheme (Basic, no prefix, etc.) → 401.
4. Correct scheme but wrong token value → 401.
5. Correct Bearer token → 200.
6. /health and /metrics are exempt even when middleware is enabled.
7. Custom skip_paths override replaces the default set.
8. Token sourced from environment variable when no explicit arg is given.
9. Response on 401 includes WWW-Authenticate header and JSON body.
10. Constant-time comparison does not short-circuit on length mismatch
    (structural regression guard – not a timing test).
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from solace_agent_mesh.gateway.http_sse.middleware.auth import GatewayAuthMiddleware

_VALID_TOKEN = "super-secret-token-abc123"
_WRONG_TOKEN = "definitely-not-the-right-token"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(
    auth_token: str | None = None,
    skip_paths=None,
) -> FastAPI:
    """Build a minimal FastAPI app with the auth middleware applied."""
    app = FastAPI()

    kwargs = {}
    if skip_paths is not None:
        kwargs["skip_paths"] = skip_paths

    app.add_middleware(GatewayAuthMiddleware, auth_token=auth_token, **kwargs)

    @app.get("/api/v1/resource")
    async def protected_resource():
        return {"data": "secret"}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics():
        return "# metrics"

    @app.get("/api/v1/admin")
    async def admin_resource():
        return {"admin": True}

    return app


# ---------------------------------------------------------------------------
# 1. Middleware disabled (no token)
# ---------------------------------------------------------------------------


class TestMiddlewareDisabled:
    """When no auth_token is provided the middleware must be a no-op."""

    @pytest.fixture
    def client(self):
        app = _make_app(auth_token=None)
        return TestClient(app)

    def test_no_auth_header_passes(self, client):
        """Requests without any Authorization header must succeed."""
        response = client.get("/api/v1/resource")
        assert response.status_code == 200

    def test_wrong_token_still_passes(self, client):
        """Even a clearly wrong token must pass when middleware is disabled."""
        response = client.get(
            "/api/v1/resource",
            headers={"Authorization": f"Bearer {_WRONG_TOKEN}"},
        )
        assert response.status_code == 200

    def test_empty_string_token_disables_middleware(self):
        """Explicitly passing an empty string is treated as 'not configured'."""
        app = _make_app(auth_token="")
        c = TestClient(app)
        assert c.get("/api/v1/resource").status_code == 200


# ---------------------------------------------------------------------------
# 2. Missing / malformed Authorization header
# ---------------------------------------------------------------------------


class TestMissingOrMalformedHeader:
    """Requests without a valid Bearer header must receive 401."""

    @pytest.fixture
    def client(self):
        app = _make_app(auth_token=_VALID_TOKEN)
        return TestClient(app, raise_server_exceptions=False)

    def test_no_header_returns_401(self, client):
        response = client.get("/api/v1/resource")
        assert response.status_code == 401

    @pytest.mark.parametrize(
        "header_value",
        [
            "Basic dXNlcjpwYXNz",          # Wrong scheme: Basic
            "Token abc123",                  # Wrong scheme: Token
            _VALID_TOKEN,                    # No scheme prefix at all
            "Bearer",                        # "Bearer" with no trailing space or value
            "bearer " + _VALID_TOKEN,        # Lowercase "bearer" – scheme is case-sensitive per RFC 6750
            "",                              # Empty string
            "   ",                           # Whitespace only
        ],
    )
    def test_bad_header_formats_return_401(self, client, header_value):
        response = client.get(
            "/api/v1/resource",
            headers={"Authorization": header_value},
        )
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# 3. Wrong token value
# ---------------------------------------------------------------------------


class TestWrongToken:
    @pytest.fixture
    def client(self):
        app = _make_app(auth_token=_VALID_TOKEN)
        return TestClient(app, raise_server_exceptions=False)

    def test_wrong_value_returns_401(self, client):
        response = client.get(
            "/api/v1/resource",
            headers={"Authorization": f"Bearer {_WRONG_TOKEN}"},
        )
        assert response.status_code == 401

    def test_prefix_of_valid_token_returns_401(self, client):
        """A token that is a prefix of the real token must be rejected."""
        partial = _VALID_TOKEN[:5]
        response = client.get(
            "/api/v1/resource",
            headers={"Authorization": f"Bearer {partial}"},
        )
        assert response.status_code == 401

    def test_valid_token_with_extra_chars_returns_401(self, client):
        """Valid token with extra trailing characters must be rejected."""
        response = client.get(
            "/api/v1/resource",
            headers={"Authorization": f"Bearer {_VALID_TOKEN}extra"},
        )
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# 4. Valid token
# ---------------------------------------------------------------------------


class TestValidToken:
    @pytest.fixture
    def client(self):
        app = _make_app(auth_token=_VALID_TOKEN)
        return TestClient(app)

    def test_valid_token_returns_200(self, client):
        response = client.get(
            "/api/v1/resource",
            headers={"Authorization": f"Bearer {_VALID_TOKEN}"},
        )
        assert response.status_code == 200
        assert response.json() == {"data": "secret"}

    def test_valid_token_passes_multiple_routes(self, client):
        """Token must work across all protected routes, not just one."""
        for path in ("/api/v1/resource", "/api/v1/admin"):
            response = client.get(
                path,
                headers={"Authorization": f"Bearer {_VALID_TOKEN}"},
            )
            assert response.status_code == 200, f"Failed for path: {path}"


# ---------------------------------------------------------------------------
# 5. Default exempt paths (/health, /metrics)
# ---------------------------------------------------------------------------


class TestDefaultExemptPaths:
    @pytest.fixture
    def client(self):
        app = _make_app(auth_token=_VALID_TOKEN)
        return TestClient(app)

    def test_health_exempt_without_token(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_metrics_exempt_without_token(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_health_exempt_with_wrong_token(self, client):
        response = client.get(
            "/health",
            headers={"Authorization": f"Bearer {_WRONG_TOKEN}"},
        )
        assert response.status_code == 200

    def test_protected_path_not_exempt(self, client):
        """Non-exempt paths must still be protected."""
        response = client.get("/api/v1/resource")
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# 6. Custom skip_paths override
# ---------------------------------------------------------------------------


class TestCustomSkipPaths:
    def test_custom_skip_path_is_exempt(self):
        """A path in the custom skip list must bypass auth."""
        app = _make_app(
            auth_token=_VALID_TOKEN,
            skip_paths=["/api/v1/admin"],
        )
        client = TestClient(app)
        # Custom exempt path: no auth needed.
        assert client.get("/api/v1/admin").status_code == 200

    def test_default_paths_no_longer_exempt_when_custom_override_used(self):
        """When skip_paths is overridden, /health and /metrics are NOT exempt."""
        app = _make_app(
            auth_token=_VALID_TOKEN,
            skip_paths=["/api/v1/admin"],  # Only admin is exempt now
        )
        client = TestClient(app, raise_server_exceptions=False)
        # /health is no longer in the skip list → should require auth.
        assert client.get("/health").status_code == 401

    def test_empty_skip_paths_exempts_nothing(self):
        """An empty skip_paths list means NO paths are exempt."""
        app = _make_app(auth_token=_VALID_TOKEN, skip_paths=[])
        client = TestClient(app, raise_server_exceptions=False)
        assert client.get("/health").status_code == 401
        assert client.get("/metrics").status_code == 401


# ---------------------------------------------------------------------------
# 7. Environment variable fallback
# ---------------------------------------------------------------------------


class TestEnvVarFallback:
    def test_token_sourced_from_env_var(self):
        """When auth_token arg is not given, the env var is used."""
        with patch.dict(os.environ, {"SOLACE_GATEWAY_AUTH_TOKEN": _VALID_TOKEN}):
            app = _make_app(auth_token=None)  # No explicit token
            client = TestClient(app)

            # With no token → 401
            assert client.get("/api/v1/resource").status_code == 401
            # With valid token → 200
            response = client.get(
                "/api/v1/resource",
                headers={"Authorization": f"Bearer {_VALID_TOKEN}"},
            )
            assert response.status_code == 200

    def test_explicit_token_takes_precedence_over_env_var(self):
        """Explicit auth_token constructor arg must override the env var."""
        explicit_token = "explicit-token-wins"
        with patch.dict(
            os.environ, {"SOLACE_GATEWAY_AUTH_TOKEN": "env-var-token-ignored"}
        ):
            app = _make_app(auth_token=explicit_token)
            client = TestClient(app, raise_server_exceptions=False)

            # Env-var token must be rejected.
            assert (
                client.get(
                    "/api/v1/resource",
                    headers={"Authorization": "Bearer env-var-token-ignored"},
                ).status_code
                == 401
            )
            # Explicit token must be accepted.
            assert (
                client.get(
                    "/api/v1/resource",
                    headers={"Authorization": f"Bearer {explicit_token}"},
                ).status_code
                == 200
            )

    def test_no_env_var_and_no_explicit_token_disables_middleware(self):
        """If neither env var nor arg is set, the middleware is disabled."""
        env_without_token = {
            k: v
            for k, v in os.environ.items()
            if k != "SOLACE_GATEWAY_AUTH_TOKEN"
        }
        with patch.dict(os.environ, env_without_token, clear=True):
            app = _make_app(auth_token=None)
            client = TestClient(app)
            assert client.get("/api/v1/resource").status_code == 200


# ---------------------------------------------------------------------------
# 8. Response structure on 401
# ---------------------------------------------------------------------------


class TestUnauthorizedResponseStructure:
    @pytest.fixture
    def client(self):
        app = _make_app(auth_token=_VALID_TOKEN)
        return TestClient(app, raise_server_exceptions=False)

    def test_401_has_www_authenticate_header(self, client):
        response = client.get("/api/v1/resource")
        assert response.status_code == 401
        assert "WWW-Authenticate" in response.headers
        assert "Bearer" in response.headers["WWW-Authenticate"]

    def test_401_body_is_json_with_error_key(self, client):
        response = client.get("/api/v1/resource")
        assert response.status_code == 401
        body = response.json()
        assert "error" in body
        assert body["error"] == "Unauthorized"

    def test_401_body_has_error_description(self, client):
        response = client.get("/api/v1/resource")
        body = response.json()
        assert "error_description" in body

    def test_401_body_does_not_leak_token(self, client):
        """The server's secret token must not appear in the error response."""
        response = client.get(
            "/api/v1/resource",
            headers={"Authorization": f"Bearer {_WRONG_TOKEN}"},
        )
        raw = response.text
        assert _VALID_TOKEN not in raw


# ---------------------------------------------------------------------------
# 9. Constant-time comparison regression guard
# ---------------------------------------------------------------------------


class TestConstantTimeComparison:
    """
    Structural guard: verify that the middleware uses hmac.compare_digest.

    This is a static / import-time check, not a timing test. We confirm that
    the implementation imports and calls hmac.compare_digest so that a future
    refactor doesn't accidentally replace it with ``==``.
    """

    def test_hmac_compare_digest_is_used(self):
        import inspect
        import hmac as _hmac

        from solace_agent_mesh.gateway.http_sse.middleware import auth as auth_module

        source = inspect.getsource(auth_module)
        assert "hmac.compare_digest" in source, (
            "GatewayAuthMiddleware must use hmac.compare_digest for "
            "constant-time token comparison."
        )

    def test_different_length_tokens_return_401(self):
        """Tokens of different lengths must both return 401 (no short-circuit)."""
        app = _make_app(auth_token=_VALID_TOKEN)
        client = TestClient(app, raise_server_exceptions=False)

        short_token = _VALID_TOKEN[:3]
        long_token = _VALID_TOKEN + "aaaaaaaaaaaa"

        for bad_token in (short_token, long_token):
            response = client.get(
                "/api/v1/resource",
                headers={"Authorization": f"Bearer {bad_token}"},
            )
            assert response.status_code == 401, (
                f"Token of different length '{bad_token}' should return 401."
            )


# ---------------------------------------------------------------------------
# 10. POST / non-GET methods are also gated
# ---------------------------------------------------------------------------


class TestAllHttpMethods:
    @pytest.fixture
    def app(self):
        fastapi_app = FastAPI()
        fastapi_app.add_middleware(GatewayAuthMiddleware, auth_token=_VALID_TOKEN)

        @fastapi_app.post("/api/v1/tasks")
        async def create_task():
            return {"id": 42}

        @fastapi_app.put("/api/v1/tasks/1")
        async def update_task():
            return {"updated": True}

        @fastapi_app.delete("/api/v1/tasks/1")
        async def delete_task():
            return {"deleted": True}

        return fastapi_app

    @pytest.mark.parametrize(
        "method,path",
        [
            ("post", "/api/v1/tasks"),
            ("put", "/api/v1/tasks/1"),
            ("delete", "/api/v1/tasks/1"),
        ],
    )
    def test_method_without_token_returns_401(self, app, method, path):
        client = TestClient(app, raise_server_exceptions=False)
        response = getattr(client, method)(path)
        assert response.status_code == 401

    @pytest.mark.parametrize(
        "method,path",
        [
            ("post", "/api/v1/tasks"),
            ("put", "/api/v1/tasks/1"),
            ("delete", "/api/v1/tasks/1"),
        ],
    )
    def test_method_with_valid_token_passes(self, app, method, path):
        client = TestClient(app)
        response = getattr(client, method)(
            path,
            headers={"Authorization": f"Bearer {_VALID_TOKEN}"},
        )
        assert response.status_code == 200
