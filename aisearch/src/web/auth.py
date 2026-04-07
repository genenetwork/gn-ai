"""Authentication and authorization utilities for ai-search.

This module provides JWT token validation using JWKS from the auth server.
"""

import logging
from datetime import datetime
from functools import wraps
from typing import Optional
from urllib.parse import urljoin

import requests
from authlib.jose import JsonWebKey, JsonWebToken, KeySet
from authlib.jose.errors import BadSignatureError
from quart import request, jsonify, current_app

logger = logging.getLogger(__name__)


class TokenValidationError(Exception):
    """Raised when token validation fails."""
    pass


def fetch_jwks(auth_server_url: str, timeout: int = 30) -> KeySet:
    """Fetch JSON Web Keys from the auth server."""
    jwks_uri = urljoin(auth_server_url, "auth/public-jwks")
    response = requests.get(jwks_uri, timeout=timeout)
    response.raise_for_status()
    jwks_data = response.json()
    keys = [JsonWebKey.import_key(key) for key in jwks_data.get(
        "jwks", {}).get("keys", [])]
    if not keys:
        raise TokenValidationError("No keys found in JWKS response")
    return KeySet(keys)


def get_cached_jwks(auth_server_url: str) -> KeySet:
    """Get cached JWKS from Redis, fetching if necessary."""
    cache = current_app.extensions["cache"]
    cached = cache.get("auth_server_jwks")
    
    if cached is not None:
        return KeySet([JsonWebKey.import_key(key) for key in cached])
    
    # Fetch and cache for 2 hours
    jwks = fetch_jwks(auth_server_url)
    key_data = [key.export() for key in jwks.keys]
    cache.set("auth_server_jwks", key_data, timeout=7200)
    return jwks


def validate_token(token: str, keys: KeySet) -> dict:
    """Validate a JWT token against the provided keys.

    Args:
        token: The JWT token string
        keys: KeySet containing valid public keys

    Returns:
        Decoded JWT payload

    Raises:
        TokenValidationError: If token is invalid or cannot be verified
    """
    for key in keys.keys:
        try:
            jwt = JsonWebToken(["RS256"])
            payload = jwt.decode(token, key)

            # Check expiration
            if payload.get("exp") and datetime.now().timestamp() > payload["exp"]:
                raise TokenValidationError("Token has expired")

            logger.debug("Token validated successfully")
            return payload
        except BadSignatureError:
            continue

    raise TokenValidationError("Token signature could not be verified")


def require_token(func):
    """Decorator that requires a valid bearer token for the endpoint.

    Expects the Authorization header with a Bearer token.
    Adds `auth_token` kwarg to the decorated function with the decoded JWT payload.

    Example:
        @app.route("/api/v1/search")
        @require_token
        async def search(auth_token=None):
            # auth_token contains the decoded JWT payload
            user_id = auth_token.get("sub")
            ...
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        auth_server_url = current_app.config.get("AUTH_SERVER_URL")

        if not auth_server_url:
            logger.error("AUTH_SERVER_URL not configured")
            return jsonify({
                "error": "ConfigurationError",
                "description": "Authentication not properly configured"
            }), 500

        # Extract bearer token from Authorization header
        bearer = request.headers.get("Authorization", "")

        if not bearer.startswith("Bearer "):
            logger.debug("No bearer token provided")
            return jsonify({
                "error": "AuthenticationError",
                "description": "Expected 'Authorization: Bearer <token>' header"
            }), 401

        try:
            _, token = bearer.split(" ", 1)
            token = token.strip()

            # Get cached JWKS and validate token
            jwks = get_cached_jwks(auth_server_url)
            jwt_payload = validate_token(token, jwks)

            # Add auth token info to kwargs
            kwargs["auth_token"] = {
                "access_token": token,
                "jwt": jwt_payload
            }

            return await func(*args, **kwargs)

        except TokenValidationError as exc:
            logger.debug("Token validation failed: %s", exc)
            return jsonify({
                "error": "TokenValidationError",
                "description": str(exc)
            }), 401
        except Exception as exc:
            logger.error(
                "Unexpected error during token validation", exc_info=True)
            return jsonify({
                "error": "InternalError",
                "description": "Failed to validate authentication token"
            }), 500

    return wrapper


def get_user_id(auth_token: Optional[dict] = None) -> Optional[str]:
    """Extract user ID from auth token.

    Args:
        auth_token: The auth token dict with 'jwt' key containing the payload

    Returns:
        User ID (subject claim) or None if not available
    """
    if auth_token is None:
        return None
    jwt_payload = auth_token.get("jwt", {})
    return jwt_payload.get("sub")
