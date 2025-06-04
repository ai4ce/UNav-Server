# middlewares/log_user_api.py

import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from api.user_api import decode_access_token

class UserAPILoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log each API call with user information, API path, HTTP method, 
    status code, and response time.

    - Extracts user ID and username from JWT token (Authorization header)
    - Logs both authenticated and anonymous calls
    - Designed for FastAPI applications with JWT authentication
    """
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        user_str = "anonymous"

        # Extract JWT token from Authorization header if available
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1]
            try:
                # Decode token to extract user ID and username
                payload = decode_access_token(token)
                user_str = f"id={payload.get('id', '-')}, username={payload.get('sub', '-')}"
            except Exception as e:
                # Log invalid or expired token
                user_str = f"invalid_token({e})"

        # Process the request and calculate processing time
        response = await call_next(request)
        elapsed = time.time() - start_time

        # Output the structured log
        print(
            f"[UNav-API] User: {user_str} | Path: {request.url.path} | "
            f"Method: {request.method} | Status: {response.status_code} | "
            f"Time: {elapsed:.3f}s"
        )
        return response
