from __future__ import annotations

import time
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = str(uuid.uuid4())
        start = time.time()

        response = await call_next(request)

        duration_ms = int((time.time() - start) * 1000)
        # structured-ish log (safe for docker logs)
        print(
            f'{{"request_id":"{rid}","method":"{request.method}","path":"{request.url.path}",'
            f'"status":{response.status_code},"duration_ms":{duration_ms}}}'
        )
        response.headers["X-Request-ID"] = rid
        return response
