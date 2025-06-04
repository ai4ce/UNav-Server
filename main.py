from fastapi import FastAPI
from api.user_api import router as user_router
from api.task_api import router as task_router
from core.unav_state import cleanup_sessions
from middlewares.log_user_api import UserAPILoggingMiddleware
import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_task = asyncio.create_task(cleanup_sessions())
    yield
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

app = FastAPI(
    lifespan=lifespan,
    title="UNav Server",
    version="1.0"
)

app.add_middleware(UserAPILoggingMiddleware)
app.include_router(user_router, prefix="/api")
app.include_router(task_router, prefix="/api")
