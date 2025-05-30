from fastapi import FastAPI
from api.user_api import router as user_router
from api.task_api import router as task_router

app = FastAPI(title="UNav Server", version="1.0")

app.include_router(user_router)
app.include_router(task_router)