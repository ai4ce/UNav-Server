from fastapi import FastAPI
from api import localization, navigation

app = FastAPI(title="UNav Server", version="1.0")

app.include_router(localization.router, prefix="/api", tags=["localization"])
app.include_router(navigation.router,    prefix="/api", tags=["navigation"])

@app.get("/")
def root():
    return {"status": "UNav server is running"}
