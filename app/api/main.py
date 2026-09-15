from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="Emotion Risk Engine",
    version="2.0.0",
    description="AI API for Emotional Risk Detection"
)

app.include_router(router)