from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

from app.database.database import Base, engine

from app.database import models # noqa: F401 (necesario para registrar Prediction)

app = FastAPI(

title="Emotion Risk Engine",

version="2.0.0",

description="AI API for Emotional Risk Detection"

)

# Crea las tablas si no existen (usa Alembic si prefieres migraciones)

Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://localhost:8443",
        "http://192.168.1.120:8443",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

app.include_router(router)
