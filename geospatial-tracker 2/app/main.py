"""
Real-Time Geospatial Tracking System
Entry point da aplicação FastAPI
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.entities import router as entities_router
from app.api.positions import router as positions_router
from app.api.websocket import router as ws_router
from app.api.ml import router as ml_router
from app.db.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa recursos ao subir a aplicação."""
    await init_db()
    yield


app = FastAPI(
    title="Real-Time Geospatial Tracker",
    description="Sistema de rastreamento geoespacial em tempo real",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(entities_router, prefix="/entities", tags=["Entities"])
app.include_router(positions_router, prefix="/positions", tags=["Positions"])
app.include_router(ws_router, tags=["WebSocket"])
app.include_router(ml_router, prefix="/ml", tags=["Machine Learning"])


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "geospatial-tracker"}
