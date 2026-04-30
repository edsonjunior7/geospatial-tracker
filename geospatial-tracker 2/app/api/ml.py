"""
API — Endpoints de Machine Learning

Endpoints:
  POST /ml/{entity_id}/train       — Treina todos os modelos para a entidade
  GET  /ml/{entity_id}/predict/next     — Prevê próximo ponto
  GET  /ml/{entity_id}/predict/destination — Prevê destino final
  GET  /ml/{entity_id}/patterns    — Detecta padrões de comportamento
"""

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.database import get_db
from app.models.models import Position, Entity
from app.ml.ml_engine import (
    PositionPoint,
    NextPointPredictor,
    DestinationPredictor,
    PatternDetector,
    get_next_point_model,
    get_destination_model,
    get_pattern_model,
)

router = APIRouter()


# ─── Helper: busca histórico e converte para PositionPoint ───

async def _get_position_points(
    db: AsyncSession,
    entity_id: int,
    limit: int = 200,
) -> list[PositionPoint]:
    result = await db.execute(
        select(Position)
        .where(Position.entity_id == entity_id)
        .order_by(Position.timestamp.asc())
        .limit(limit)
    )
    positions = result.scalars().all()
    return [
        PositionPoint(
            latitude=p.latitude,
            longitude=p.longitude,
            timestamp=p.timestamp,
            speed=p.speed,
            bearing=p.bearing,
        )
        for p in positions
    ]


# ─── Treinar todos os modelos ─────────────────────────────────

@router.post("/{entity_id}/train")
async def train_models(entity_id: int, db: AsyncSession = Depends(get_db)):
    """
    Treina todos os modelos ML para a entidade com base no histórico salvo.
    Chame esse endpoint após acumular posições suficientes.
    """
    # Verifica se entidade existe
    entity = await db.get(Entity, entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entidade não encontrada")

    points = await _get_position_points(db, entity_id)

    if len(points) < 8:
        raise HTTPException(
            status_code=422,
            detail=f"Apenas {len(points)} posições disponíveis. Mínimo: 8. "
                   "Rode o simulador para gerar mais dados."
        )

    results = {}

    # 1. Treina NextPointPredictor
    next_model = get_next_point_model(entity_id)
    results["next_point"] = next_model.train(points)

    # 2. Treina DestinationPredictor
    #    Simula viagens: divide o histórico em segmentos de 10 pontos
    dest_model = get_destination_model(entity_id)
    trips = []
    step = max(10, len(points) // 5)
    for i in range(0, len(points) - step, step):
        start = points[i]
        mids = points[i + 1: i + step - 1]
        end = points[i + step - 1]
        trips.append((start, mids, end))

    if trips:
        results["destination"] = dest_model.train(trips)
    else:
        results["destination"] = {"error": "Histórico insuficiente para extrair viagens"}

    # 3. Treina PatternDetector
    pattern_model = get_pattern_model(entity_id)
    results["patterns"] = pattern_model.train(points)

    return {
        "entity_id": entity_id,
        "entity_name": entity.name,
        "total_points_used": len(points),
        "models": results,
    }


# ─── Prever próximo ponto ─────────────────────────────────────

@router.get("/{entity_id}/predict/next")
async def predict_next_point(entity_id: int, db: AsyncSession = Depends(get_db)):
    """
    Prevê a próxima posição geográfica da entidade.
    Requer que o modelo tenha sido treinado via POST /ml/{entity_id}/train
    """
    next_model = get_next_point_model(entity_id)
    if not next_model.is_trained:
        raise HTTPException(
            status_code=400,
            detail="Modelo não treinado. Execute POST /ml/{entity_id}/train primeiro."
        )

    points = await _get_position_points(db, entity_id, limit=20)
    if len(points) < next_model.WINDOW_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Mínimo {next_model.WINDOW_SIZE} posições necessárias para previsão."
        )

    prediction = next_model.predict(points)
    if not prediction:
        raise HTTPException(status_code=500, detail="Falha ao gerar previsão")

    return {
        "entity_id": entity_id,
        "current_position": {
            "latitude": points[-1].latitude,
            "longitude": points[-1].longitude,
        },
        "predicted_next": {
            "latitude": prediction.latitude,
            "longitude": prediction.longitude,
            "distance_km": prediction.distance_km,
            "confidence": prediction.confidence,
            "confidence_pct": f"{prediction.confidence * 100:.1f}%",
            "method": prediction.method,
        },
    }


# ─── Prever destino final ─────────────────────────────────────

@router.get("/{entity_id}/predict/destination")
async def predict_destination(entity_id: int, db: AsyncSession = Depends(get_db)):
    """
    Prevê o destino final da viagem atual com base em padrões históricos.
    """
    dest_model = get_destination_model(entity_id)
    if not dest_model.is_trained:
        raise HTTPException(
            status_code=400,
            detail="Modelo não treinado. Execute POST /ml/{entity_id}/train primeiro."
        )

    points = await _get_position_points(db, entity_id, limit=20)
    if not points:
        raise HTTPException(status_code=422, detail="Nenhuma posição disponível")

    current = points[-1]
    recent = points[-10:] if len(points) >= 10 else points

    prediction = dest_model.predict(current, recent)
    if not prediction:
        raise HTTPException(status_code=500, detail="Falha ao gerar previsão de destino")

    return {
        "entity_id": entity_id,
        "current_position": {
            "latitude": current.latitude,
            "longitude": current.longitude,
        },
        "predicted_destination": {
            "latitude": prediction.latitude,
            "longitude": prediction.longitude,
            "estimated_distance_km": prediction.estimated_distance_km,
            "confidence": prediction.confidence,
            "confidence_pct": f"{prediction.confidence * 100:.1f}%",
            "method": prediction.method,
        },
    }


# ─── Detectar padrões de comportamento ───────────────────────

@router.get("/{entity_id}/patterns")
async def get_behavior_patterns(entity_id: int, db: AsyncSession = Depends(get_db)):
    """
    Detecta e retorna padrões de comportamento identificados no histórico.
    Usa K-Means clustering sobre velocidade, direção e horário.
    """
    pattern_model = get_pattern_model(entity_id)
    if not pattern_model.is_trained:
        raise HTTPException(
            status_code=400,
            detail="Modelo não treinado. Execute POST /ml/{entity_id}/train primeiro."
        )

    # Classifica o ponto mais recente
    points = await _get_position_points(db, entity_id, limit=5)
    current_pattern = None
    if points:
        current_pattern = pattern_model.predict_pattern(points[-1])

    return {
        "entity_id": entity_id,
        "total_patterns_detected": len(pattern_model.cluster_profiles),
        "detected_patterns": pattern_model.cluster_profiles,
        "current_movement_pattern": current_pattern,
    }
