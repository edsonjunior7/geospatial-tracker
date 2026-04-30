"""
ML Engine — Módulo de Machine Learning para Rastreamento Geoespacial

Modelos implementados:
  1. NextPointPredictor   — Prevê o próximo ponto da rota (regressão linear sequencial)
  2. DestinationPredictor — Prevê o destino final (Random Forest)
  3. PatternDetector      — Detecta padrões de comportamento (K-Means clustering)

Nota sobre LSTM:
  Em produção usaríamos LSTM (TensorFlow/PyTorch). Aqui usamos scikit-learn
  para manter zero dependências pesadas e foco na lógica de ML geoespacial.
  O conceito é idêntico: aprender sequências de coordenadas.
"""

import math
import pickle
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ─── Diretório para salvar modelos treinados ─────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ─── Estruturas de dados ──────────────────────────────────────

@dataclass
class PositionPoint:
    latitude: float
    longitude: float
    timestamp: datetime
    speed: float = 0.0
    bearing: float = 0.0


@dataclass
class NextPointPrediction:
    latitude: float
    longitude: float
    confidence: float          # 0.0 a 1.0
    distance_km: float         # distância prevista do ponto atual
    method: str = "sequential_regression"


@dataclass
class DestinationPrediction:
    latitude: float
    longitude: float
    confidence: float
    estimated_distance_km: float
    method: str = "random_forest"


@dataclass
class BehaviorPattern:
    cluster_id: int
    pattern_name: str          # ex: "Rota matinal", "Viagem longa"
    avg_speed: float
    avg_distance: float
    typical_hour: int
    frequency: int             # quantas vezes esse padrão apareceu


# ─── Funções auxiliares ───────────────────────────────────────

def _haversine(lat1, lon1, lat2, lon2) -> float:
    """Distância entre dois pontos em km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return R * 2 * math.asin(math.sqrt(a))


def _extract_sequence_features(points: list[PositionPoint]) -> np.ndarray:
    """
    Extrai features de uma sequência de pontos para o modelo sequencial.
    Cada ponto vira um vetor: [lat, lon, speed, bearing, hour_sin, hour_cos]
    Usar sin/cos do horário captura a natureza cíclica do tempo (23h ≈ 0h).
    """
    features = []
    for p in points:
        hour = p.timestamp.hour
        features.append([
            p.latitude,
            p.longitude,
            p.speed,
            p.bearing,
            math.sin(2 * math.pi * hour / 24),   # componente cíclica do horário
            math.cos(2 * math.pi * hour / 24),
        ])
    return np.array(features)


# ═════════════════════════════════════════════════════════════
# 1. NEXT POINT PREDICTOR
#    Aprende padrões sequenciais: dado N pontos anteriores, prevê o próximo
#    Técnica: Ridge Regression sobre features de janela deslizante
# ═════════════════════════════════════════════════════════════

class NextPointPredictor:
    """
    Prevê a próxima posição (lat, lon) com base no histórico recente.

    Funcionamento:
    - Usa janela de WINDOW_SIZE pontos anteriores como features
    - Treina dois modelos separados: um para latitude, outro para longitude
    - Ridge Regression: regressão linear com regularização (evita overfitting)

    Em produção: substituir por LSTM para capturar dependências longas.
    """

    WINDOW_SIZE = 5     # quantos pontos anteriores usar para prever o próximo
    MIN_POINTS = 8      # mínimo de pontos para treinar

    def __init__(self):
        self.model_lat = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0)),
        ])
        self.model_lon = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0)),
        ])
        self.is_trained = False
        self._training_points = 0

    def train(self, points: list[PositionPoint]) -> dict:
        """
        Treina o modelo com o histórico de posições.

        Args:
            points: Lista de posições em ordem cronológica

        Returns:
            Métricas de treinamento
        """
        if len(points) < self.MIN_POINTS:
            return {"error": f"Mínimo {self.MIN_POINTS} pontos necessários", "trained": False}

        features = _extract_sequence_features(points)
        X, y_lat, y_lon = [], [], []

        # Janela deslizante: cada linha de X são W pontos, target é o W+1
        for i in range(self.WINDOW_SIZE, len(features)):
            window = features[i - self.WINDOW_SIZE:i].flatten()
            X.append(window)
            y_lat.append(points[i].latitude)
            y_lon.append(points[i].longitude)

        X = np.array(X)
        self.model_lat.fit(X, y_lat)
        self.model_lon.fit(X, y_lon)
        self.is_trained = True
        self._training_points = len(points)

        # Calcula erro médio de treino (MAE em graus → km)
        pred_lat = self.model_lat.predict(X)
        pred_lon = self.model_lon.predict(X)
        errors = [
            _haversine(y_lat[i], y_lon[i], pred_lat[i], pred_lon[i])
            for i in range(len(X))
        ]
        mae_km = float(np.mean(errors))

        return {
            "trained": True,
            "training_points": len(points),
            "samples_used": len(X),
            "mae_km": round(mae_km, 4),
            "window_size": self.WINDOW_SIZE,
        }

    def predict(self, recent_points: list[PositionPoint]) -> Optional[NextPointPrediction]:
        """
        Prevê o próximo ponto com base nos pontos recentes.

        Args:
            recent_points: Últimos WINDOW_SIZE pontos (mínimo)

        Returns:
            NextPointPrediction ou None se não treinado
        """
        if not self.is_trained:
            return None
        if len(recent_points) < self.WINDOW_SIZE:
            return None

        window_points = recent_points[-self.WINDOW_SIZE:]
        features = _extract_sequence_features(window_points).flatten().reshape(1, -1)

        pred_lat = float(self.model_lat.predict(features)[0])
        pred_lon = float(self.model_lon.predict(features)[0])

        # Confiança: baseada na quantidade de dados de treino
        # Mais dados = mais confiança (máx 95%)
        confidence = min(0.95, 0.5 + (self._training_points / 200))

        last = recent_points[-1]
        distance = _haversine(last.latitude, last.longitude, pred_lat, pred_lon)

        return NextPointPrediction(
            latitude=round(pred_lat, 6),
            longitude=round(pred_lon, 6),
            confidence=round(confidence, 3),
            distance_km=round(distance, 4),
        )

    def save(self):
        path = os.path.join(MODELS_DIR, "next_point_model.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls) -> "NextPointPredictor":
        path = os.path.join(MODELS_DIR, "next_point_model.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return cls()


# ═════════════════════════════════════════════════════════════
# 2. DESTINATION PREDICTOR
#    Prevê o destino final da viagem com Random Forest
#    Features: posição atual, velocidade, bearing, horário
# ═════════════════════════════════════════════════════════════

class DestinationPredictor:
    """
    Prevê para onde a entidade vai chegar ao final da viagem.

    Funcionamento:
    - Aprende associações: "quando sai daqui com essa velocidade/direção → vai para lá"
    - Random Forest: conjunto de árvores de decisão (robusto, sem overfitting)
    - Treina lat e lon separadamente como regressão

    Em produção: adicionar destinos conhecidos como classes (classificação).
    """

    MIN_TRIPS = 3       # mínimo de viagens completas para treinar

    def __init__(self):
        self.model_lat = RandomForestRegressor(n_estimators=50, random_state=42)
        self.model_lon = RandomForestRegressor(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self._n_trips = 0

    def _extract_trip_features(self, start: PositionPoint, mid_points: list[PositionPoint]) -> np.ndarray:
        """Features de uma viagem: ponto inicial + contexto temporal."""
        hour = start.timestamp.hour
        day_of_week = start.timestamp.weekday()
        avg_speed = np.mean([p.speed for p in mid_points]) if mid_points else start.speed

        return np.array([[
            start.latitude,
            start.longitude,
            start.bearing,
            avg_speed,
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            math.sin(2 * math.pi * day_of_week / 7),
            math.cos(2 * math.pi * day_of_week / 7),
        ]])

    def train(self, trips: list[tuple[PositionPoint, list[PositionPoint], PositionPoint]]) -> dict:
        """
        Treina com histórico de viagens completas.

        Args:
            trips: Lista de (ponto_inicial, pontos_intermediários, destino_final)

        Returns:
            Métricas de treinamento
        """
        if len(trips) < self.MIN_TRIPS:
            return {"error": f"Mínimo {self.MIN_TRIPS} viagens necessárias", "trained": False}

        X, y_lat, y_lon = [], [], []
        for start, mids, destination in trips:
            features = self._extract_trip_features(start, mids)
            X.append(features[0])
            y_lat.append(destination.latitude)
            y_lon.append(destination.longitude)

        X = np.array(X)
        X_scaled = self.scaler.fit_transform(X)
        self.model_lat.fit(X_scaled, y_lat)
        self.model_lon.fit(X_scaled, y_lon)
        self.is_trained = True
        self._n_trips = len(trips)

        # Importância das features
        feature_names = ["lat", "lon", "bearing", "avg_speed",
                         "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
        importances = dict(zip(
            feature_names,
            [round(float(i), 3) for i in self.model_lat.feature_importances_]
        ))

        return {
            "trained": True,
            "trips_used": len(trips),
            "feature_importances": importances,
        }

    def predict(self, current: PositionPoint, recent: list[PositionPoint]) -> Optional[DestinationPrediction]:
        if not self.is_trained:
            return None

        features = self._extract_trip_features(current, recent)
        features_scaled = self.scaler.transform(features)

        pred_lat = float(self.model_lat.predict(features_scaled)[0])
        pred_lon = float(self.model_lon.predict(features_scaled)[0])

        confidence = min(0.90, 0.4 + (self._n_trips / 30))
        distance = _haversine(current.latitude, current.longitude, pred_lat, pred_lon)

        return DestinationPrediction(
            latitude=round(pred_lat, 6),
            longitude=round(pred_lon, 6),
            confidence=round(confidence, 3),
            estimated_distance_km=round(distance, 4),
        )

    def save(self):
        path = os.path.join(MODELS_DIR, "destination_model.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls) -> "DestinationPredictor":
        path = os.path.join(MODELS_DIR, "destination_model.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return cls()


# ═════════════════════════════════════════════════════════════
# 3. PATTERN DETECTOR
#    Detecta padrões de comportamento via K-Means Clustering
#    Agrupa posições por: velocidade, distância, horário, direção
# ═════════════════════════════════════════════════════════════

class PatternDetector:
    """
    Detecta padrões de comportamento no histórico de movimento.

    Funcionamento:
    - K-Means: agrupa pontos similares em clusters
    - Cada cluster = um padrão (ex: "deslocamento rápido pela manhã")
    - Nomeia padrões automaticamente com base nas características do cluster

    Aplicação real: identificar rotinas, rotas favoritas, horários de pico.
    """

    N_CLUSTERS = 4
    MIN_POINTS = 10

    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=self.N_CLUSTERS, random_state=42, n_init=10)),
        ])
        self.is_trained = False
        self.cluster_profiles: list[dict] = []

    def _name_pattern(self, avg_speed: float, avg_distance: float, typical_hour: int) -> str:
        """Nomeia automaticamente um padrão com base nas suas características."""
        if avg_speed > 100:
            speed_label = "alta velocidade"
        elif avg_speed > 40:
            speed_label = "velocidade média"
        else:
            speed_label = "baixa velocidade"

        if 5 <= typical_hour < 12:
            time_label = "matinal"
        elif 12 <= typical_hour < 18:
            time_label = "vespertino"
        elif 18 <= typical_hour < 23:
            time_label = "noturno"
        else:
            time_label = "madrugada"

        if avg_distance > 50:
            dist_label = "longa distância"
        elif avg_distance > 10:
            dist_label = "média distância"
        else:
            dist_label = "curta distância"

        return f"Deslocamento {speed_label} {time_label} ({dist_label})"

    def train(self, points: list[PositionPoint]) -> dict:
        """
        Treina o detector de padrões com histórico de posições.

        Args:
            points: Lista de posições com métricas (speed, bearing, etc.)

        Returns:
            Perfis dos clusters detectados
        """
        if len(points) < self.MIN_POINTS:
            return {"error": f"Mínimo {self.MIN_POINTS} pontos necessários", "trained": False}

        # Features para clustering: o que define um "padrão de comportamento"
        X = []
        for p in points:
            hour = p.timestamp.hour
            X.append([
                p.speed,
                p.bearing,
                math.sin(2 * math.pi * hour / 24),
                math.cos(2 * math.pi * hour / 24),
            ])

        X = np.array(X)
        labels = self.model.fit_predict(X)

        # Gera perfil de cada cluster
        self.cluster_profiles = []
        for cluster_id in range(self.N_CLUSTERS):
            mask = labels == cluster_id
            cluster_points = [p for p, m in zip(points, mask) if m]

            if not cluster_points:
                continue

            avg_speed = float(np.mean([p.speed for p in cluster_points]))
            avg_bearing = float(np.mean([p.bearing for p in cluster_points]))
            hours = [p.timestamp.hour for p in cluster_points]
            typical_hour = int(np.bincount(hours).argmax())

            self.cluster_profiles.append({
                "cluster_id": cluster_id,
                "pattern_name": self._name_pattern(avg_speed, avg_speed * 0.1, typical_hour),
                "avg_speed_kmh": round(avg_speed, 1),
                "avg_bearing_deg": round(avg_bearing, 1),
                "typical_hour": typical_hour,
                "frequency": int(np.sum(mask)),
                "percentage": round(float(np.sum(mask)) / len(points) * 100, 1),
            })

        self.is_trained = True
        return {
            "trained": True,
            "points_analyzed": len(points),
            "clusters_found": len(self.cluster_profiles),
            "patterns": self.cluster_profiles,
        }

    def predict_pattern(self, point: PositionPoint) -> Optional[dict]:
        """Classifica um ponto em um dos padrões detectados."""
        if not self.is_trained:
            return None

        hour = point.timestamp.hour
        features = np.array([[
            point.speed,
            point.bearing,
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
        ]])

        cluster_id = int(self.model.predict(features)[0])
        profile = next((p for p in self.cluster_profiles if p["cluster_id"] == cluster_id), None)
        return profile

    def save(self):
        path = os.path.join(MODELS_DIR, "pattern_model.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls) -> "PatternDetector":
        path = os.path.join(MODELS_DIR, "pattern_model.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return cls()


# ─── Instâncias globais (singleton por processo) ──────────────
_next_point_models: dict[int, NextPointPredictor] = {}
_destination_models: dict[int, DestinationPredictor] = {}
_pattern_models: dict[int, PatternDetector] = {}


def get_next_point_model(entity_id: int) -> NextPointPredictor:
    if entity_id not in _next_point_models:
        _next_point_models[entity_id] = NextPointPredictor.load()
    return _next_point_models[entity_id]


def get_destination_model(entity_id: int) -> DestinationPredictor:
    if entity_id not in _destination_models:
        _destination_models[entity_id] = DestinationPredictor()
    return _destination_models[entity_id]


def get_pattern_model(entity_id: int) -> PatternDetector:
    if entity_id not in _pattern_models:
        _pattern_models[entity_id] = PatternDetector()
    return _pattern_models[entity_id]
