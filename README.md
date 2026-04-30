<div align="center">

# 🛰️ Real-Time Geospatial Tracking System

**A production-grade backend for tracking moving entities in real time**  
Built with FastAPI · SQLAlchemy · WebSocket · scikit-learn

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![Tests](https://img.shields.io/badge/tests-28%20passed-success?style=flat-square)](./tests)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](./LICENSE)

</div>

---

## 📌 Overview

A distributed backend system that tracks moving entities (vehicles, aircraft, drones) in real time, applying real-world geospatial mathematics and machine learning to predict movement patterns.

Inspired by systems used in **Uber** (driver tracking), **aviation** (flight tracking), and **logistics** (delivery routing).

---

## ✨ Features

- **Real-time tracking** via WebSocket — positions broadcast instantly to all connected clients
- **Geospatial engine** — Haversine distance, bearing calculation, speed derived from coordinates
- **Automatic event detection** — high speed, route change, stopped entity
- **3 Machine Learning models**:
  - `NextPointPredictor` — predicts the next GPS coordinate (Ridge Regression)
  - `DestinationPredictor` — predicts the final destination (Random Forest)
  - `PatternDetector` — clusters movement behavior (K-Means)
- **RESTful API** with full OpenAPI documentation
- **Async database** with SQLAlchemy + SQLite (dev) / PostgreSQL (prod)
- **28 unit tests** covering geo engine and ML models

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  CLIENT / SIMULATOR                     │
│             REST API  ·  WebSocket                      │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                  FASTAPI BACKEND                        │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────────┐  │
│  │  REST API   │  │  WebSocket  │  │   ML Engine    │  │
│  │  /entities  │  │ /ws/posit.. │  │  /ml/{id}/..   │  │
│  │  /positions │  │             │  │                │  │
│  └──────┬──────┘  └──────┬──────┘  └───────┬────────┘  │
│         │                │                 │            │
│  ┌──────▼────────────────▼─────────────────▼────────┐   │
│  │                SERVICE LAYER                     │   │
│  │      TrackingService · EntityService             │   │
│  └──────────────────────┬────────────────────────── ┘   │
│                         │                               │
│  ┌──────────────────────▼───────────────────────────┐   │
│  │              GEO ENGINE (CORE)                   │   │
│  │  Haversine · Bearing · Speed · Event Detection   │   │
│  └──────────────────────┬───────────────────────────┘   │
│                         │                               │
│  ┌──────────────────────▼───────────────────────────┐   │
│  │          DATABASE (SQLAlchemy Async)              │   │
│  │       entities · positions · events              │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## 📐 Geospatial Mathematics

### Haversine Distance
Calculates the great-circle distance between two points on Earth's surface:

```
d = 2r * arcsin( sqrt( sin²(Δφ/2) + cos(φ1)*cos(φ2)*sin²(Δλ/2) ) )
```

### Bearing (Direction)
Calculates the compass angle of movement:

```
θ = atan2( sin(Δλ)*cos(φ2), cos(φ1)*sin(φ2) − sin(φ1)*cos(φ2)*cos(Δλ) )
```

### Speed
```
speed (km/h) = distance (km) / elapsed_time (h)
```

---

## 🤖 Machine Learning

| Model | Algorithm | Task |
|-------|-----------|------|
| `NextPointPredictor` | Ridge Regression + sliding window | Predict next GPS coordinate |
| `DestinationPredictor` | Random Forest Regressor | Predict trip destination |
| `PatternDetector` | K-Means Clustering | Detect movement behavior patterns |

**Feature engineering:**
- Cyclic time encoding: `sin/cos(hour * 2π/24)` — captures the circular nature of time
- Cyclic day encoding: `sin/cos(day * 2π/7)`
- Geospatial features: latitude, longitude, bearing, speed

---

## 📁 Project Structure

```
geospatial-tracker/
├── app/
│   ├── main.py                      # FastAPI entry point
│   ├── api/
│   │   ├── entities.py              # Entity CRUD endpoints
│   │   ├── positions.py             # Position recording + WebSocket broadcast
│   │   ├── websocket.py             # WebSocket endpoint
│   │   └── ml.py                    # ML prediction endpoints
│   ├── core/
│   │   ├── geo_engine.py            # Haversine, bearing, speed, event detection
│   │   └── ws_manager.py            # WebSocket connection manager
│   ├── db/
│   │   └── database.py              # Async SQLAlchemy setup
│   ├── models/
│   │   └── models.py                # ORM: Entity, Position, Event
│   ├── schemas/
│   │   └── schemas.py               # Pydantic v2 validation schemas
│   ├── services/
│   │   └── tracking_service.py      # Business logic / service layer
│   └── ml/
│       └── ml_engine.py             # ML models: Ridge, RandomForest, KMeans
├── scripts/
│   └── simulate_movement.py         # Movement data simulator
├── tests/
│   ├── test_geo_engine.py           # 14 geo engine unit tests
│   └── test_ml_engine.py            # 14 ML engine unit tests
└── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12+

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/geospatial-tracker.git
cd geospatial-tracker

# 2. Create virtual environment
python3.12 -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
uvicorn app.main:app --reload
```

Server runs at: **http://localhost:8000**  
Interactive API docs: **http://localhost:8000/docs**

---

## 📡 API Reference

### Entities

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/entities/` | Create a new trackable entity |
| `GET` | `/entities/` | List all entities |
| `GET` | `/entities/{id}` | Get entity details |
| `GET` | `/entities/{id}/location` | Get current position |
| `GET` | `/entities/{id}/history` | Get position history |
| `GET` | `/entities/{id}/events` | Get detected events |

### Positions

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/positions/` | Record position — auto-calculates distance, speed, bearing and detects events |

### Machine Learning

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ml/{id}/train` | Train all ML models for an entity |
| `GET` | `/ml/{id}/predict/next` | Predict next GPS coordinate |
| `GET` | `/ml/{id}/predict/destination` | Predict trip destination |
| `GET` | `/ml/{id}/patterns` | Get detected behavior patterns |

### WebSocket

Connect to `ws://localhost:8000/ws/positions` to receive real-time updates:

```json
{
  "entity_id": 1,
  "entity_name": "Car 01",
  "latitude": -23.5505,
  "longitude": -46.6333,
  "speed": 65.2,
  "bearing": 180.0,
  "distance": 0.42,
  "timestamp": "2024-01-01T12:00:00",
  "events": ["high_speed"]
}
```

---

## 🎮 Running the Simulator

Open a second terminal:

```bash
source venv/bin/activate
python scripts/simulate_movement.py
```

The simulator sends positions every second along a circular route around São Paulo, Brazil — you can watch metrics update live in the terminal.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_geo_engine.py::TestHaversine::test_same_point_returns_zero    PASSED
tests/test_geo_engine.py::TestHaversine::test_sao_paulo_to_rio           PASSED
tests/test_geo_engine.py::TestHaversine::test_short_distance             PASSED
...
tests/test_ml_engine.py::TestPatternDetector::test_percentages_sum_to_100 PASSED

28 passed in 0.XX s
```

---

## 🔧 Training the ML Models

After the simulator generates position data:

```bash
# Train all models for entity #1
curl -X POST http://localhost:8000/ml/1/train

# Predict next GPS point
curl http://localhost:8000/ml/1/predict/next

# Predict trip destination
curl http://localhost:8000/ml/1/predict/destination

# Get detected behavior patterns
curl http://localhost:8000/ml/1/patterns
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Web Framework | FastAPI 0.115 |
| Database ORM | SQLAlchemy 2.0 (async) |
| Database | SQLite (dev) / PostgreSQL (prod) |
| Real-time | WebSocket (native FastAPI) |
| ML | scikit-learn 1.5, NumPy 1.26 |
| Validation | Pydantic v2 |
| Testing | pytest + pytest-asyncio |
| Server | Uvicorn |

---

## 🔮 Roadmap

- [ ] PostgreSQL production setup
- [ ] Redis Pub/Sub for horizontal scaling
- [ ] JWT authentication
- [ ] Docker + docker-compose
- [ ] LSTM neural network for sequence prediction
- [ ] Live map dashboard

---

## 📄 License

MIT — feel free to use this project for learning and portfolio purposes.

---

<div align="center">
Built with Python · FastAPI · scikit-learn
</div>
