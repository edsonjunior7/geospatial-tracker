"""
Testes do ML Engine
"""

import pytest
from datetime import datetime, timedelta
from app.ml.ml_engine import (
    PositionPoint,
    NextPointPredictor,
    DestinationPredictor,
    PatternDetector,
)


def make_points(n: int, base_lat=-23.55, base_lon=-46.63) -> list[PositionPoint]:
    """Gera pontos simulados em linha reta para testes."""
    base_time = datetime(2024, 6, 15, 8, 0, 0)
    points = []
    for i in range(n):
        points.append(PositionPoint(
            latitude=base_lat + i * 0.001,
            longitude=base_lon + i * 0.001,
            timestamp=base_time + timedelta(seconds=i * 30),
            speed=60.0 + (i % 5) * 5,
            bearing=45.0 + (i % 3) * 10,
        ))
    return points


class TestNextPointPredictor:

    def test_train_requires_minimum_points(self):
        model = NextPointPredictor()
        result = model.train(make_points(3))
        assert result["trained"] is False

    def test_train_succeeds_with_enough_points(self):
        model = NextPointPredictor()
        result = model.train(make_points(20))
        assert result["trained"] is True
        assert "mae_km" in result
        assert result["training_points"] == 20

    def test_predict_returns_coordinates(self):
        model = NextPointPredictor()
        points = make_points(20)
        model.train(points)
        prediction = model.predict(points[-10:])
        assert prediction is not None
        assert -90 <= prediction.latitude <= 90
        assert -180 <= prediction.longitude <= 180
        assert 0 < prediction.confidence <= 1.0

    def test_predict_without_training_returns_none(self):
        model = NextPointPredictor()
        result = model.predict(make_points(10))
        assert result is None

    def test_predict_insufficient_window_returns_none(self):
        model = NextPointPredictor()
        points = make_points(20)
        model.train(points)
        result = model.predict(make_points(2))  # menos que WINDOW_SIZE
        assert result is None


class TestDestinationPredictor:

    def _make_trips(self, n_trips: int):
        trips = []
        base = datetime(2024, 6, 15, 8, 0, 0)
        for i in range(n_trips):
            start = PositionPoint(-23.55 + i * 0.01, -46.63, base + timedelta(hours=i), 0, 0)
            mids = make_points(5, -23.54 + i * 0.01, -46.62)
            end = PositionPoint(-23.50 + i * 0.01, -46.60, base + timedelta(hours=i, minutes=30), 0, 0)
            trips.append((start, mids, end))
        return trips

    def test_train_requires_minimum_trips(self):
        model = DestinationPredictor()
        result = model.train(self._make_trips(1))
        assert result["trained"] is False

    def test_train_succeeds(self):
        model = DestinationPredictor()
        result = model.train(self._make_trips(5))
        assert result["trained"] is True
        assert "feature_importances" in result

    def test_predict_returns_valid_coordinates(self):
        model = DestinationPredictor()
        model.train(self._make_trips(5))
        current = PositionPoint(-23.55, -46.63, datetime.now(), 60.0, 45.0)
        recent = make_points(5)
        prediction = model.predict(current, recent)
        assert prediction is not None
        assert -90 <= prediction.latitude <= 90
        assert -180 <= prediction.longitude <= 180
        assert prediction.confidence > 0


class TestPatternDetector:

    def test_train_requires_minimum_points(self):
        model = PatternDetector()
        result = model.train(make_points(5))
        assert result["trained"] is False

    def test_train_detects_clusters(self):
        model = PatternDetector()
        result = model.train(make_points(30))
        assert result["trained"] is True
        assert result["clusters_found"] <= 4
        assert len(result["patterns"]) > 0

    def test_patterns_have_names(self):
        model = PatternDetector()
        model.train(make_points(30))
        for pattern in model.cluster_profiles:
            assert "pattern_name" in pattern
            assert len(pattern["pattern_name"]) > 0

    def test_predict_pattern_returns_cluster(self):
        model = PatternDetector()
        points = make_points(30)
        model.train(points)
        result = model.predict_pattern(points[-1])
        assert result is not None
        assert "cluster_id" in result
        assert "pattern_name" in result

    def test_percentages_sum_to_100(self):
        model = PatternDetector()
        model.train(make_points(30))
        total = sum(p["percentage"] for p in model.cluster_profiles)
        assert abs(total - 100.0) < 1.0  # tolerância de 1%
