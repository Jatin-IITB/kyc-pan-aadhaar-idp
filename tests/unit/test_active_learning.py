import tempfile
from pathlib import Path

from services.active_learning.ground_truth_db import GroundTruthDB
from services.active_learning.retrain_trigger import RetrainTrigger
from services.active_learning.model_registry import ModelRegistry
from services.active_learning.regression_checker import RegressionChecker


def test_ground_truth_ingest_and_stats():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = GroundTruthDB(db_path=tmpdir)
        db.ingest_correction("doc-1", "pan", "pan_number", "ABCDE1234F", "ABCDE1234G", "reviewer1")
        db.ingest_correction("doc-2", "aadhaar", "name", "RAHUL", "RAHUL KUMAR", "reviewer1")

        stats = db.get_stats()
        assert stats["total_corrections"] == 2
        assert stats["by_doc_type"]["pan"] == 1


def test_ground_truth_export():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = GroundTruthDB(db_path=tmpdir)
        db.ingest_correction("doc-1", "pan", "pan_number", "OLD", "NEW", "r1")

        out_path = str(Path(tmpdir) / "export.jsonl")
        result = db.export_training_set(out_path)
        assert result["exported"] == 1
        assert Path(out_path).exists()


def test_retrain_trigger_volume():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = GroundTruthDB(db_path=tmpdir)
        for i in range(100):
            db.ingest_correction(f"doc-{i}", "pan", "name", "old", "new", "r1")

        trigger = RetrainTrigger(correction_threshold=100)
        result = trigger.should_retrain(db)
        assert result["should_retrain"] is True


def test_retrain_trigger_no_trigger():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = GroundTruthDB(db_path=tmpdir)
        db.ingest_correction("doc-1", "pan", "name", "old", "new", "r1")

        trigger = RetrainTrigger(correction_threshold=100)
        result = trigger.should_retrain(db)
        assert result["should_retrain"] is False


def test_model_registry_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ModelRegistry(registry_path=str(Path(tmpdir) / "registry.json"))
        registry.register("pan_detector", "v1.0", "/models/pan_v1.pt", metrics={"f1": 0.90})
        registry.register("pan_detector", "v1.1", "/models/pan_v1.1.pt", metrics={"f1": 0.93})

        registry.promote("pan_detector", "v1.1")
        active = registry.get_active("pan_detector")
        assert active["version"] == "v1.1"

        registry.rollback("pan_detector")
        active = registry.get_active("pan_detector")
        assert active["version"] == "v1.0"


def test_regression_checker():
    checker = RegressionChecker()
    baseline = {"f1": 0.94, "field_scores": {"name": {"f1": 0.95}, "pan_number": {"f1": 0.97}}}
    new = {"f1": 0.91, "field_scores": {"name": {"f1": 0.90}, "pan_number": {"f1": 0.97}}}

    result = checker.check_regression(new, baseline)
    assert result["is_regression"] is True
    assert result["recommendation"] == "ROLLBACK"


def test_regression_checker_no_regression():
    checker = RegressionChecker()
    baseline = {"f1": 0.94, "field_scores": {"name": {"f1": 0.95}}}
    new = {"f1": 0.95, "field_scores": {"name": {"f1": 0.96}}}

    result = checker.check_regression(new, baseline)
    assert result["is_regression"] is False
    assert result["recommendation"] == "PROMOTE"
