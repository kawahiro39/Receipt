from __future__ import annotations

from app.classifier import partial_train


def test_partial_train_preserves_existing_classes() -> None:
    initial_samples = [
        {"text": "first receipt", "label": "A"},
        {"text": "second receipt", "label": "B"},
    ]

    model, metrics = partial_train(initial_samples)
    assert model is not None
    assert metrics["classes"] == ["A", "B"]

    follow_up_samples = [
        {"text": "follow up data", "label": "A"},
    ]

    updated_model, updated_metrics = partial_train(follow_up_samples, existing_model=model)

    assert updated_model is not None
    assert updated_metrics["classes"] == ["A", "B"]
    assert set(updated_model.classifier.classes_) == {"A", "B"}
