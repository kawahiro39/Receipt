"""Receipt classification helpers using scikit-learn primitives."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

TextFeatures = Dict[str, Any]
Sample = Dict[str, Any]

DEFAULT_ALTERNATIVES = [
    ("事務用品費", 0.5),
    ("雑費", 0.4),
]


@dataclass
class ModelBundle:
    """Container holding the artefacts required for inference."""

    vectorizer: TfidfVectorizer
    classifier: SGDClassifier


def build_or_load_vectorizer(existing: Optional[TfidfVectorizer] = None) -> TfidfVectorizer:
    """Return a TF-IDF vectorizer for receipt text features."""

    if existing is not None:
        return existing
    return TfidfVectorizer(ngram_range=(1, 2), min_df=1)


def build_classifier(existing: Optional[SGDClassifier] = None) -> SGDClassifier:
    """Return a linear classifier suited for incremental learning."""

    if existing is not None:
        return existing
    return SGDClassifier(loss="log_loss", max_iter=5, tol=1e-3)


def _compose_text(features: TextFeatures) -> str:
    parts = [features.get("raw_text") or ""]
    merchant = features.get("merchant")
    if isinstance(merchant, dict):
        merchant = merchant.get("value")
    if merchant:
        parts.append(str(merchant))
    amount = features.get("amount")
    if isinstance(amount, dict):
        amount = amount.get("value")
    if amount is not None:
        parts.append(str(amount))
    return "\n".join(part for part in parts if part)


def _softmax(scores: Sequence[float]) -> List[float]:
    max_score = max(scores)
    exps = [math.exp(s - max_score) for s in scores]
    denom = sum(exps) or 1.0
    return [val / denom for val in exps]


def predict_category(features: TextFeatures, model: Optional[ModelBundle]) -> Tuple[str, float, List[Tuple[str, float]]]:
    """Return the best guess category, its score, and two alternatives."""

    if not model:
        top, *rest = DEFAULT_ALTERNATIVES
        return top[0], top[1], DEFAULT_ALTERNATIVES

    text = _compose_text(features)
    vectorizer = model.vectorizer
    classifier = model.classifier
    transformed = vectorizer.transform([text])

    if hasattr(classifier, "predict_proba"):
        proba = classifier.predict_proba(transformed)[0]
    else:
        decision = classifier.decision_function(transformed)
        if np.ndim(decision) == 1:
            decision = np.array([decision])
        proba = _softmax(decision[0])

    classes = getattr(classifier, "classes_", np.array([]))
    if classes.size == 0:
        top, *rest = DEFAULT_ALTERNATIVES
        return top[0], top[1], DEFAULT_ALTERNATIVES

    ranked = sorted(zip(classes, proba), key=lambda item: item[1], reverse=True)
    top_label, top_score = ranked[0]
    alternatives = [(label, float(score)) for label, score in ranked[:3]]
    return str(top_label), float(top_score), alternatives


def partial_train(
    samples: Iterable[Sample],
    existing_model: Optional[ModelBundle] = None,
) -> Tuple[Optional[ModelBundle], Dict[str, Any]]:
    """Perform an incremental training round and return metrics."""

    sample_list = list(samples)
    metrics: Dict[str, Any] = {"n": len(sample_list)}
    if not sample_list:
        metrics["skipped"] = True
        return existing_model, metrics

    texts = []
    labels = []
    for sample in sample_list:
        text_fragments = [sample.get("text") or ""]
        merchant = sample.get("merchant")
        if merchant:
            text_fragments.append(str(merchant))
        amount = sample.get("amount")
        if amount is not None:
            text_fragments.append(str(amount))
        texts.append("\n".join(part for part in text_fragments if part))
        label_value = sample.get("label")
        labels.append(str(label_value) if label_value is not None else "未分類")

    vectorizer = build_or_load_vectorizer(existing_model.vectorizer if existing_model else None)
    classifier = build_classifier(existing_model.classifier if existing_model else None)

    if existing_model:
        transformed = vectorizer.transform(texts)
    else:
        vectorizer.fit(texts)
        transformed = vectorizer.transform(texts)

    existing_class_labels = getattr(classifier, "classes_", None)
    existing_label_set = (
        {str(label) for label in existing_class_labels}
        if existing_class_labels is not None
        else set()
    )
    new_label_set = {str(label) for label in labels}
    merged_labels = sorted(existing_label_set | new_label_set)

    classifier.partial_fit(transformed, labels, classes=np.array(merged_labels))

    metrics["classes"] = merged_labels
    return ModelBundle(vectorizer=vectorizer, classifier=classifier), metrics


__all__ = [
    "ModelBundle",
    "build_or_load_vectorizer",
    "build_classifier",
    "predict_category",
    "partial_train",
]
