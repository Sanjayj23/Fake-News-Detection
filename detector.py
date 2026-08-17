"""Reusable inference utilities for the fake-news classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


DEFAULT_MODEL_NAME = "shi13u/fake_news_detection_bert"
DEFAULT_MAX_TOKENS = 512
DEFAULT_MIN_WORDS = 5
DEFAULT_MAX_CHARACTERS = 20_000


class InputValidationError(ValueError):
    """Raised when submitted text is unsuitable for model inference."""


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    confidence_band: str
    low_confidence: bool
    token_count: int
    truncated: bool


def validate_text(
    text: str,
    min_words: int = DEFAULT_MIN_WORDS,
    max_characters: int = DEFAULT_MAX_CHARACTERS,
) -> str:
    """Normalize user input and enforce useful inference limits."""
    normalized = " ".join((text or "").split())
    if not normalized:
        raise InputValidationError("Enter a headline or article before running detection.")
    if len(normalized.split()) < min_words:
        raise InputValidationError(
            f"Provide at least {min_words} words so the model has enough context."
        )
    if len(normalized) > max_characters:
        raise InputValidationError(
            f"Text exceeds the {max_characters:,}-character safety limit."
        )
    return normalized


def normalize_model_label(raw_label: str | None, label_id: int) -> str:
    """Map common Hugging Face label conventions to fake/real labels."""
    normalized = str(raw_label or "").strip().upper()
    if "FAKE" in normalized:
        return "fake"
    if "REAL" in normalized or "TRUE" in normalized:
        return "real"
    # The fine-tuning notebook explicitly defines 0 = fake and 1 = real.
    return "real" if label_id == 1 else "fake" if label_id == 0 else f"class_{label_id}"


def confidence_band(confidence: float) -> str:
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.70:
        return "medium"
    return "low"


def interpret_probabilities(
    probabilities: Sequence[float],
    id2label: Mapping[int | str, str] | None = None,
    minimum_confidence: float = 0.65,
) -> Prediction:
    """Convert model probabilities into a stable, auditable prediction."""
    if len(probabilities) < 2:
        raise ValueError("Binary classification requires at least two probabilities")
    values = [float(value) for value in probabilities]
    if any(value < 0 or value > 1 for value in values):
        raise ValueError("Probabilities must be between zero and one")
    total = sum(values)
    if abs(total - 1.0) > 1e-3:
        raise ValueError("Probabilities must sum to one")

    label_map = id2label or {}
    resolved_labels = [
        normalize_model_label(
            label_map.get(index, label_map.get(str(index), f"LABEL_{index}")), index
        )
        for index in range(len(values))
    ]
    probability_map = {
        resolved_labels[index]: round(value, 6) for index, value in enumerate(values)
    }
    predicted_id = max(range(len(values)), key=values.__getitem__)
    confidence = values[predicted_id]
    return Prediction(
        label=resolved_labels[predicted_id],
        confidence=confidence,
        probabilities=probability_map,
        confidence_band=confidence_band(confidence),
        low_confidence=confidence < minimum_confidence,
        token_count=0,
        truncated=False,
    )


class FakeNewsDetector:
    """Lazy, testable wrapper around a Hugging Face sequence classifier."""

    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        minimum_confidence: float = 0.65,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.max_tokens = max_tokens
        self.minimum_confidence = minimum_confidence

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = DEFAULT_MODEL_NAME,
        token: str | None = None,
        **kwargs: Any,
    ) -> "FakeNewsDetector":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, token=token)
        model.eval()
        return cls(tokenizer=tokenizer, model=model, **kwargs)

    def predict(self, text: str) -> Prediction:
        import torch

        normalized = validate_text(text)
        full_token_count = len(
            self.tokenizer.encode(normalized, add_special_tokens=True, truncation=False)
        )
        encoded = self.tokenizer(
            normalized,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_tokens,
        )
        with torch.inference_mode():
            logits = self.model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()

        id2label = getattr(getattr(self.model, "config", None), "id2label", {})
        interpreted = interpret_probabilities(
            probabilities,
            id2label=id2label,
            minimum_confidence=self.minimum_confidence,
        )
        return Prediction(
            label=interpreted.label,
            confidence=interpreted.confidence,
            probabilities=interpreted.probabilities,
            confidence_band=interpreted.confidence_band,
            low_confidence=interpreted.low_confidence,
            token_count=full_token_count,
            truncated=full_token_count > self.max_tokens,
        )
