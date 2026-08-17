import unittest

from detector import (
    InputValidationError,
    confidence_band,
    interpret_probabilities,
    normalize_model_label,
    validate_text,
)


class InputValidationTests(unittest.TestCase):
    def test_normalizes_whitespace(self):
        self.assertEqual(
            validate_text("A   sufficiently detailed\nnews headline appears here"),
            "A sufficiently detailed news headline appears here",
        )

    def test_rejects_empty_or_short_text(self):
        with self.assertRaises(InputValidationError):
            validate_text("  ")
        with self.assertRaises(InputValidationError):
            validate_text("Too little context")

    def test_rejects_excessively_long_text(self):
        with self.assertRaises(InputValidationError):
            validate_text("word " * 5_000, max_characters=100)


class PredictionInterpretationTests(unittest.TestCase):
    def test_understands_explicit_model_labels(self):
        result = interpret_probabilities(
            [0.08, 0.92], id2label={0: "FAKE", 1: "REAL"}
        )
        self.assertEqual(result.label, "real")
        self.assertAlmostEqual(result.confidence, 0.92)
        self.assertEqual(result.confidence_band, "high")

    def test_supports_default_label_ids(self):
        result = interpret_probabilities([0.78, 0.22])
        self.assertEqual(result.label, "fake")
        self.assertEqual(result.probabilities, {"fake": 0.78, "real": 0.22})

    def test_marks_uncertain_predictions(self):
        result = interpret_probabilities([0.53, 0.47], minimum_confidence=0.65)
        self.assertTrue(result.low_confidence)
        self.assertEqual(result.confidence_band, "low")

    def test_rejects_invalid_probabilities(self):
        with self.assertRaises(ValueError):
            interpret_probabilities([0.8, 0.5])
        with self.assertRaises(ValueError):
            interpret_probabilities([1.0])

    def test_label_normalization(self):
        self.assertEqual(normalize_model_label("LABEL_0", 0), "fake")
        self.assertEqual(normalize_model_label("truthful", 1), "real")

    def test_confidence_bands(self):
        self.assertEqual(confidence_band(0.90), "high")
        self.assertEqual(confidence_band(0.75), "medium")
        self.assertEqual(confidence_band(0.60), "low")


if __name__ == "__main__":
    unittest.main()
