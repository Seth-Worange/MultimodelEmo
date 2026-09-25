import numpy as np

from utils.normalization import apply_input_normalization, fit_input_normalization


def test_normalization_uses_valid_positions_and_keeps_missing_zero():
    data = {
        "audio": np.array([[[1.0, 10.0], [3.0, 14.0], [99.0, 99.0]]], dtype=np.float32),
        "vision": np.array([[[2.0], [6.0], [88.0]]], dtype=np.float32),
        "audio_mask": np.array([[True, True, False]]),
        "vision_mask": np.array([[True, True, False]]),
    }
    stats = fit_input_normalization(data)
    normalized = apply_input_normalization(data, stats)
    np.testing.assert_allclose(normalized["audio"][0, :2].mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(normalized["vision"][0, :2].mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(normalized["audio"][0, 2], 0.0)
    np.testing.assert_allclose(normalized["vision"][0, 2], 0.0)
    np.testing.assert_allclose(data["audio"][0, 2], [99.0, 99.0])
