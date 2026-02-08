import numpy as np

from src.change_detection import compute_change_score, detect_change, normalize, threshold_change


def test_normalize_constant_array():
    data = np.ones((2, 2), dtype=np.float32)
    normalized = normalize(data)
    assert np.allclose(normalized, 0.0)


def test_change_score_shape_mismatch():
    image_a = np.zeros((2, 2), dtype=np.float32)
    image_b = np.zeros((3, 3), dtype=np.float32)
    try:
        compute_change_score(image_a, image_b)
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched shapes")


def test_detect_change_binary_map():
    image_a = np.array([[0, 0], [0, 1]], dtype=np.float32)
    image_b = np.array([[0, 1], [0, 1]], dtype=np.float32)
    result = detect_change(image_a, image_b, threshold=0.25)
    assert result.change_map.dtype == np.uint8
    assert set(np.unique(result.change_map)).issubset({0, 1})


def test_threshold_change_fixed_value():
    score = np.array([[0.1, 0.5], [0.2, 0.9]], dtype=np.float32)
    change_map, used_threshold = threshold_change(score, threshold=0.4)
    assert used_threshold == 0.4
    assert np.array_equal(change_map, np.array([[0, 1], [0, 1]], dtype=np.uint8))
