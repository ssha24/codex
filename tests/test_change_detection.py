import numpy as np

from src.change_detection import (
    compute_change_score,
    detect_change,
    detect_change_deeplearning,
    normalize,
    threshold_change,
    torch,
)


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


def test_detect_change_deeplearning_optional(tmp_path):
    if torch is None:
        image_a = np.zeros((2, 2), dtype=np.float32)
        image_b = np.zeros((2, 2), dtype=np.float32)
        try:
            detect_change_deeplearning(image_a, image_b, tmp_path / "model.pt")
        except RuntimeError as exc:
            assert "PyTorch" in str(exc)
        else:
            raise AssertionError("Expected RuntimeError when PyTorch is unavailable.")
        return

    class DummyModel(torch.nn.Module):
        def forward(self, tensor):
            return tensor[:, :1, :, :]

    model = torch.jit.script(DummyModel())
    model_path = tmp_path / "dummy.pt"
    model.save(str(model_path))

    image_a = np.array([[0, 1], [0, 1]], dtype=np.float32)
    image_b = np.array([[0, 0], [0, 0]], dtype=np.float32)
    result = detect_change_deeplearning(image_a, image_b, model_path, threshold=0.5)
    assert np.array_equal(result.change_map, np.array([[0, 1], [0, 1]], dtype=np.uint8))
