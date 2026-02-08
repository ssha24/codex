# Multitemporal Satellite Change Detection

This project provides a minimal Python workflow for detecting change between two multitemporal satellite images. The approach is intentionally simple and transparent:

1. Read two co-registered images (GeoTIFF, TIFF, or PNG).
2. Normalize them into a comparable range.
3. Compute an absolute difference change score.
4. Threshold the score (Otsu by default) to obtain a binary change map.

## Project layout

```
.
├── src/
│   ├── change_detection.py   # core change detection utilities
│   └── cli.py                # command-line interface
├── tests/
│   └── test_change_detection.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run on two images

```bash
python -m src.cli \
  --image-a data/before.tif \
  --image-b data/after.tif \
  --output change_map.png
```

The output is a single-band PNG where white (255) means change detected and black (0) means no change.

## Run with a deep learning model

Provide a TorchScript model that accepts a tensor shaped `(1, 2, H, W)` (stacked normalized images) and returns
either a single-channel change probability map or a two-channel softmax output.

```bash
python -m src.cli \
  --image-a data/before.tif \
  --image-b data/after.tif \
  --output change_map.png \
  --model-path models/change_detector.pt \
  --model-device cpu \
  --threshold 0.5
```

Install PyTorch separately (for example, `pip install torch`) to enable the deep learning path.

## Notes

- Images must be co-registered (same projection/extent) for meaningful results.
- You can switch thresholding to a fixed numeric threshold using `--threshold`.
- For multiband imagery, the script uses the first band by default. You can specify `--band` to select another band.

## Next steps

- Add cloud masking or quality filtering.
- Swap in a more advanced change detector (e.g., CVA, MAD, or deep learning).
- Export results as GeoTIFF to preserve georeferencing.
