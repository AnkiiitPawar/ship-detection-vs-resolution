"""Make a tiny fake dataset in the Airbus Ship Detection layout, for testing the notebook without the real data.

Writes <output>/airbus-ship-detection/train_v2/*.jpg and train_ship_segmentations_v2.csv:
70 tiles of noisy "sea" with bright rotated rectangles as "ships", each stored as an Airbus-style run-length mask.

Usage:
    python tests/make_fake_data.py /tmp/fake_input
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SIZE = 768        # Airbus tiles are 768 x 768 pixels
N_TILES = 70

root = Path(sys.argv[1]) / "airbus-ship-detection"
(root / "train_v2").mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1)


def mask_to_rle(mask):
    """Airbus format: pixels numbered down each column, then across, 1-indexed 'start length' pairs."""
    flat = mask.T.flatten()
    padded = np.concatenate([[0], flat, [0]])
    runs = np.where(padded[1:] != padded[:-1])[0] + 1
    runs[1::2] -= runs[0::2]
    return " ".join(str(x) for x in runs)


rows = []
for i in range(N_TILES):
    img_id = f"{i:09x}.jpg"
    sea = rng.normal(70, 12, (SIZE, SIZE, 3)).clip(0, 255).astype(np.uint8)
    sea[..., 0] = (sea[..., 0] * 1.3).clip(0, 255).astype(np.uint8)   # a little bluer (OpenCV stores BGR)
    n_ships = 0 if i % 4 == 0 else int(rng.integers(1, 6))              # every fourth tile is empty sea
    if n_ships == 0:
        rows.append({"ImageId": img_id, "EncodedPixels": np.nan})
    for _ in range(n_ships):
        length, width = float(rng.uniform(14, 90)), float(rng.uniform(5, 16))
        cx, cy, angle = float(rng.uniform(60, SIZE - 60)), float(rng.uniform(60, SIZE - 60)), float(rng.uniform(0, 180))
        corners = cv2.boxPoints(((cx, cy), (length, width), angle)).astype(np.int32)
        mask = np.zeros((SIZE, SIZE), np.uint8)
        cv2.fillPoly(mask, [corners], 1)
        if mask.sum() == 0:
            continue
        sea[mask == 1] = (225, 225, 225)
        rows.append({"ImageId": img_id, "EncodedPixels": mask_to_rle(mask)})
    cv2.imwrite(str(root / "train_v2" / img_id), sea, [cv2.IMWRITE_JPEG_QUALITY, 92])

pd.DataFrame(rows).to_csv(root / "train_ship_segmentations_v2.csv", index=False)
print(f"Fake data written to {root}: {N_TILES} tiles, {len(rows)} rows")
