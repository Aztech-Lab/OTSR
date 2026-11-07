"""Convert an image to the raw float layout the Node tools read (tools/bench.mjs): a 16-byte header
(magic b"OTSI", u32 W, u32 H, u32 0) followed by W x H x 3 float32 RGB in [0, 1], little-endian.

    python reference/dump_image.py input.png output.rgb.bin
"""
from __future__ import annotations

import struct
import sys

import numpy as np
from PIL import Image


def main(src: str, dst: str) -> None:
    img = np.asarray(Image.open(src).convert("RGB"), dtype=np.float32) / 255.0
    H, W, _ = img.shape
    with open(dst, "wb") as f:
        f.write(struct.pack("<4sIII", b"OTSI", W, H, 0))
        f.write(np.ascontiguousarray(img, dtype="<f4").tobytes())
    print(f"{src}: {W}x{H} -> {dst} ({W * H * 12 / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
