from __future__ import annotations

import io

import numpy as np
from PIL import Image

TEMPLATE = """export default function generate(THREE) {
  const material = new THREE.MeshStandardMaterial({ color: 0x%s, roughness: 0.6, metalness: 0.0 });
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.5, 48, 32), material);
  const root = new THREE.Group();
  root.add(mesh);
  return root;
}
"""


def dominant_color(image: bytes) -> str:
    pixels = Image.open(io.BytesIO(image)).convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
    center = np.asarray(pixels, dtype=np.uint8)[16:48, 16:48].reshape(-1, 3)
    red, green, blue = (int(v) for v in np.median(center, axis=0))
    return f"{red:02x}{green:02x}{blue:02x}"


def fallback_module(image: bytes) -> str:
    return TEMPLATE % dominant_color(image)
