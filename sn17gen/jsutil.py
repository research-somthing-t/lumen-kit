from __future__ import annotations

import re

_FENCE = re.compile(r"```(?:[a-zA-Z]+)?\s*\n(.*?)```", re.DOTALL)
_EXPORT = re.compile(r"export\s+default\s+function\s*([A-Za-z_$][\w$]*)?\s*\(\s*THREE\s*\)")
_CODE_START = ("function", "const", "let", "var", "export", "//", "/*", "'use", '"use')
FIT = 0.99

WRAP = """
export default function generate(THREE) {
  const inner = __inner(THREE);
  const pivot = new THREE.Group();
  pivot.add(inner);
  pivot.rotation.set(%(rx)s, %(ry)s, 0);
  const root = new THREE.Group();
  root.add(pivot);
  root.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(root);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const s = 0.99 / maxDim;
  root.scale.setScalar(s);
  root.position.set(-center.x * s, -center.y * s, -center.z * s);
  return root;
}
"""


def extract(text: str | None) -> str | None:
    if not text:
        return None
    blocks = [b for b in _FENCE.findall(text) if "export default" in b]
    src = max(blocks, key=len) if blocks else text
    if "export default" not in src:
        return None
    lines = src.strip().splitlines()
    while lines and not lines[0].lstrip().startswith(_CODE_START):
        lines.pop(0)
    return "\n".join(lines).strip() or None


def wrap(js: str, ry: str = "0", rx: str = "0") -> str | None:
    m = _EXPORT.search(js)
    if not m or "__inner" in js:
        return None
    inner = js[:m.start()] + "function __inner(THREE)" + js[m.end():]
    return inner.rstrip() + "\n" + WRAP % {"rx": rx, "ry": ry}


FLAT = ("  pivot.updateMatrixWorld(true);\n"
        "  const box0 = new THREE.Box3().setFromObject(pivot);\n"
        "  const size0 = new THREE.Vector3();\n"
        "  box0.getSize(size0);\n"
        "  if (size0.y < 0.5 * Math.max(size0.x, size0.z)) pivot.rotation.set(0.261799, 0, 0);\n")


WORLD = """
function __world(THREE, root) {
  root.updateMatrixWorld(true);
  const shapes = [];
  let count = 0;
  let preserve = false;
  root.traverse(function (o) {
    if (!o.visible || !o.matrixAutoUpdate || !o.matrixWorldAutoUpdate ||
        (o.isInstancedMesh && o.matrixWorld.determinant() < 0)) preserve = true;
    if ((o.isMesh || o.isLine || o.isPoints) && o.geometry && o.geometry.attributes.position) {
      shapes.push(o);
      count += o.geometry.attributes.position.count;
    }
  });
  if (preserve || count > 200000) return;
  const spots = [];
  const back = new THREE.Matrix4();
  const m = new THREE.Matrix4();
  for (const o of shapes) {
    const spot = new THREE.Vector3().setFromMatrixPosition(o.matrixWorld);
    spots.push(spot);
    back.makeTranslation(-spot.x, -spot.y, -spot.z);
    if (o.isInstancedMesh) {
      for (let i = 0; i < o.count; i++) {
        o.getMatrixAt(i, m);
        o.setMatrixAt(i, m.premultiply(o.matrixWorld).premultiply(back));
      }
      o.instanceMatrix.needsUpdate = true;
      o.boundingBox = null;
      o.boundingSphere = null;
      continue;
    }
    const g = o.geometry.clone();
    g.applyMatrix4(o.matrixWorld);
    g.applyMatrix4(back);
    if (o.isMesh && o.matrixWorld.determinant() < 0) {
      const n = g.index ? g.index.count : g.attributes.position.count;
      const order = new Uint32Array(n - (n % 3));
      for (let i = 0; i < order.length; i++) {
        const c = i - (i % 3) + [0, 2, 1][i % 3];
        order[i] = g.index ? g.index.getX(c) : c;
      }
      g.setIndex(new THREE.BufferAttribute(order, 1));
    }
    o.geometry = g;
  }
  shapes.forEach(function (o, i) {
    o.removeFromParent();
    root.add(o);
    o.position.copy(spots[i]);
    o.quaternion.identity();
    o.scale.set(1, 1, 1);
  });
  root.updateMatrixWorld(true);
}
"""
WRAP_EXPORT = "export default function generate(THREE) {\n"
WRAP_BOX = "  root.updateMatrixWorld(true);\n  const box = new THREE.Box3().setFromObject(root);\n"
WORLD_BOX = "  __world(THREE, root);\n  const box = new THREE.Box3().setFromObject(root);\n"
WRAP_SCALE = "  const s = 0.99 / maxDim;\n"


def plain(js: str) -> str | None:
    out = wrap(js)
    return None if out is None else out.replace("  pivot.rotation.set(0, 0, 0);\n", FLAT, 1)


def final(js: str) -> str | None:
    out = plain(js)
    if out is None:
        return None
    cut = out.rindex(WRAP_EXPORT)
    tail = out[cut:].replace(WRAP_BOX, WORLD_BOX, 1).replace(WRAP_SCALE, f"  const s = {FIT} / maxDim;\n", 1)
    return out[:cut] + WORLD + "\n" + tail
