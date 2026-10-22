from __future__ import annotations

import difflib
import hashlib
import io
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, Unpack, cast

import numpy as np
from PIL import Image
from openai import APIConnectionError, APIError, APITimeoutError
from openai.types.responses import Response, ResponseInputParam
from openai.types.responses.response_create_params import ResponseCreateParamsBase
from witness_client import ExchangePending, RenderOptions, RenderRejected, RenderUnavailable, WitnessError

from . import edits, metrics, powrepair, prompts, route
from .budget import CREDIT_LIMIT, SAFETY, Budget, call_credits, worst_case
from .fallback import fallback_module
from .images import data_url, judge_shots, object_closeup, png_data_url, to_jpeg, view_sheet, zoomed_views
from .layout import Renderer

SOL = "gpt-6.2-sol"
SOL_FALLBACK = "gpt-6.1-sol"
FALLBACK_OFFSET = 100000
ASTRA = "gpt-6-astra"
GENERATE_EFFORT = "high"
SELECT_EFFORT = "medium"
GENERATE_MAX_OUTPUT = 32768
SELECT_MAX_OUTPUT = 8192
PENALTY_WEIGHT = 2.0
ASTRA_MAX_OUTPUT = 22000
EFFORT_MAX_OUTPUT = {"xhigh": 44000, "max": 64000}
GRID_WEIGHT = 0.25
PARENT_MARGIN = 0.25
JOIN_ATTEMPTS = 6
RENDER_WAITS = (5, 10, 20, 40, 60, 60)
RENDER_TIMEOUT_RETRIES = 2
RENDER_BUSY_LIMIT = 900
DRAFT_ID = 1
REDRAFT_ID = 50
REWRITE_BASE = 100
REPAIR_OFFSET = 5000
SELECT_ID = 9000
FINAL_SELECT_ID = 9800
CRITIQUE_ID = 9400
MERGE_ID = 9410
CRITIQUE_EFFORT = "medium"
JUDGE_MAX_OUTPUT = 4096
JUDGE_DRAW = 0.7
MODULE_GROWTH = 0.15
MODULE_CHARS_PER_TOKEN = 2.85
PATCH_FLOOR = 0.4
PATCH_PRIOR = 0.6
SELECT_CALL_CREDITS = 15731
FORECAST_IMAGES = 7
FORECAST_EXTRA_CHARS = 400
ROUND_COST = {(ASTRA, "high"): (494, 10313), (SOL, "xhigh"): (10712, 9125), (SOL, "high"): (748, 9575)}
BRIEF_ID = 9990
BRIEF_MAX_OUTPUT = 4096
MEASURED_ASTRA_MAX_OUTPUT = 18000
MATCH_WEIGHT = 0.5
FRAME_LIMIT = 12.0

FRONT = RenderOptions(thetas=(30, 330, 0, 0), phis=(0, 0, 15, -30), img_size=384, bg_color="ffffff",
                      lighting="neutral")
SIDES = RenderOptions(thetas=(90, 180, 270, 0), phis=(0, 0, 0, -90), img_size=256, bg_color="ffffff",
                      lighting="neutral")
FRONT_LARGE = RenderOptions(thetas=(30, 330, 0, 0), phis=(0, 0, 15, -30), img_size=1024, gap=0, bg_color="ffffff",
                            lighting="neutral")
GRID_VIEW = RenderOptions(lighting="neutral")

WRAPPER = """
function __drawCalls404(root) {
  let count = 0;
  root.traverse(function (node) { if (node.isMesh || node.isLine || node.isPoints) count += 1; });
  return count;
}

function __mergeByMaterial404(THREE, root) {
  root.updateMatrixWorld(true);
  const byMaterial = new Map();
  const groups = [];
  root.traverse(function (node) {
    if (!node.isMesh || node.isInstancedMesh || !node.visible || node.children.length || Array.isArray(node.material)) return;
    const geometry = node.geometry;
    if (!geometry || !geometry.attributes.position) return;
    if (geometry.morphAttributes && Object.keys(geometry.morphAttributes).length) return;
    const names = Object.keys(geometry.attributes).sort();
    const key = names.map(function (n) { return n + ':' + geometry.attributes[n].itemSize; }).join(',');
    if (!byMaterial.has(node.material)) byMaterial.set(node.material, new Map());
    const byLayout = byMaterial.get(node.material);
    if (!byLayout.has(key)) { const group = { names: names, meshes: [] }; byLayout.set(key, group); groups.push(group); }
    byLayout.get(key).meshes.push(node);
  });
  const vector = new THREE.Vector3();
  const normalMatrix = new THREE.Matrix3();
  const merged = [];
  for (const group of groups) {
    const meshes = group.meshes;
    if (meshes.length < 2) continue;
    let vertexCount = 0;
    let indexCount = 0;
    for (const mesh of meshes) {
      const geometry = mesh.geometry;
      vertexCount += geometry.attributes.position.count;
      indexCount += geometry.index ? geometry.index.count : geometry.attributes.position.count;
    }
    const arrays = {};
    for (const name of group.names) arrays[name] = new Float32Array(vertexCount * meshes[0].geometry.attributes[name].itemSize);
    const index = vertexCount > 65535 ? new Uint32Array(indexCount) : new Uint16Array(indexCount);
    let vertexOffset = 0;
    let indexOffset = 0;
    for (const mesh of meshes) {
      const geometry = mesh.geometry;
      const matrix = mesh.matrixWorld;
      normalMatrix.getNormalMatrix(matrix);
      const count = geometry.attributes.position.count;
      for (const name of group.names) {
        const attribute = geometry.attributes[name];
        const size = attribute.itemSize;
        const out = arrays[name];
        for (let i = 0; i < count; i++) {
          const base = (vertexOffset + i) * size;
          if (name === 'position' || name === 'normal') {
            vector.fromBufferAttribute(attribute, i);
            if (name === 'position') vector.applyMatrix4(matrix); else vector.applyNormalMatrix(normalMatrix);
            out[base] = vector.x; out[base + 1] = vector.y; out[base + 2] = vector.z;
          } else {
            out[base] = attribute.getX(i);
            if (size > 1) out[base + 1] = attribute.getY(i);
            if (size > 2) out[base + 2] = attribute.getZ(i);
            if (size > 3) out[base + 3] = attribute.getW(i);
          }
        }
      }
      const flip = matrix.determinant() < 0;
      const triangles = geometry.index ? geometry.index.count : count;
      for (let i = 0; i < triangles; i++) {
        const source = flip ? i - (i % 3) + [0, 2, 1][i % 3] : i;
        index[indexOffset + i] = (geometry.index ? geometry.index.getX(source) : source) + vertexOffset;
      }
      vertexOffset += count;
      indexOffset += triangles;
    }
    const combined = new THREE.BufferGeometry();
    for (const name of group.names) {
      combined.setAttribute(name, new THREE.BufferAttribute(arrays[name], meshes[0].geometry.attributes[name].itemSize));
    }
    combined.setIndex(new THREE.BufferAttribute(index, 1));
    for (const mesh of meshes) mesh.parent.remove(mesh);
    merged.push(new THREE.Mesh(combined, meshes[0].material));
  }
  for (const mesh of merged) root.add(mesh);
}

function __textures404(root) {
  const found = new Map();
  root.traverse(function (node) {
    const materials = node.material ? [].concat(node.material) : [];
    for (const material of materials) {
      for (const key of Object.keys(material)) {
        const value = material[key];
        if (value && value.isDataTexture && value.image && value.image.data) {
          if (!found.has(value)) found.set(value, []);
          found.get(value).push([material, key]);
        }
      }
    }
  });
  return found;
}

function __shrinkTextures404(THREE, root, limit) {
  for (let pass = 0; pass < 8; pass++) {
    const found = __textures404(root);
    let total = 0;
    let largest = null;
    for (const texture of found.keys()) {
      total += texture.image.data.byteLength;
      if (!largest || texture.image.data.byteLength > largest.image.data.byteLength) largest = texture;
    }
    if (total <= limit || !largest || largest.image.width < 16 || largest.image.height < 16) return;
    const width = largest.image.width;
    const height = largest.image.height;
    const data = largest.image.data;
    const channels = Math.round(data.length / (width * height));
    const halfWidth = Math.floor(width / 2);
    const halfHeight = Math.floor(height / 2);
    const size = halfWidth * halfHeight * channels;
    const out = data instanceof Float32Array ? new Float32Array(size)
      : data instanceof Uint16Array ? new Uint16Array(size) : new Uint8Array(size);
    for (let y = 0; y < halfHeight; y++) {
      for (let x = 0; x < halfWidth; x++) {
        for (let c = 0; c < channels; c++) {
          const a = ((2 * y) * width + 2 * x) * channels + c;
          const b = a + channels;
          const d = a + width * channels;
          out[(y * halfWidth + x) * channels + c] = (data[a] + data[b] + data[d] + data[d + channels]) / 4;
        }
      }
    }
    const smaller = new THREE.DataTexture(out, halfWidth, halfHeight, largest.format, largest.type);
    smaller.wrapS = largest.wrapS;
    smaller.wrapT = largest.wrapT;
    smaller.magFilter = largest.magFilter;
    smaller.minFilter = largest.minFilter;
    smaller.generateMipmaps = largest.generateMipmaps;
    smaller.colorSpace = largest.colorSpace;
    smaller.flipY = largest.flipY;
    smaller.repeat.copy(largest.repeat);
    smaller.offset.copy(largest.offset);
    smaller.needsUpdate = true;
    for (const [material, key] of found.get(largest)) material[key] = smaller;
  }
}

export default function generate(THREE) {
  const root = new THREE.Group();
  root.add(__asset404(THREE));
  if (__drawCalls404(root) > 200) __mergeByMaterial404(THREE, root);
  __shrinkTextures404(THREE, root, 4100000);
  const box = new THREE.Box3().setFromObject(root);
  const size = box.getSize(new THREE.Vector3());
  const extent = Math.max(size.x, size.y, size.z);
  if (extent > 0 && Number.isFinite(extent)) {
    const center = box.getCenter(new THREE.Vector3());
    const scale = 0.95 / extent;
    root.scale.setScalar(scale);
    root.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
  }
  return root;
}
"""
_FIT = "  const box = new THREE.Box3().setFromObject(root);\n"


_GLOW_FUNCTION = """
function __lift404(root, amount) {
  const seen = new Set();
  root.traverse(function (node) {
    for (const material of (node.material ? [].concat(node.material) : [])) {
      if (seen.has(material) || !material.isMeshStandardMaterial || material.vertexColors) continue;
      seen.add(material);
      material.emissive.multiplyScalar(material.emissiveIntensity).add(material.color.clone().multiplyScalar(amount));
      material.emissiveIntensity = 1;
      if (material.map && !material.emissiveMap) material.emissiveMap = material.map;
      material.color.multiplyScalar(1 - 1.6 * amount);
    }
  });
}
"""
_EXPORT_LINE = "export default function generate(THREE) {\n"
_OUTLINE_FUNCTION = """
function __bake404(THREE, root) {
  root.updateMatrixWorld(true);
  let vertices = 0;
  root.traverse(function (node) {
    if (node.geometry && node.geometry.attributes.position && !node.isInstancedMesh) vertices += node.geometry.attributes.position.count;
  });
  if (vertices > 200000) return;
  const instance = new THREE.Matrix4();
  const offset = new THREE.Matrix4();
  const places = new Map();
  // A mirroring matrix moved into the vertices or instances no longer tells three.js to flip the winding.
  const reversed = function (geometry) {
    const count = geometry.index ? geometry.index.count : geometry.attributes.position.count;
    const order = new Uint32Array(count - (count % 3));
    for (let i = 0; i < order.length; i++) {
      const corner = i - (i % 3) + [0, 2, 1][i % 3];
      order[i] = geometry.index ? geometry.index.getX(corner) : corner;
    }
    geometry.setIndex(new THREE.BufferAttribute(order, 1));
    return geometry;
  };
  root.traverse(function (node) {
    const isShape = node.isMesh || node.isLine || node.isPoints;
    if (!isShape || !node.geometry || !node.geometry.attributes.position) return;
    // Each part keeps its own position: three.js sorts transparent parts by it, so glass still draws in the same order.
    const place = new THREE.Vector3().setFromMatrixPosition(node.matrixWorld);
    places.set(node, place);
    offset.makeTranslation(-place.x, -place.y, -place.z);
    if (node.isInstancedMesh) {
      for (let i = 0; i < node.count; i++) {
        node.getMatrixAt(i, instance);
        node.setMatrixAt(i, instance.premultiply(node.matrixWorld).premultiply(offset));
      }
      node.instanceMatrix.needsUpdate = true;
      node.boundingBox = null;
      node.boundingSphere = null;
      if (node.matrixWorld.determinant() < 0) node.geometry = reversed(node.geometry.clone());
      return;
    }
    let geometry = node.geometry.clone();
    geometry.applyMatrix4(node.matrixWorld);
    geometry.applyMatrix4(offset);
    if (node.isMesh && node.matrixWorld.determinant() < 0) geometry = reversed(geometry);
    node.geometry = geometry;
  });
  root.traverse(function (node) {
    const place = places.get(node);
    // A part inside another part keeps its world position: its parent's own position is taken off.
    let up = new THREE.Vector3();
    for (let p = node.parent; p && p !== root.parent; p = p.parent) {
      const q = places.get(p);
      if (q) { up = q; break; }
    }
    if (place) node.position.copy(place).sub(up); else node.position.set(0, 0, 0);
    node.quaternion.identity();
    node.scale.set(1, 1, 1);
  });
  root.updateMatrixWorld(true);
}
"""
_OUTLINE_STEPS = """  __bake404(THREE, root);
"""


def chain_wrapper(fill: float, glow: float = 0.0, outline_fit: bool = False) -> str:
    wrapper = WRAPPER.replace("0.95 / extent", f"{fill} / extent", 1)
    if outline_fit:
        wrapper = wrapper.replace(_FIT, _OUTLINE_STEPS + _FIT, 1).replace(_EXPORT_LINE, _OUTLINE_FUNCTION + "\n" + _EXPORT_LINE, 1)
    if glow:
        wrapper = wrapper.replace(_FIT, f"  __lift404(root, {glow});\n" + _FIT, 1)
        wrapper = wrapper.replace(_EXPORT_LINE, _GLOW_FUNCTION + "\n" + _EXPORT_LINE, 1)
    return wrapper


_EXPORT = re.compile(r"export\s+default\s+function\s*(?:[A-Za-z_$][\w$]*\s*)?\(")


class ModelCalls(Protocol):
    def create(self, *, request_id: int = 0, **params: Unpack[ResponseCreateParamsBase]) -> Response: ...


class PatientRenderer:

    def __init__(self, renderer: Renderer) -> None:
        self._renderer = renderer

    def render(self, source: str, options: RenderOptions) -> bytes:
        waits = iter(RENDER_WAITS)
        timeouts = 0
        busy = 2
        busy_waited = 0
        while True:
            try:
                return self._renderer.render(source, options)
            except RenderRejected as rejection:
                timeouts += 1
                if "TIMEOUT_EXCEEDED" not in rejection.detail_json or timeouts > RENDER_TIMEOUT_RETRIES:
                    raise
            except RenderUnavailable as error:
                response = getattr(error.__cause__, "response", None)
                if getattr(response, "status_code", None) == 503 and busy_waited < RENDER_BUSY_LIMIT:
                    time.sleep(busy)
                    busy_waited += busy
                    busy = min(busy * 2, 10)
                    continue
                wait = next(waits, None)
                if wait is None:
                    raise
                time.sleep(wait)


OVERLOAD_WAITS = (5, 15, 30, 60)
OVERLOADED = ("Unusable provider response", "currently overloaded", "server_is_overloaded", "HTTP 429")


class CallFailed(Exception):
    pass


@dataclass
class Candidate:
    name: str
    source: str
    front: bytes
    sides: bytes
    raw: str = ""
    views: list[bytes] = field(default_factory=list)
    parent: Candidate | None = None
    metrics: float = 0.0
    measurements: str = ""
    overlay: bytes = b""
    shots: list[bytes] = field(default_factory=list)


@dataclass
class Judgement:
    score: float
    fixes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Plan:

    drafts: tuple[tuple[str, str, str], ...]
    draft_max_output: int = 16384
    group_size: int = 4
    refine_rounds: int = 1
    refine_focuses: tuple[str, ...] = ("blind", "shape")
    refine_effort: str = "medium"


@dataclass(frozen=True)
class Draft:
    start: str
    model: str
    effort: str


@dataclass(frozen=True)
class Step:

    rewrites: tuple[tuple[str, str], ...] = ()
    lineage: tuple[tuple[str, str], ...] = ()
    keep: int = 0
    focus: tuple[str, ...] = ()
    each_lineage: bool = False
    fallback: tuple[str, str] | None = None
    judge_pick: bool = False
    nudge: str = ""
    patch: bool = False
    feed_diffs: bool = False


@dataclass(frozen=True)
class StagePlan:
    drafts: tuple[Draft, ...]
    steps: tuple[Step, ...]
    draft_keep: int = 0
    draft_each_start: bool = False
    fill: float = 0.99
    glow: float = 0.0
    measured: bool = False
    outline_fit: bool = False
    outline_fit_last: bool = False
    replan: bool = False
    merge_pick: bool = False


@dataclass
class Call:
    request_id: int
    model: str
    instructions: str
    content: list[dict[str, Any]]
    effort: str
    max_output_tokens: int
    text: dict[str, Any] | None = None
    estimate: float = 0.0
    refused: bool = False


def extract_source(text: str) -> str | None:
    text = text.strip()
    blocks = [b for b in re.findall(r"```(?:javascript|js|mjs)?[^\n]*\n([\s\S]*?)```", text) if _EXPORT.search(b)]
    if blocks:
        text = blocks[-1].strip()
    return text if _EXPORT.search(text) else None


def call_estimate(call: Call) -> float:
    text_chars = len(call.instructions) + sum(len(c.get("text", "")) for c in call.content)
    images = sum(1 for c in call.content if c["type"] == "input_image")
    return worst_case(call.model, text_chars, images, call.max_output_tokens)


def code_changes(before: str, after: str) -> list[str]:
    diff = list(difflib.unified_diff(before.splitlines(), after.splitlines(), n=0, lineterm=""))[2:]
    changed = [f"{line[0]} {line[1:].strip()[:110]}" for line in diff if line[:1] in ("+", "-") and line[1:].strip()]
    if len(changed) > 10:
        return changed[:10] + [prompts.CHANGE_MORE.format(count=len(changed) - 10)]
    return changed


def stage1_penalties(reply: Response | None) -> tuple[float, float] | None:
    if reply is None:
        return None
    try:
        verdict = json.loads(reply.output_text)
        values = (verdict["penalty_1"], verdict["penalty_2"])
    except (ValueError, KeyError, TypeError):
        return None
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        return None
    return float(values[0]), float(values[1])


def _written(step: Step) -> tuple[str, str] | None:
    if step.lineage:
        return step.lineage[0]
    return step.rewrites[0] if step.rewrites else None


def _with_rewrite(step: Step, pair: tuple[str, str]) -> Step:
    if step.lineage:
        return replace(step, lineage=(pair,) * len(step.lineage))
    return replace(step, rewrites=(pair,) * len(step.rewrites))


def normalize(source: str, wrapper: str = WRAPPER) -> str:
    source = powrepair.repaired(source)
    metrics = _EXPORT.search(source)
    if metrics is None:
        return source
    inner = source[: metrics.start()] + "function __asset404(" + source[metrics.end():]
    return inner + "\n" + wrapper


def frames_close(shown: list[bytes], chosen: list[bytes], limit: float = 8.0) -> bool:
    if len(shown) != len(chosen):
        return False
    for one, two in zip(shown, chosen, strict=True):
        a = np.asarray(Image.open(io.BytesIO(one)).convert("RGB").resize((96, 96), Image.Resampling.BILINEAR), dtype=np.int16)
        b = np.asarray(Image.open(io.BytesIO(two)).convert("RGB").resize((96, 96), Image.Resampling.BILINEAR), dtype=np.int16)
        if float(np.abs(a - b).mean()) > limit:
            return False
    return True


class Pipeline:
    def __init__(self, image: bytes, seed: int, calls: ModelCalls, renderer: Renderer) -> None:
        self.image = image
        self.P = route.prompt_set(image)
        self.seed = seed
        self.calls = calls
        self.renderer = PatientRenderer(renderer)
        self.budget = Budget()
        self.target = {"type": "input_image", "image_url": png_data_url(image), "detail": "high"}
        self.digest = hashlib.sha256(image).hexdigest()
        self.best_so_far: str | None = None
        self.wrapper = WRAPPER
        self.closeup: list[dict[str, Any]] = []
        self.instructions = self.P.GENERATE_CHAIN
        self.astra_cap = ASTRA_MAX_OUTPUT
        self.photo: metrics.Appearance | None = None
        self.brief = ""
        self.sol_refused = False
        self.keep_shots = False
        self.changes: list[tuple[int, list[str], bool]] = []
        self.patch_shares: list[float] = []
        self.patch_log: list[dict[str, Any]] = []


    def run_stage(self, pool: ThreadPoolExecutor, calls: list[Call]) -> list[Response | None]:
        admitted: list[Call | None] = []
        for call in calls:
            call.estimate = call_estimate(call)
            admitted.append(call if self.budget.reserve(call.estimate) else None)
        skip_sol = self.sol_refused
        futures = [pool.submit(self._dispatch, call, skip_sol) if call is not None else None for call in admitted]
        replies = [future.result() if future is not None else None for future in futures]
        if any(call is not None and call.refused for call in admitted):
            self.sol_refused = True
        return replies

    def fits(self, call: Call) -> bool:
        return self.budget.allows([call_estimate(call)])

    def _dispatch(self, call: Call, skip_sol: bool = False) -> Response | None:
        usage = None
        params: dict[str, Any] = {
            "model": call.model,
            "store": False,
            "service_tier": "default",
            "reasoning": {"effort": call.effort},
            "max_output_tokens": call.max_output_tokens,
            "instructions": call.instructions,
            "input": cast(ResponseInputParam, [{"role": "user", "content": call.content}]),
        }
        if call.text is not None:
            params["text"] = call.text
        model = call.model
        try:
            fall_back = call.model == SOL and skip_sol
            if not fall_back:
                try:
                    result = self._create(call.request_id, params)
                except APITimeoutError:
                    raise
                except APIConnectionError:
                    if call.model != SOL:
                        raise
                    call.refused = fall_back = True
            if fall_back:
                model = SOL_FALLBACK
                result = self._create(call.request_id + FALLBACK_OFFSET, {**params, "model": model})
            usage = result.usage
            if usage is None:
                raise CallFailed(f"call {call.request_id}: reply without usage")
            return result if result.status == "completed" else None
        except APIConnectionError:
            raise
        except APIError as error:
            raise CallFailed(f"call {call.request_id}: {error}") from error
        finally:
            self.budget.settle(call.estimate, model, usage)

    def _create(self, request_id: int, params: dict[str, Any]) -> Response:
        waits = iter(OVERLOAD_WAITS)
        for attempt in range(JOIN_ATTEMPTS + len(OVERLOAD_WAITS)):
            try:
                return self.calls.create(request_id=request_id, **params)
            except APITimeoutError:
                if attempt == JOIN_ATTEMPTS - 1:
                    raise
            except APIError as error:
                wait = next(waits, None)
                if wait is None or not any(sign in str(error) for sign in OVERLOADED):
                    raise
                time.sleep(wait)
        raise AssertionError("unreachable")

    def generation_call(self, request_id: int, model: str, extra: list[dict[str, Any]], effort: str = GENERATE_EFFORT,
                        instructions: str = "base", max_output: int = GENERATE_MAX_OUTPUT) -> Call:
        content = [{"type": "input_text", "text": self.P.TARGET}, self.target, *extra]
        text = self.P.GENERATE_V2 if instructions == "handbook" else self.P.GENERATE
        return Call(request_id, model, text, content, effort, max_output)

    def module_call(self, request_id: int, model: str, effort: str, extra: list[dict[str, Any]],
                    start: str = "plain") -> Call:
        target = self.P.TARGET_PROPORTIONS if start == "proportions" else self.P.TARGET
        if self.brief and not extra:
            target = f"{target}\n\n{self.brief}"
        content = [{"type": "input_text", "text": target}, self.target, *self.closeup, *extra]
        cap = self.astra_cap if model == ASTRA else EFFORT_MAX_OUTPUT.get(effort, GENERATE_MAX_OUTPUT)
        return Call(request_id, model, self.instructions, content, effort, cap)

    def rewrite_call(self, request_id: int, model: str, effort: str, parent: Candidate, focus: str = "",
                     step: Step | None = None) -> Call:
        views = [{"type": "input_image", "image_url": data_url(view), "detail": "high"} for view in parent.views]
        if self.photo is not None or self.brief:
            overlay = [{"type": "input_image", "image_url": data_url(parent.overlay), "detail": "low"}] if parent.overlay else []
            text = self.P.REWRITE_MEASURED.format(source=parent.raw, focus=self.P.REWRITE_FOCUS[focus], brief=self.brief,
                                                 measurements=parent.measurements)
            return self.module_call(request_id, model, effort, [
                *views, {"type": "input_image", "image_url": data_url(parent.sides), "detail": "high"}, *overlay,
                {"type": "input_text", "text": text}])
        extra = self.P.REWRITE_FOCUS[focus]
        if step is not None and step.feed_diffs:
            extra += self.memory()
        if step is not None and step.nudge:
            extra += self.P.NUDGES[step.nudge]
        patch = step is not None and step.patch
        call = self.module_call(request_id, model, effort, [
            {"type": "input_text", "text": "Current module, in order: 30 degrees left, 30 degrees right, from slightly below, from above; then the second-stage grid."},
            *views,
            {"type": "input_image", "image_url": data_url(parent.sides), "detail": "high"},
            {"type": "input_text", "text": (self.P.PATCH_REWRITE if patch else self.P.REWRITE).format(
                source=parent.raw, focus=extra)},
        ])
        if patch and call.instructions == self.P.GENERATE_CHAIN:
            call.instructions = self.P.GENERATE_PATCH
        return call

    def memory(self) -> str:
        if not self.changes:
            return ""
        lines = [self.P.CHANGES_HEAD]
        for number, changed, kept in self.changes[-4:]:
            lines.append(f"Round {number}: {self.P.CHANGE_KEPT if kept else self.P.CHANGE_REJECTED}")
            lines += [f"  {line}" for line in changed or [self.P.CHANGE_NONE]]
        return "\n\n" + "\n".join(lines)


    def render(self, name: str, source: str, front_options: RenderOptions = FRONT) -> Candidate | tuple[str, str]:
        try:
            front = self.renderer.render(source, front_options)
        except RenderRejected as rejection:
            return source, rejection.detail_json
        return Candidate(name, source, front, self.renderer.render(source, SIDES))

    def build(self, pool: ThreadPoolExecutor, named: list[tuple[str, Response | None]],
              repair_base: int) -> list[Candidate]:
        candidates: list[Candidate] = []
        rejected: list[tuple[str, str, str]] = []
        for name, reply in named:
            source = extract_source(reply.output_text) if reply is not None else None
            if source is None:
                continue
            outcome = self.render(name, normalize(source))
            if isinstance(outcome, Candidate):
                candidates.append(outcome)
            else:
                rejected.append((name, *outcome))
        repairs = [
            self.generation_call(repair_base + i, SOL, [{"type": "input_text", "text": self.P.REPAIR.format(
                errors=errors, source=source)}])
            for i, (_, source, errors) in enumerate(rejected)
        ]
        for (name, _, _), reply in zip(rejected, self.run_stage(pool, repairs), strict=True):
            source = extract_source(reply.output_text) if reply is not None else None
            if source is not None:
                outcome = self.render(name + "r", normalize(source))
                if isinstance(outcome, Candidate):
                    candidates.append(outcome)
        return candidates

    def render_module(self, name: str, source: str, raw: str = "") -> Candidate | tuple[str, str]:
        try:
            front = self.renderer.render(source, FRONT_LARGE)
            views = zoomed_views(front)
            grid = to_jpeg(self.renderer.render(source, GRID_VIEW))
        except RenderRejected as rejection:
            return source, rejection.detail_json
        candidate = Candidate(name, source, view_sheet(views), grid, raw, views)
        if self.keep_shots:
            candidate.shots = judge_shots(front)
        if self.photo is not None:
            candidate.metrics = metrics.grid_score(front, self.photo)
            candidate.measurements, candidate.overlay = metrics.feedback(front, self.photo)
        return candidate

    def modules(self, pool: ThreadPoolExecutor, calls: list[Call], names: list[str],
                bases: dict[int, str] | None = None) -> list[Candidate | None]:
        results: list[Candidate | None] = [None] * len(calls)
        rejected: list[tuple[int, str, str]] = []
        for index, reply in enumerate(self.run_stage(pool, calls)):
            if reply is None:
                raw = None
            elif bases is not None and index in bases:
                raw = self.patched(bases[index], reply)
            else:
                raw = extract_source(reply.output_text)
            if raw is None:
                continue
            outcome = self.render_module(names[index], normalize(raw, self.wrapper), raw)
            if isinstance(outcome, Candidate):
                results[index] = outcome
            else:
                rejected.append((index, raw, outcome[1]))
        repairs = [
            self.module_call(calls[index].request_id + REPAIR_OFFSET, SOL, calls[index].effort, [
                {"type": "input_text", "text": self.P.REPAIR.format(errors=errors, source=raw)}])
            for index, raw, errors in rejected
        ]
        for (index, _, _), reply in zip(rejected, self.run_stage(pool, repairs), strict=True):
            raw = extract_source(reply.output_text) if reply is not None else None
            if raw is not None:
                outcome = self.render_module(names[index], normalize(raw, self.wrapper), raw)
                if isinstance(outcome, Candidate):
                    results[index] = outcome
        return results

    def patched(self, base: str, reply: Response) -> str | None:
        text = reply.output_text
        blocks = edits.edit_blocks(text)
        if blocks:
            raw, applied, skipped = edits.apply_blocks(base, blocks)
            if not applied:
                raw = None
        else:
            raw, applied, skipped = extract_source(text), 0, 0
        if raw is not None and "export default function" not in raw:
            raw = None
        share = None
        if raw is not None and base:
            usage = reply.usage
            output = getattr(usage, "output_tokens", 0) or 0
            reasoning = getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0) or 0
            share = 1.0 if not blocks else (output - reasoning) / (len(base) / MODULE_CHARS_PER_TOKEN)
            self.patch_shares.append(share)
        self.patch_log.append({"blocks": len(blocks), "applied": applied, "skipped": skipped, "whole": not blocks,
                               "used": raw is not None, "share": share})
        return raw


    def judge_calls(self, candidates: list[Candidate], request_id: int, model: str = SOL, mirrored: bool = True,
                    instructions: str | None = None) -> tuple[dict[str, Candidate], list[Call]]:
        instructions = self.P.SELECT if instructions is None else instructions
        ordered = candidates.copy()
        random.Random(f"{self.seed}:{self.digest}:{request_id}").shuffle(ordered)
        labels = {chr(65 + i): candidate for i, candidate in enumerate(ordered)}
        calls = []
        orders = (list(labels), list(reversed(labels))) if mirrored else (list(labels),)
        for offset, order in enumerate(orders):
            content: list[dict[str, Any]] = [{"type": "input_text", "text": "TARGET:"}, self.target]
            for label in order:
                content += [
                    {"type": "input_text", "text": f"CANDIDATE {label} FRONT VIEWS:"},
                    {"type": "input_image", "image_url": png_data_url(labels[label].front), "detail": "high"},
                    {"type": "input_text", "text": f"CANDIDATE {label} SIDE VIEWS:"},
                    {"type": "input_image", "image_url": png_data_url(labels[label].sides), "detail": "low"},
                ]
            calls.append(Call(request_id + offset, model, instructions, content, SELECT_EFFORT, SELECT_MAX_OUTPUT,
                              text=self.P.select_schema(list(labels))))
        return labels, calls

    @staticmethod
    def parse_judgement(labels: dict[str, Candidate], replies: list[Response | None]) -> dict[str, Judgement]:
        scores: dict[str, list[float]] = {}
        fixes: dict[str, list[str]] = {}
        for reply in replies:
            if reply is None:
                continue
            try:
                entries = json.loads(reply.output_text)["candidates"]
            except (ValueError, KeyError, TypeError):
                continue
            seen = set()
            for entry in entries:
                label = entry.get("label")
                if label not in labels or label in seen:
                    continue
                seen.add(label)
                penalties = [entry[k] for k in ("front_left", "front_right", "front_below", "front_above")]
                score = sum(min(10, max(0, int(p))) for p in penalties) / 4
                if int(entry["garbage_sides"]) >= 3:
                    score += PENALTY_WEIGHT
                if "grid" in entry:
                    score += GRID_WEIGHT * min(10, max(0, int(entry["grid"])))
                scores.setdefault(label, []).append(score)
                fixes.setdefault(label, []).extend(str(f) for f in entry.get("fixes", [])[:6])
        return {
            labels[label].name: Judgement(sum(values) / len(values), list(dict.fromkeys(fixes[label]))[:8])
            for label, values in scores.items()
        }

    def judge(self, pool: ThreadPoolExecutor, candidates: list[Candidate], request_id: int, model: str = SOL,
              mirrored: bool = True, instructions: str | None = None) -> dict[str, Judgement]:
        instructions = self.P.SELECT if instructions is None else instructions
        labels, calls = self.judge_calls(candidates, request_id, model, mirrored, instructions)
        return self.parse_judgement(labels, self.run_stage(pool, calls))

    def rank_groups(self, pool: ThreadPoolExecutor, candidates: list[Candidate], group_size: int,
                   base: int) -> tuple[Candidate, Judgement | None]:
        alive = candidates.copy()
        verdict: Judgement | None = None
        round_index = 0
        while len(alive) > 1:
            random.Random(f"{self.seed}:{self.digest}:t{round_index}").shuffle(alive)
            size = max(2, group_size)
            groups = [alive[i:i + size] for i in range(0, len(alive), size)]
            if len(groups) > 1 and len(groups[-1]) == 1:
                groups[-2].extend(groups.pop())
            stage: list[tuple[dict[str, Candidate], int]] = []
            calls: list[Call] = []
            for index, group in enumerate(groups):
                labels, group_calls = self.judge_calls(group, base + 100 * round_index + 2 * index)
                stage.append((labels, len(group_calls)))
                calls += group_calls
            replies = self.run_stage(pool, calls)
            winners: list[Candidate] = []
            offset = 0
            for (labels, count), group in zip(stage, groups, strict=True):
                judgements = self.parse_judgement(labels, replies[offset:offset + count])
                offset += count
                winner = self.best(group, judgements)
                winners.append(winner)
                verdict = judgements.get(winner.name)
            alive = winners
            round_index += 1
        return alive[0], verdict

    def scores(self, pool: ThreadPoolExecutor, candidates: list[Candidate], request_id: int,
               instructions: str | None = None) -> dict[str, Judgement]:
        instructions = self.P.SELECT_CLOSE if instructions is None else instructions
        ordered = candidates.copy()
        random.Random(f"{self.seed}:{self.digest}:{request_id}").shuffle(ordered)
        labels = {chr(65 + i): candidate for i, candidate in enumerate(ordered)}
        calls = []
        for offset, order in enumerate((list(labels), list(reversed(labels)))):
            content: list[dict[str, Any]] = [{"type": "input_text", "text": "TARGET:"}, self.target, *self.closeup]
            if self.brief:
                content.append({"type": "input_text", "text": self.brief})
            for label in order:
                content += [
                    {"type": "input_text", "text": f"CANDIDATE {label} FRONT VIEWS:"},
                    {"type": "input_image", "image_url": data_url(labels[label].front), "detail": "high"},
                    {"type": "input_text", "text": f"CANDIDATE {label} GRID:"},
                    {"type": "input_image", "image_url": data_url(labels[label].sides), "detail": "high"},
                ]
            calls.append(Call(request_id + offset, SOL, instructions, content, SELECT_EFFORT, SELECT_MAX_OUTPUT,
                              text=self.P.select_close_schema(list(labels))))
        return self.parse_judgement(labels, self.run_stage(pool, calls))

    def select(self, pool: ThreadPoolExecutor, candidates: list[Candidate], parents: set[str], keep: int,
               request_id: int) -> list[Candidate]:
        if len(candidates) <= keep:
            return candidates
        judgements = self.scores(pool, candidates, request_id)
        if not judgements and parents:
            return ([c for c in candidates if c.name in parents] + [c for c in candidates if c.name not in parents])[:keep]

        def rank(item: tuple[int, Candidate]) -> tuple[float, int]:
            index, candidate = item
            judgement = judgements.get(candidate.name)
            score = judgement.score if judgement else 99.0
            return (score - MATCH_WEIGHT * candidate.metrics + (PARENT_MARGIN if candidate.name in parents else 0.0), index)

        return [candidate for _, candidate in sorted(enumerate(candidates), key=rank)[:keep]]


    def run(self, plan: Plan) -> str:
        with ThreadPoolExecutor(max_workers=8, thread_name_prefix="miner404") as pool:
            draft_calls = [
                self.generation_call(1 + i, model, [], effort=effort, instructions=instructions,
                                     max_output=plan.draft_max_output)
                for i, (model, effort, instructions) in enumerate(plan.drafts)
            ]
            replies = self.run_stage(pool, draft_calls)
            candidates = self.build(pool, [(f"c{1 + i}", reply) for i, reply in enumerate(replies)],
                                    repair_base=500)
            if not candidates:
                return fallback_module(self.image)
            self.best_so_far = candidates[0].source
            if len(candidates) == 1 or plan.group_size >= len(candidates):
                judgements = self.judge(pool, candidates, 1000)
                incumbent = self.best(candidates, judgements)
                verdict = judgements.get(incumbent.name)
            else:
                incumbent, verdict = self.rank_groups(pool, candidates, plan.group_size, base=1000)
            self.best_so_far = incumbent.source
            for round_index in range(1, plan.refine_rounds + 1):
                if verdict is None:
                    break
                base = 2000 + 100 * round_index
                calls = [
                    self.generation_call(base + i, SOL, [
                        {"type": "input_image", "image_url": png_data_url(incumbent.front), "detail": "high"},
                        {"type": "input_image", "image_url": png_data_url(incumbent.sides), "detail": "low"},
                        {"type": "input_text", "text": self.P.REFINE.format(
                            focus=self.P.REFINE_FOCUS["all" if focus == "blind" else focus],
                            fixes=("- (none listed)" if focus == "blind" else
                                   "\n".join(f"- {fix}" for fix in verdict.fixes) or "- (none listed)"),
                            source=incumbent.source)},
                    ], effort=plan.refine_effort)
                    for i, focus in enumerate(plan.refine_focuses)
                ]
                calls = self.affordable(calls)
                if not calls:
                    break
                replies = self.run_stage(pool, calls)
                children = self.build(pool, [(f"r{round_index}{c.request_id % 100}", r) for c, r in
                                             zip(calls, replies, strict=True)], repair_base=base + 50)
                if not children:
                    continue
                judgements = self.judge(pool, [incumbent, *children], base + 80)
                best = self.best([incumbent, *children], judgements)
                if judgements.get(best.name) is not None:
                    incumbent, verdict = best, judgements[best.name]
                    self.best_so_far = incumbent.source
            return incumbent.source

    def run_stages(self, plan: StagePlan) -> str:
        self.wrapper = chain_wrapper(plan.fill, plan.glow, plan.outline_fit or plan.measured)
        self.keep_shots = any(step.judge_pick for step in plan.steps)
        closeup = object_closeup(self.image)
        if closeup is not None:
            self.closeup = [{"type": "input_image", "image_url": data_url(closeup), "detail": "high"}]
        with ThreadPoolExecutor(max_workers=8, thread_name_prefix="miner404") as pool:
            if plan.measured:
                self.instructions = self.P.GENERATE_MEASURED
                self.astra_cap = MEASURED_ASTRA_MAX_OUTPUT
                self.photo = metrics.reference_stats(self.image)
                self.brief = self.summary(pool)
            calls = [self.module_call(DRAFT_ID + i, draft.model, draft.effort, [], draft.start)
                     for i, draft in enumerate(plan.drafts)]
            survivors = [c for c in self.modules(pool, calls, [f"d{i}" for i in range(len(calls))]) if c is not None]
            if not survivors:
                calls = [self.module_call(REDRAFT_ID + i, SOL, "high", [], draft.start)
                         for i, draft in enumerate(plan.drafts[:3])]
                survivors = [c for c in self.modules(pool, calls, [f"e{i}" for i in range(len(calls))]) if c is not None]
            if not survivors:
                return fallback_module(self.image)
            self.best_so_far = survivors[0].source
            if plan.draft_keep:
                valid = list(survivors)
                survivors = self.drafts_kept(pool, survivors, plan)
                if plan.merge_pick:
                    survivors[0] = self.merged_pick(pool, valid, survivors[0])
                self.best_so_far = survivors[0].source
            for index, step in enumerate(plan.steps):
                if plan.replan and index > 0:
                    step = self.fit_rest(plan.steps[index:], survivors[0])[0]
                survivors = self.advance(pool, survivors, step, index)
                self.best_so_far = survivors[0].source
            winner = self.select(pool, survivors, set(), 1, FINAL_SELECT_ID)[0]
            self.best_so_far = winner.source
            return self.frame_step(winner, plan) if plan.outline_fit_last else winner.source

    def frame_step(self, winner: Candidate, plan: StagePlan) -> str:
        if not winner.raw:
            return winner.source
        source = normalize(winner.raw, chain_wrapper(plan.fill, plan.glow, True))
        try:
            front = self.renderer.render(source, FRONT_LARGE)
        except RenderRejected:
            return winner.source
        return source if frames_close(zoomed_views(front), winner.views, FRAME_LIMIT) else winner.source

    def summary(self, pool: ThreadPoolExecutor) -> str:
        call = Call(BRIEF_ID, SOL, self.P.BRIEF, [self.target, *self.closeup], SELECT_EFFORT, BRIEF_MAX_OUTPUT,
                    text=self.P.brief_schema())
        reply = self.run_stage(pool, [call])[0]
        try:
            brief = json.loads(reply.output_text) if reply is not None else None
            return self.P.BRIEF_BLOCK.format(
                main_object=brief["main_object"], items="; ".join(brief["items"]), features="; ".join(brief["features"]),
                pose=brief["pose"], true_colors="; ".join(brief["true_colors"]))
        except (TypeError, KeyError, ValueError):
            return ""

    def drafts_kept(self, pool: ThreadPoolExecutor, drafts: list[Candidate], plan: StagePlan) -> list[Candidate]:
        ordered = self.select(pool, drafts, set(), len(drafts) - 1 if plan.draft_each_start and len(drafts) > plan.draft_keep else plan.draft_keep, SELECT_ID)
        if not plan.draft_each_start or len(drafts) <= plan.draft_keep:
            return ordered
        kept, starts = [], set()
        for draft in ordered:
            start = plan.drafts[int(draft.name[1:])].start
            if start not in starts:
                starts.add(start)
                kept.append(draft)
        kept += [draft for draft in ordered if draft not in kept]
        return kept[:plan.draft_keep]

    def merged_pick(self, pool: ThreadPoolExecutor, drafts: list[Candidate], picked: Candidate) -> Candidate:
        if len(drafts) < 2:
            return picked
        labels = {chr(65 + i): draft for i, draft in enumerate(drafts)}
        content: list[dict[str, Any]] = [{"type": "input_text", "text": "TARGET:"}, self.target, *self.closeup]
        for label, draft in labels.items():
            content += [{"type": "input_text", "text": f"CANDIDATE {label} FRONT VIEWS:"},
                        {"type": "input_image", "image_url": data_url(draft.front), "detail": "high"}]
        for label, draft in labels.items():
            content += [{"type": "input_text", "text": f"CANDIDATE {label} GRID:"},
                        {"type": "input_image", "image_url": data_url(draft.sides), "detail": "high"}]
        content.append({"type": "input_text", "text": self.P.CRITIQUE})
        critique = Call(CRITIQUE_ID, SOL, self.P.CRITIQUE_SYSTEM, content, CRITIQUE_EFFORT, SELECT_MAX_OUTPUT,
                        text=self.P.critique_schema(list(labels)))
        reply = self.run_stage(pool, [critique])[0]
        if reply is None:
            return picked
        notes: dict[str, tuple[list[str], list[str]]] = {}
        try:
            verdict = json.loads(reply.output_text)
            for entry in verdict["candidates"]:
                if entry["label"] in labels:
                    notes[entry["label"]] = ([str(n) for n in entry["strengths"]], [str(n) for n in entry["weaknesses"]])
            base = verdict["base"]
        except (ValueError, KeyError, TypeError):
            return picked
        if not isinstance(base, str) or base not in labels:
            base = next((label for label, draft in labels.items() if draft is picked), next(iter(labels)))
        others = sorted((label for label in labels if label != base),
                        key=lambda label: -len(notes.get(label, ([], []))[0]))[:2]
        blocks = []
        for role, label in [("BASE", base), *(("OTHER", label) for label in others)]:
            strengths, weaknesses = notes.get(label, ([], []))
            blocks.append(self.P.MERGE_BLOCK.format(role=role, label=label, strengths="; ".join(strengths) or "none listed",
                                                     weaknesses="; ".join(weaknesses) or "none listed",
                                                     source=labels[label].raw))
        text = "\n\n".join([self.P.MERGE, *blocks])
        merge = Call(MERGE_ID, SOL, self.instructions, [{"type": "input_text", "text": text}, self.target, *self.closeup],
                     GENERATE_EFFORT, GENERATE_MAX_OUTPUT)
        merged = self.modules(pool, [merge], ["dm"])[0]
        return picked if merged is None else merged

    def judge_pick(self, pool: ThreadPoolExecutor, pairs: list[tuple[Candidate, Candidate]],
                   request_ids: list[int]) -> list[bool]:
        asked: list[tuple[int, list[Call]]] = []
        for position, ((parent, child), request_id) in enumerate(zip(pairs, request_ids, strict=True)):
            if len(parent.shots) < 2 or len(child.shots) < 2:
                continue
            calls = [self.stage1_call(request_id, "front_left", parent.shots[0], child.shots[0]),
                     self.stage1_call(request_id + 1, "front_right", child.shots[1], parent.shots[1])]
            asked.append((position, calls))
        replies = self.run_stage(pool, [call for _, calls in asked for call in calls])
        verdicts = [False] * len(pairs)
        for number, (position, _) in enumerate(asked):
            gaps = []
            for reply, sign in zip(replies[2 * number:2 * number + 2], (1, -1), strict=True):
                penalties = stage1_penalties(reply)
                if penalties is not None:
                    gaps.append(sign * (penalties[0] - penalties[1]))
            verdicts[position] = bool(gaps) and sum(gaps) / len(gaps) >= -JUDGE_DRAW
        return verdicts

    def stage1_call(self, request_id: int, angle: str, first: bytes, second: bytes) -> Call:
        content = [{"type": "input_text", "text": self.P.JUDGE_REFERENCE}, self.target,
                   {"type": "input_text", "text": self.P.JUDGE_MODEL_1},
                   {"type": "input_image", "image_url": png_data_url(first), "detail": "high"},
                   {"type": "input_text", "text": self.P.JUDGE_MODEL_2},
                   {"type": "input_image", "image_url": png_data_url(second), "detail": "high"},
                   {"type": "input_text", "text": self.P.JUDGE_USER.format(angle_desc=self.P.JUDGE_ANGLES[angle])}]
        return Call(request_id, SOL, self.P.JUDGE_SYSTEM, content, SELECT_EFFORT, JUDGE_MAX_OUTPUT,
                    text=self.P.judge_schema())

    def fit_rest(self, rest: tuple[Step, ...], incumbent: Candidate) -> tuple[Step, ...]:
        written = [_written(step) for step in rest]
        code = len(incumbent.raw or incumbent.source) * (1 + MODULE_GROWTH * len(rest) / 9)
        module_tokens = code / MODULE_CHARS_PER_TOKEN
        share = PATCH_PRIOR
        if self.patch_shares:
            share = min(1.0, max(PATCH_FLOOR, sum(self.patch_shares) / len(self.patch_shares)))
        usable = CREDIT_LIMIT * (1 - SAFETY)
        text_chars = len(self.instructions) + code + len(self.P.PATCH_REPLY) + FORECAST_EXTRA_CHARS
        judge_worst = worst_case(SOL, 4000, 9, SELECT_MAX_OUTPUT)

        def fits(pairs: list[tuple[str, str] | None]) -> bool:
            spent = self.budget.credits
            for step, pair in zip(rest, pairs, strict=True):
                if pair is None:
                    continue
                model, effort = pair
                cap = self.astra_cap if model == ASTRA else EFFORT_MAX_OUTPUT.get(effort, GENERATE_MAX_OUTPUT)
                if spent + worst_case(model, text_chars, FORECAST_IMAGES, cap) > usable:
                    return False
                reasoning, other = ROUND_COST.get(pair, ROUND_COST[(SOL, "xhigh")])
                spent += call_credits(model, other + module_tokens, reasoning + module_tokens * (share if step.patch else 1.0))
                if spent + 2 * judge_worst > usable:
                    return False
                spent += 2 * SELECT_CALL_CREDITS
            return True

        xhigh = [i for i, pair in enumerate(written) if pair == (SOL, "xhigh")]
        later = xhigh[len(xhigh) - len(xhigh) // 2:]
        earlier = xhigh[:len(xhigh) - len(xhigh) // 2]
        astra = [i for i, pair in enumerate(written) if pair is not None and pair[0] == ASTRA]
        moves = ([(i, (SOL, "high")) for i in reversed(later)] + [(i, (SOL, "xhigh")) for i in reversed(astra)]
                 + [(i, (SOL, "high")) for i in reversed(earlier)])
        pairs = list(written)
        for i, pair in [(None, None), *moves]:
            if i is not None:
                pairs[i] = pair
            if fits(pairs):
                break
        return tuple(step if pair == old else _with_rewrite(step, pair)
                     for step, pair, old in zip(rest, pairs, written, strict=True))

    def advance(self, pool: ThreadPoolExecutor, survivors: list[Candidate], step: Step, index: int) -> list[Candidate]:
        select_id = SELECT_ID + 10 * (index + 1)
        if not step.rewrites and not step.lineage:
            parents = [c.parent for c in survivors if c.parent is not None]
            return self.select(pool, [*survivors, *parents], {p.name for p in parents}, step.keep or 1, select_id)
        calls, names, owners = [], [], []
        bases: dict[int, str] = {}
        for position, parent in enumerate(survivors):
            rewrites = (step.lineage[position % len(step.lineage)],) if step.lineage else step.rewrites
            focus = step.focus[position % len(step.focus)] if step.focus else ""
            for number, (model, effort) in enumerate(rewrites):
                request_id = REWRITE_BASE * (index + 1) + 10 * position + number
                call = self.rewrite_call(request_id, model, effort, parent, focus, step)
                if step.fallback is not None and not self.fits(call):
                    call = self.rewrite_call(request_id, *step.fallback, parent, focus, step)
                if step.patch:
                    bases[len(calls)] = parent.raw
                calls.append(call)
                names.append(f"r{index}.{position}.{number}")
                owners.append(position)
        children = self.modules(pool, calls, names, bases if step.patch else None)
        for owner, child in zip(owners, children, strict=True):
            if child is not None:
                child.parent = survivors[owner]
        if step.each_lineage and step.judge_pick:
            pairs = [(k, survivors[owner], child) for k, (owner, child) in enumerate(zip(owners, children, strict=True))
                     if child is not None]
            verdicts = self.judge_pick(pool, [(parent, child) for _, parent, child in pairs],
                                       [select_id + 2 * k for k, _, _ in pairs])
            kept = list(survivors)
            for (k, parent, child), child_kept in zip(pairs, verdicts, strict=True):
                if child_kept:
                    kept[owners[k]] = child
                self.changes.append((index + 1, code_changes(parent.raw, child.raw), child_kept))
            return kept
        if step.each_lineage:
            judgements = self.scores(pool, [*[c for c in children if c is not None], *survivors], select_id)
            kept = list(survivors)
            for owner, child in zip(owners, children, strict=True):
                if child is None:
                    continue
                mine, parent = judgements.get(child.name), judgements.get(survivors[owner].name)
                if mine is not None and parent is not None and mine.score <= parent.score + PARENT_MARGIN:
                    kept[owner] = child
            return kept
        if step.keep:
            valid = [child for child in children if child is not None]
            return self.select(pool, [*valid, *survivors], {c.name for c in survivors}, step.keep, select_id)
        newest = list(survivors)
        for owner, child in zip(owners, children, strict=True):
            if child is not None:
                newest[owner] = child
        return newest

    def affordable(self, calls: list[Call]) -> list[Call]:
        judge_estimate = worst_case(SOL, 4000, 9, SELECT_MAX_OUTPUT)
        while calls:
            estimates = []
            for call in calls:
                text_chars = len(call.instructions) + sum(len(c.get("text", "")) for c in call.content)
                images = sum(1 for c in call.content if c["type"] == "input_image")
                estimates.append(worst_case(call.model, text_chars, images, call.max_output_tokens))
            if self.budget.allows(estimates, keep_calls=2 + len(calls), keep_credits=2 * judge_estimate):
                return calls
            calls = calls[:-1]
        return calls

    @staticmethod
    def best(candidates: list[Candidate], judgements: dict[str, Judgement]) -> Candidate:
        def key(item: tuple[int, Candidate]) -> tuple[float, int]:
            index, candidate = item
            judgement = judgements.get(candidate.name)
            return (judgement.score if judgement else 99.0, index)

        return min(enumerate(candidates), key=key)[1]


BASIC_PLAN = Plan(drafts=((SOL, "high", "base"),) * 9, draft_max_output=32768)


def run(image: bytes, seed: int, calls: ModelCalls, renderer: Renderer, plan: Plan | StagePlan) -> str | None:
    pipeline = Pipeline(image, seed, calls, renderer)
    try:
        return pipeline.run_stages(plan) if isinstance(plan, StagePlan) else pipeline.run(plan)
    except (APIConnectionError, CallFailed, WitnessError, ExchangePending, RenderUnavailable):
        raise
    except Exception:
        return pipeline.best_so_far or fallback_module(image)
