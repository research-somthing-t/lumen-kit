from __future__ import annotations

import inspect
import io
import json
import re
import time

from PIL import Image

from . import prompts, views
from .budget import CALL_LIMIT, Ledger
from .jsutil import extract, final, wrap

MODEL = "gpt-6.1-sol"
EFFORT = "medium"
REFINE_EFFORT = "high"
SERVICE_TIER = "default"
CALLS = 3
REPAIRS = 1
CHAINS = 2
PANEL_MODEL = "gpt-6-astra"
PANEL_EFFORT = "low"
PROVIDER_RETRIES = 2
BACKOFF_SECONDS = 5
EXPECTED_CALL_CREDITS = 450_000


def _png(image: bytes) -> bytes:
    im = Image.open(io.BytesIO(image)).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def _fresh_ids(witness) -> bool:
    try:
        return "request_id" in inspect.signature(witness.responses.create).parameters
    except (TypeError, ValueError):
        return False


def _call(witness, ledger: Ledger, kind: str, text: str, pngs: list[bytes], *, model: str | None = None,
          effort: str | None = None, instructions: str | None = None) -> str | None:
    model, effort = model or MODEL, effort or EFFORT
    instructions = prompts.GEN_SYSTEM if instructions is None else instructions
    content = [{"type": "input_text", "text": text}]
    content += [{"type": "input_image", "image_url": views.data_url(p), "detail": "high"} for p in pngs]
    ids = {"request_id": ledger.calls} if ledger.calls and _fresh_ids(witness) else {}
    try:
        result = witness.responses.create(
            model=model, instructions=instructions,
            reasoning={"effort": effort}, service_tier=SERVICE_TIER,
            input=[{"role": "user", "content": content}], **ids,
        )
    except Exception as exc:
        if getattr(exc, "source", None) != "provider" and not _provider_failure(exc):
            raise
        ledger.charge(model, None, kind)
        ledger.provider_errors += 1
        time.sleep(BACKOFF_SECONDS)
        return None
    usage = getattr(result, "usage", None) or {}
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    ledger.charge(model, usage, kind)
    return getattr(result, "output_text", None)


def _provider_failure(exc) -> bool:
    try:
        import openai
    except ImportError:
        return False
    return isinstance(exc, openai.APIError)


def _ask(witness, ledger: Ledger, kind: str, text: str, pngs: list[bytes], effort: str | None = None) -> str | None:
    return extract(_call(witness, ledger, kind, text, pngs, effort=effort))


def _rejection(exc) -> str:
    return str(getattr(exc, "detail_json", None) or getattr(exc, "detail", None) or exc)[:1500]


def _sheet_or_error(renderer, source: str, sheets: dict) -> tuple[bytes | None, str | None]:
    if source in sheets:
        return sheets[source], None
    try:
        png = views.sheet(renderer.render, source)
    except Exception as exc:
        if type(exc).__name__ != "RenderRejected":
            raise
        return None, _rejection(exc)
    sheets[source] = png
    return png, None


def _chain(witness, renderer, ledger: Ledger, ref: bytes) -> str | None:
    stop = ledger.calls + CALLS
    spare = REPAIRS
    retries, seen = PROVIDER_RETRIES, ledger.provider_errors
    best = prior = None
    later = REFINE_EFFORT or EFFORT
    sheets = {}
    js = _ask(witness, ledger, "gen", prompts.GEN_USER, [ref])
    while True:
        err = None
        if js is not None:
            wrapped = wrap(js)
            err = "the module must contain `export default function generate(THREE)`" if wrapped is None \
                else _sheet_or_error(renderer, wrapped, sheets)[1]
            if err is None:
                prior, best = best, js
        while ledger.provider_errors > seen:
            seen += 1
            if retries > 0:
                retries -= 1
                stop += 1
        if ledger.calls >= stop and best is None and spare > 0:
            spare -= 1
            stop += 1
        if ledger.calls >= stop or not ledger.can_call(EXPECTED_CALL_CREDITS):
            break
        if js is None and best is None:
            js = _ask(witness, ledger, "gen", prompts.GEN_USER, [ref], later)
        elif err is not None:
            js = _ask(witness, ledger, "repair", prompts.REPAIR_USER % (err, js), [ref], later)
        else:
            js = js if js is not None else best
            sheet, rejected = _sheet_or_error(renderer, wrap(js), sheets)
            if rejected is not None:
                best = prior
                js = _ask(witness, ledger, "repair", prompts.REPAIR_USER % (rejected, js), [ref], later)
                continue
            js = _ask(witness, ledger, "refine", prompts.REFINE_USER % js, [ref, sheet], later)
    return best


def _panel(witness, renderer, ledger: Ledger, ref: bytes, modules: list[str]) -> str:
    sheets = [views.front_sheet(renderer.render, wrap(js)) for js in modules]
    text = _call(witness, ledger, "pick", prompts.PANEL_USER % (len(modules) + 1), [ref, *sheets], model=PANEL_MODEL,
                 effort=PANEL_EFFORT, instructions=prompts.PICK_SYSTEM) or ""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        pens = [float(x) for x in json.loads(m.group(0))["penalties"]] if m else []
    except (ValueError, KeyError, TypeError):
        pens = []
    if len(pens) != len(modules):
        return modules[0]
    return modules[min(range(len(modules)), key=lambda i: (pens[i], i))]


def run(image: bytes, seed: int, witness, renderer) -> str | None:
    ref = _png(image)
    ledger = Ledger(call_cap=CALL_LIMIT)
    found = [js for js in (_chain(witness, renderer, ledger, ref) for _ in range(CHAINS)) if js is not None]
    best = None if not found else found[0] if len(found) == 1 else _panel(witness, renderer, ledger, ref, found)
    return final(best) if best is not None else None
