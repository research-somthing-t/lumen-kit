from __future__ import annotations

import inspect
import io
import json
import re
import time

from PIL import Image

from . import looks, prompts, views
from .budget import CALL_LIMIT, Ledger
from .jsutil import extract, final, wrap

MODEL = "gpt-6.1-sol"
EFFORT = "medium"
REFINE_EFFORT = "high"
SERVICE_TIER = "default"
CALLS = 3
REPAIRS = 1
CHAINS = 4
PANEL_MODEL = "gpt-6-astra"
PANEL_EFFORT = "low"
PANEL_WEIGHT = 2.0
FINAL_REFINES = 2
DETAIL_WEIGHT = 1.0
PROVIDER_RETRIES = 2
BACKOFF_SECONDS = 5
EXPECTED_CALL_CREDITS = 450_000
PLAN_CHARS = 2_500
CHECK_ITEMS = 7
PICK_EFFORT = "low"
PICK_PX = 1024
PICK_CALL_CREDITS = 40_000
SEMIS = ((0, 1), (2, 3))
S1_DRAW_GAP = 0.7
S1_BOTH_BAD_AVG = 8.0
S1_BOTH_BAD_GAP = 1.5
S1_SPLIT_CONSISTENCY = 0.65
S1_SPLIT_MIN_SIGNED = 3
FINAL_MODEL = "gpt-6-astra"
FINAL_EFFORT = "high"
FINAL_MAX_OUTPUT = 32_000
ASTRA_WORST = 2_000_000
ASTRA_CREDIT_LIMIT = 4_900_000
_ITEM = re.compile(r"\s*(?:[-*\u2022]|\d+[.)])\s+(.+)")


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
          effort: str | None = None, instructions: str | None = None, content: list | None = None,
          max_output: int | None = None) -> str | None:
    model, effort = model or MODEL, effort or EFFORT
    instructions = prompts.GEN_SYSTEM if instructions is None else instructions
    if content is None:
        content = [{"type": "input_text", "text": text}]
        content += [{"type": "input_image", "image_url": views.data_url(p), "detail": "high"} for p in pngs]
    ids = {"request_id": ledger.calls} if ledger.calls and _fresh_ids(witness) else {}
    cap = {"max_output_tokens": max_output} if max_output else {}
    try:
        result = witness.responses.create(
            model=model, instructions=instructions,
            reasoning={"effort": effort}, service_tier=SERVICE_TIER,
            input=[{"role": "user", "content": content}], **ids, **cap,
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
    if max_output and (getattr(result, "status", None) == "incomplete"
                       or int(usage.get("output_tokens") or 0) >= max_output):
        return None
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


def _checklist(notes: str) -> list[str]:
    lines = notes.splitlines()
    for i, line in enumerate(lines):
        if not line.strip().lstrip("#*_ ").lower().startswith("checklist"):
            continue
        items = []
        for row in lines[i + 1:]:
            m = _ITEM.match(row)
            if m is None and row.strip():
                break
            if m is not None:
                items.append(m.group(1).strip())
        return items[:CHECK_ITEMS]
    return []


def _plan(witness, ledger: Ledger, ref: bytes) -> tuple[list[str], str]:
    notes = (_call(witness, ledger, "plan", prompts.PLAN_USER, [ref], instructions=prompts.PLAN_SYSTEM) or "").strip()
    if len(notes) > PLAN_CHARS:
        notes = notes[:PLAN_CHARS].rsplit("\n", 1)[0].rstrip()
    head = prompts.PLAN_NOTES % notes if notes else ""
    gens = [head + (prompts.GEN_USER_ALT if i % 2 else prompts.GEN_USER) for i in range(CHAINS)]
    return gens, ("\n".join("- " + item for item in _checklist(notes)) if notes else "")


def _chain(witness, renderer, ledger: Ledger, ref: bytes, gen: str, check: str) -> str | None:
    stop = ledger.calls + CALLS
    spare = REPAIRS
    retries, seen = PROVIDER_RETRIES, ledger.provider_errors
    best = prior = None
    later = REFINE_EFFORT or EFFORT
    sheets = {}
    js = _ask(witness, ledger, "gen", gen, [ref])
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
            js = _ask(witness, ledger, "gen", gen, [ref], later)
        elif err is not None:
            js = _ask(witness, ledger, "repair", prompts.REPAIR_USER % (err, js), [ref], later)
        else:
            js = js if js is not None else best
            sheet, rejected = _sheet_or_error(renderer, wrap(js), sheets)
            if rejected is not None:
                best = prior
                js = _ask(witness, ledger, "repair", prompts.REPAIR_USER % (rejected, js), [ref], later)
                continue
            text = prompts.REFINE_USER % js + (prompts.CHECK_REFINE % check if check else "")
            js = _ask(witness, ledger, "refine", text, [ref, sheet], later)
    return best


def _penalties(text: str, n: int) -> list[float] | None:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        pens = [float(x) for x in json.loads(m.group(0))["penalties"]] if m else []
    except (ValueError, KeyError, TypeError):
        return None
    return pens if len(pens) == n else None


def _panel(witness, renderer, ledger: Ledger, ref: bytes, modules: list[str], check: str) -> str:
    sheets = [views.front_sheet(renderer.render, wrap(js)) for js in modules]
    extra = prompts.CHECK_PANEL % check if check else ""
    n = len(modules)
    sums, counts = [0.0] * n, [0] * n
    for order in (list(range(n)), list(range(n - 1, -1, -1))):
        text = _call(witness, ledger, "pick", prompts.PANEL_USER % (n + 1, extra), [ref, *(sheets[i] for i in order)],
                     model=PANEL_MODEL, effort=PANEL_EFFORT, instructions=prompts.PICK_SYSTEM) or ""
        pens = _penalties(text, n)
        if pens is None:
            continue
        for position, i in enumerate(order):
            sums[i] += pens[position]
            counts[i] += 1
    mean = [sums[i] / counts[i] if counts[i] else None for i in range(n)]
    detail = [looks.detail(sheet, views.FRONT_PX) for sheet in sheets]
    return modules[looks.choose(mean, detail, PANEL_WEIGHT, DETAIL_WEIGHT)]


def _shots(renderer, js: str) -> list[bytes] | None:
    try:
        return [renderer.render(final(js), views.view_options(t, p, PICK_PX)) for t, p, _ in prompts.JUDGE_VIEWS]
    except Exception as exc:
        if type(exc).__name__ != "RenderRejected":
            raise
        return None


def _pair_penalties(text: str | None) -> tuple[float, float] | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
        pens = float(d["penalty_1"]), float(d["penalty_2"])
    except (ValueError, KeyError, TypeError):
        return None
    return pens if all(0 <= p <= 10 for p in pens) else None


def _s1_choice(answers: list) -> int | None:
    kept = []
    for ab, ba in answers:
        if ab is None or ba is None or (ab[0] - ab[1]) * (ba[1] - ba[0]) < 0:
            continue
        kept.append(((ab[0] + ba[1]) / 2, (ab[1] + ba[0]) / 2))
    if not kept:
        return None
    wpa, wpb = sum(a for a, _ in kept) / len(kept), sum(b for _, b in kept) / len(kept)
    diff, avg = wpa - wpb, (wpa + wpb) / 2
    if (avg >= S1_BOTH_BAD_AVG and abs(diff) <= S1_BOTH_BAD_GAP) or abs(diff) <= S1_DRAW_GAP:
        return None
    signed = [a < b for a, b in kept if a != b]
    if len(signed) >= S1_SPLIT_MIN_SIGNED and abs(2 * sum(signed) - len(signed)) / len(signed) < S1_SPLIT_CONSISTENCY:
        return None
    return 0 if diff < 0 else 1


def _judge_content(ref: bytes, one: bytes, two: bytes, desc: str) -> list:
    content = []
    for label, png in zip(prompts.JUDGE_LABELS, (ref, one, two)):
        content += [{"type": "input_text", "text": label},
                    {"type": "input_image", "image_url": views.data_url(png), "detail": "high"}]
    return content + [{"type": "input_text", "text": prompts.JUDGE_USER.format(angle_desc=desc)}]


def _match(witness, ledger: Ledger, ref: bytes, shots: dict, a: int, b: int) -> int:
    answers = []
    for v, (_, _, desc) in enumerate(prompts.JUDGE_VIEWS):
        answers.append([_pair_penalties(_call(witness, ledger, "pick", "", [], effort=PICK_EFFORT,
                                              instructions=prompts.JUDGE_SYSTEM,
                                              content=_judge_content(ref, shots[x][v], shots[y][v], desc)))
                        for x, y in ((a, b), (b, a))])
    choice = _s1_choice(answers)
    if choice is not None:
        return (a, b)[choice]
    detail = [looks.views_detail(shots[i]) for i in (a, b)]
    return (a, b)[looks.choose([0.0, 0.0], detail, PANEL_WEIGHT, DETAIL_WEIGHT)]


def _knockout(witness, renderer, ledger: Ledger, ref: bytes, bests: list[str | None]) -> str | None:
    semis = [[i for i in pair if bests[i] is not None] for pair in SEMIS]
    calls = 2 * len(prompts.JUDGE_VIEWS) * (sum(len(s) == 2 for s in semis) + all(semis))
    if ledger.calls + calls > ledger.call_cap or ledger.credits + calls * PICK_CALL_CREDITS > ledger.credit_cap:
        return None
    shots = {}
    for i in (i for s in semis for i in s):
        got = _shots(renderer, bests[i])
        if got is not None:
            shots[i] = got
    if not shots:
        return next(js for js in bests if js is not None)
    winners = []
    for semi in ([i for i in s if i in shots] for s in semis):
        if len(semi) == 2:
            winners.append(_match(witness, ledger, ref, shots, *semi))
        elif semi:
            winners.append(semi[0])
    return bests[winners[0] if len(winners) == 1 else _match(witness, ledger, ref, shots, *winners)]


def _pick(witness, renderer, ledger: Ledger, ref: bytes, bests: list[str | None], found: list[str],
          check: str) -> str:
    best = _knockout(witness, renderer, ledger, ref, bests)
    return best if best is not None else _panel(witness, renderer, ledger, ref, found, check)


def _astra_fits(ledger: Ledger) -> bool:
    return ledger.calls < ledger.call_cap and ledger.credits + ASTRA_WORST <= ASTRA_CREDIT_LIMIT


def _final_refine(witness, renderer, ledger: Ledger, ref: bytes, best: str, check: str) -> str:
    for _ in range(FINAL_REFINES):
        astra = _astra_fits(ledger)
        if not astra and not ledger.can_call(EXPECTED_CALL_CREDITS):
            break
        sheet, rejected = _sheet_or_error(renderer, wrap(best), {})
        if rejected is not None:
            break
        text = prompts.REFINE_USER % best + (prompts.CHECK_REFINE % check if check else "")
        if astra:
            js = extract(_call(witness, ledger, "final", text, [ref, sheet], model=FINAL_MODEL, effort=FINAL_EFFORT,
                               max_output=FINAL_MAX_OUTPUT))
        else:
            js = _ask(witness, ledger, "final", text, [ref, sheet], REFINE_EFFORT)
        wrapped = wrap(js) if js is not None else None
        if wrapped is not None and _sheet_or_error(renderer, wrapped, {})[1] is None:
            best = js
    return best


def run(image: bytes, seed: int, witness, renderer) -> str | None:
    ref = _png(image)
    ledger = Ledger(call_cap=CALL_LIMIT)
    gens, check = _plan(witness, ledger, ref)
    bests = [_chain(witness, renderer, ledger, ref, gen, check) for gen in gens]
    found = [js for js in bests if js is not None]
    best = None if not found else found[0] if len(found) == 1 else \
        _pick(witness, renderer, ledger, ref, bests, found, check)
    if best is not None:
        best = _final_refine(witness, renderer, ledger, ref, best, check)
    return final(best) if best is not None else None
