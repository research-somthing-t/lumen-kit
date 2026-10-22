from __future__ import annotations

import re

_BLOCK = re.compile(
    r"^<{5,9} ?SEARCH[ \t]*\n(?P<search>.*?)^={5,9}[ \t]*\n(?P<replace>.*?)^>{5,9} ?REPLACE[ \t]*$",
    re.MULTILINE | re.DOTALL,
)


def edit_blocks(reply: str) -> list[tuple[str, str]]:
    return [(found.group("search"), found.group("replace")) for found in _BLOCK.finditer(reply)]


def _apply_one(code: str, search: str, replace: str) -> str | None:
    if code.count(search) == 1:
        return code.replace(search, replace, 1)
    if code.count(search) > 1:
        return None
    lines = code.split("\n")
    wanted = [line.rstrip() for line in search.split("\n")]
    if wanted and wanted[-1] == "":
        wanted.pop()
    spans = [start for start in range(len(lines) - len(wanted) + 1)
             if all(lines[start + k].rstrip() == wanted[k] for k in range(len(wanted)))]
    if len(spans) != 1:
        return None
    new = replace.split("\n")
    if new and new[-1] == "":
        new.pop()
    start = spans[0]
    return "\n".join(lines[:start] + new + lines[start + len(wanted):])


def apply_blocks(code: str, blocks: list[tuple[str, str]]) -> tuple[str, int, int]:
    closed = code.endswith("\n")
    current = code if closed else code + "\n"
    applied = skipped = 0
    for search, replace in blocks:
        result = None if not search.strip() else _apply_one(current, search, replace)
        if result is None:
            skipped += 1
            continue
        current = result
        applied += 1
    if not closed and current.endswith("\n"):
        current = current[:-1]
    return current, applied, skipped
