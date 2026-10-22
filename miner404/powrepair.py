from __future__ import annotations

KEYWORDS = frozenset(("return", "typeof", "void", "delete", "await", "in", "of", "instanceof", "new", "case", "do",
                      "else", "throw", "yield"))
_UNARY_WORDS = frozenset(("typeof", "void", "delete", "await"))
_IDENT = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_$")
_QUOTES = "'\"`"
_SPACE = " \t\r\n"
_CHAINS = ")]'\"`"
_CLOSER = {"(": ")", "[": "]", "{": "}"}
_OPENER = {")": "(", "]": "[", "}": "{"}
_NUMBER_BODY = frozenset("0123456789._")


def _string_end(code: str, i: int) -> int | None:
    quote, j = code[i], i + 1
    while j < len(code):
        c = code[j]
        if c == "\n":
            return None
        if c == "\\":
            if code[j + 1:j + 2] in ("", "\n"):
                return None
            j += 2
            continue
        if c == quote:
            return j + 1
        j += 1
    return None


def _expression_end(code: str, i: int) -> int | None:
    depth, j = 0, i
    while j < len(code):
        c = code[j]
        if c in "'\"":
            end = _string_end(code, j)
        elif c == "`":
            end = _template_end(code, j)
        elif code.startswith("//", j):
            end = code.find("\n", j)
            end = None if end < 0 else end
        elif code.startswith("/*", j):
            end = code.find("*/", j + 2)
            end = None if end < 0 else end + 2
        else:
            if c == "{":
                depth += 1
            elif c == "}":
                if depth == 0:
                    return j
                depth -= 1
            end = j + 1
        if end is None:
            return None
        j = end
    return None


def _template_end(code: str, i: int) -> int | None:
    j = i + 1
    while j < len(code):
        c = code[j]
        if c == "\\":
            j += 2
            continue
        if c == "`":
            return j + 1
        if code.startswith("${", j):
            end = _expression_end(code, j + 2)
            if end is None:
                return None
            j = end + 1
            continue
        j += 1
    return None


def _masked(code: str) -> str | None:
    out = list(code)
    i = 0
    while i < len(code):
        if code.startswith("//", i):
            end = code.find("\n", i)
            end = len(code) if end < 0 else end
            out[i:end] = " " * (end - i)
        elif code.startswith("/*", i):
            close = code.find("*/", i + 2)
            if close < 0:
                return None
            end = close + 2
            out[i:end] = ["\n" if c == "\n" else " " for c in code[i:end]]
        elif code[i] in _QUOTES:
            end = _template_end(code, i) if code[i] == "`" else _string_end(code, i)
            if end is None:
                return None
            out[i + 1:end - 1] = "x" * (end - i - 2)
        else:
            end = i + 1
        i = end
    return "".join(out)


def _match_back(m: str, p: int) -> int | None:
    stack: list[str] = []
    for k in range(p, -1, -1):
        c = m[k]
        if c in _OPENER:
            stack.append(_OPENER[c])
        elif c in _CLOSER:
            if not stack or stack.pop() != c:
                return None
            if not stack:
                return k
    return None


def _match_forward(m: str, p: int) -> int | None:
    stack: list[str] = []
    for k in range(p, len(m)):
        c = m[k]
        if c in _CLOSER:
            stack.append(_CLOSER[c])
        elif c in _OPENER:
            if not stack or stack.pop() != c:
                return None
            if not stack:
                return k
    return None


def _exponent_start(m: str, k: int) -> int:
    if k < 2 or m[k - 1] not in "+-" or m[k - 2] not in "eE" or not m[k].isdigit():
        return k
    r = k - 2
    while r > 0 and m[r - 1] in _NUMBER_BODY:
        r -= 1
    if r == k - 2 or not (m[r].isdigit() or (m[r] == "." and m[r + 1].isdigit())):
        return k
    return k if r > 0 and m[r - 1] in _IDENT else r


def _left_start(m: str, end: int) -> int | None:
    j = end
    while j > 0:
        c = m[j - 1]
        if c in ")]":
            k = _match_back(m, j - 1)
            if k is None:
                return None
            j = k
            if m[max(0, j - 2):j] == "?.":
                j -= 2
                continue
            if j > 0 and (m[j - 1] in _IDENT or m[j - 1] in _CHAINS):
                continue
            return j
        if c in _QUOTES:
            k = m.rfind(c, 0, j - 1)
            return None if k < 0 else k
        if c not in _IDENT:
            return None
        k = j
        while k > 0 and m[k - 1] in _IDENT:
            k -= 1
        k = _exponent_start(m, k)
        if k == 0 or m[k - 1] != ".":
            return k
        dot = k - 1
        if m[k].isdigit():
            while dot > 0 and m[dot - 1].isdigit():
                dot -= 1
            return None if dot > 0 and m[dot - 1] in _IDENT else dot
        start = dot - 1 if dot > 0 and m[dot - 1] == "?" else dot
        if start > 0 and (m[start - 1] in _IDENT or m[start - 1] in _CHAINS):
            j = start
            continue
        return None
    return None


def _skip_space(m: str, j: int) -> int:
    while j < len(m) and m[j] in _SPACE:
        j += 1
    return j


def _name_end(m: str, j: int) -> int:
    while j < len(m) and m[j] in _IDENT:
        j += 1
    return j


def _primary_end(m: str, j: int) -> int | None:
    if j >= len(m):
        return None
    c = m[j]
    if c in "([":
        k = _match_forward(m, j)
        return None if k is None else k + 1
    if c in _QUOTES:
        k = m.find(c, j + 1)
        return None if k < 0 else k + 1
    if c.isdigit() or (c == "." and m[j + 1:j + 2].isdigit()):
        prefixed = c == "0" and m[j + 1:j + 2] != "" and m[j + 1] in "xXbBoO"
        k = j
        while k < len(m) and (m[k] in _IDENT or m[k] == "."):
            k += 1
            if not prefixed and m[k - 1] in "eE" and m[k:k + 1] in ("+", "-") and m[k + 1:k + 2].isdigit():
                k += 1
        return k
    if c in _IDENT:
        return _name_end(m, j)
    return None


def _starts_name(c: str) -> bool:
    return c != "" and (c.isalpha() or c in "_$")


def _postfix_end(m: str, j: int) -> int | None:
    while j < len(m):
        if m[j] == "." and _starts_name(m[j + 1:j + 2]):
            j = _name_end(m, j + 1)
        elif m.startswith("?.", j) and not m[j + 2:j + 3].isdigit():
            after = m[j + 2:j + 3]
            if after in ("(", "["):
                k = _match_forward(m, j + 2)
                if k is None:
                    return None
                j = k + 1
            elif _starts_name(after):
                j = _name_end(m, j + 2)
            else:
                return j
        elif m[j] in "([":
            k = _match_forward(m, j)
            if k is None:
                return None
            j = k + 1
        else:
            return j
    return j


def _right_end(m: str, j: int) -> int | None:
    while True:
        j = _skip_space(m, j)
        while j < len(m):
            if m[j] in "-+!~":
                j = _skip_space(m, j + 1)
                continue
            word_end = _name_end(m, j)
            if m[j:word_end] in _UNARY_WORDS and word_end < len(m):
                j = _skip_space(m, word_end)
                continue
            break
        end = _primary_end(m, j)
        end = None if end is None else _postfix_end(m, end)
        if end is None:
            return None
        k = _skip_space(m, end)
        if m.startswith("**", k) and m[k + 2:k + 3] not in ("=", "*"):
            j = k + 2
            continue
        return end


def _binary(m: str, p: int) -> bool:
    t = p
    while t > 0 and m[t - 1] in _SPACE:
        t -= 1
    if t == 0:
        return False
    c = m[t - 1]
    if c in ")]}" or c in _QUOTES:
        return True
    if c not in _IDENT:
        return False
    w = t
    while w > 0 and m[w - 1] in _IDENT:
        w -= 1
    return m[w:t] not in KEYWORDS


def _span(m: str, at: int) -> tuple[int, int] | None:
    end = at
    while end > 0 and m[end - 1] in _SPACE:
        end -= 1
    start = _left_start(m, end)
    if start is None:
        return None
    k = start
    while k > 0 and m[k - 1] in _SPACE:
        k -= 1
    if k == 0 or m[k - 1] not in "-+!~":
        return None
    sign = m[k - 1]
    if sign in "-+" and (m[k - 2:k - 1] == sign or _binary(m, k - 1)):
        return None
    close = _right_end(m, at + 2)
    return None if close is None else (start, close)


def repaired(code: str) -> str:
    if "**" not in code:
        return code
    m = _masked(code)
    if m is None or "**" not in m:
        return code
    opens: dict[int, int] = {}
    closes: dict[int, int] = {}
    at = m.find("**")
    while at >= 0:
        run = at
        while run < len(m) and m[run] == "*":
            run += 1
        if run - at == 2 and m[run:run + 1] != "=":
            span = _span(m, at)
            if span is not None:
                opens[span[0]] = opens.get(span[0], 0) + 1
                closes[span[1]] = closes.get(span[1], 0) + 1
        at = m.find("**", run)
    if not opens:
        return code
    pieces, last = [], 0
    for place in sorted({*opens, *closes}):
        pieces += [code[last:place], ")" * closes.get(place, 0) + "(" * opens.get(place, 0)]
        last = place
    pieces.append(code[last:])
    return "".join(pieces)
