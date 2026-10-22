from __future__ import annotations

import json
import threading
from pathlib import Path

PRICING = json.loads(Path(__file__).with_name("pricing.json").read_text())
CREDIT_LIMIT = PRICING["credits_per_prompt"]
CALL_LIMIT = PRICING["calls_per_prompt"]
LONG_CONTEXT = PRICING["long_context_above_input_tokens"]
SAFETY = 0.06
TOKENS_PER_IMAGE = 3000
CHARS_PER_TOKEN = 3.0


def call_credits(model: str, input_tokens: int, output_tokens: int) -> float:
    weights = PRICING["models"][model]
    long = input_tokens > LONG_CONTEXT
    return (input_tokens * weights["long_input_credits_per_token" if long else "input_credits_per_token"]
            + output_tokens * weights["long_output_credits_per_token" if long else "output_credits_per_token"])


RESERVE_OUTPUT = {("gpt-6-astra", 22000): 18500, ("gpt-6.1-sol", 44000): 36000, ("gpt-6.1-sol", 32768): 24000,
                  ("gpt-6.1-sol", 8192): 2048}


def worst_case(model: str, text_chars: int, images: int, max_output_tokens: int) -> float:
    input_tokens = int(text_chars / CHARS_PER_TOKEN) + images * TOKENS_PER_IMAGE
    return call_credits(model, input_tokens, RESERVE_OUTPUT.get((model, max_output_tokens), max_output_tokens))


class Budget:

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.credits = 0.0
        self._reserved_calls = 0
        self._reserved_credits = 0.0

    def allows(self, estimates: list[float], keep_calls: int = 0, keep_credits: float = 0.0) -> bool:
        with self._lock:
            calls = self.calls + self._reserved_calls + len(estimates) + keep_calls
            credits = self.credits + self._reserved_credits + sum(estimates) + keep_credits
            return calls <= CALL_LIMIT and credits <= CREDIT_LIMIT * (1 - SAFETY)

    def reserve(self, estimate: float) -> bool:
        with self._lock:
            if (self.calls + self._reserved_calls + 1 > CALL_LIMIT
                    or self.credits + self._reserved_credits + estimate > CREDIT_LIMIT * (1 - SAFETY)):
                return False
            self._reserved_calls += 1
            self._reserved_credits += estimate
            return True

    def settle(self, estimate: float, model: str, usage: object | None) -> None:
        if usage is not None:
            charged = call_credits(model, getattr(usage, "input_tokens", 0), getattr(usage, "output_tokens", 0))
        else:
            charged = estimate
        with self._lock:
            self._reserved_calls -= 1
            self._reserved_credits -= estimate
            self.calls += 1
            self.credits += charged
