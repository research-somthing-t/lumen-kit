from __future__ import annotations

import json
import threading
from pathlib import Path

PRICING = json.loads(Path(__file__).with_name("pricing.json").read_text())
CREDIT_LIMIT = PRICING["credits_per_prompt"]
CALL_LIMIT = PRICING["calls_per_prompt"]
HARD_CALL_LIMIT = min(CALL_LIMIT, 57)
HARD_CREDIT_LIMIT = min(CREDIT_LIMIT, 4_850_000)
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
                  ("gpt-6.1-sol", 8192): 2048, ("gpt-6.2-sol", 44000): 36000, ("gpt-6.2-sol", 32768): 24000,
                  ("gpt-6.2-sol", 8192): 2048}


def worst_case(model: str, text_chars: int, images: int, max_output_tokens: int) -> float:
    input_tokens = int(text_chars / CHARS_PER_TOKEN) + images * TOKENS_PER_IMAGE
    return call_credits(model, input_tokens, RESERVE_OUTPUT.get((model, max_output_tokens), max_output_tokens))


def hard_worst_case(model: str, text_chars: int, images: int, max_output_tokens: int) -> float:
    input_tokens = int(text_chars / CHARS_PER_TOKEN) + images * TOKENS_PER_IMAGE
    return call_credits(model, input_tokens, max_output_tokens)


class Budget:

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.credits = 0.0
        self._reserved_calls = 0
        self._reserved_credits = 0.0
        self._reserved_hard_credits = 0.0

    def allows(self, estimates: list[float], keep_calls: int = 0, keep_credits: float = 0.0,
               hard_estimates: list[float] | None = None) -> bool:
        with self._lock:
            calls = self.calls + self._reserved_calls + len(estimates) + keep_calls
            credits = self.credits + self._reserved_credits + sum(estimates) + keep_credits
            soft_cap = min(CREDIT_LIMIT * (1 - SAFETY), HARD_CREDIT_LIMIT)
            if calls > HARD_CALL_LIMIT or credits > soft_cap:
                return False
            if hard_estimates is not None:
                hard_total = self.credits + self._reserved_hard_credits + sum(hard_estimates) + keep_credits
                if hard_total > CREDIT_LIMIT:
                    return False
            return True

    def reserve(self, estimate: float, hard_estimate: float | None = None) -> bool:
        with self._lock:
            soft_cap = min(CREDIT_LIMIT * (1 - SAFETY), HARD_CREDIT_LIMIT)
            hard = estimate if hard_estimate is None else hard_estimate
            if (self.calls + self._reserved_calls + 1 > HARD_CALL_LIMIT
                    or self.credits + self._reserved_credits + estimate > soft_cap
                    or self.credits + self._reserved_hard_credits + hard > CREDIT_LIMIT):
                return False
            self._reserved_calls += 1
            self._reserved_credits += estimate
            self._reserved_hard_credits += hard
            return True

    def settle(self, estimate: float, model: str, usage: object | None,
               hard_estimate: float | None = None) -> None:
        if usage is not None:
            charged = call_credits(model, getattr(usage, "input_tokens", 0), getattr(usage, "output_tokens", 0))
        else:
            charged = estimate
        hard = estimate if hard_estimate is None else hard_estimate
        with self._lock:
            self._reserved_calls -= 1
            self._reserved_credits -= estimate
            self._reserved_hard_credits -= hard
            self.calls += 1
            self.credits += charged
