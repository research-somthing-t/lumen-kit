from __future__ import annotations

from typing import Protocol

from witness_client import RenderOptions

FRONT_THETAS = (30.0, 330.0, 0.0, 0.0)
FRONT_PHIS = (0.0, 0.0, 15.0, -30.0)


class Renderer(Protocol):
    def render(self, source: str, options: RenderOptions) -> bytes: ...
