import asyncio
import contextvars
import functools
from concurrent.futures import ThreadPoolExecutor

from sn17gen.pipeline import run


class _Bridge:
    def __init__(self, witness, loop):
        self._witness = witness
        self._loop = loop
        self.responses = self

    def _wait(self, call):
        return asyncio.run_coroutine_threadsafe(call, self._loop).result()

    def create(self, *, request_id: int = 0, **params):
        return self._wait(self._witness.responses.create(request_id=request_id, **params))

    def render(self, source, options):
        return self._wait(self._witness.render(source, options))


async def generate(image: bytes, seed: int, witness, renderer=None) -> str | None:
    loop = asyncio.get_running_loop()
    bridge = _Bridge(witness, loop)
    call = functools.partial(contextvars.copy_context().run, run, image, seed, bridge,
                             bridge if renderer is None else renderer)
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return await loop.run_in_executor(pool, call)
    finally:
        pool.shutdown(wait=False)
