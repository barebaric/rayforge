import contextlib
import os
import time
from typing import Iterator


@contextlib.contextmanager
def profile_if_enabled(name: str, generation_id: int) -> Iterator[None]:
    profile_dir = os.environ.get("RAYFORGE_PROFILE_DIR")
    if not profile_dir:
        yield
        return

    import cProfile

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        yield
    finally:
        profiler.disable()
        timestamp = time.time_ns()
        filename = f"{name}_{generation_id}_{timestamp}.prof"
        filepath = os.path.join(profile_dir, filename)
        os.makedirs(profile_dir, exist_ok=True)
        profiler.dump_stats(filepath)
