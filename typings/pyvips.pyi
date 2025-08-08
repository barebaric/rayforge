from typing import Any


# Ignore all type stubs from pyvips, as they are incorrect and
# produce tons of incorrect linter warnings.
def __getattr__(name: str) -> Any: ...
