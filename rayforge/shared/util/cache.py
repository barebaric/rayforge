import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterator
from functools import lru_cache, wraps
from typing import Any, Generic, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


def lru_cache_unless_forced(maxsize: int = 128):
    """
    Extends functools.lru_cache by a "force" argument that allows to
    force a cache update.
    """

    def decorator(func: Callable) -> Callable:
        cached_func = lru_cache(maxsize=maxsize)(func)

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Check if 'force' is in kwargs and is True
            force = kwargs.pop("force", False)
            if force:
                # If force is True, bypass the cache and call the original
                # function
                return func(*args, **kwargs)
            else:
                # Otherwise, use the cached version
                return cached_func(*args, **kwargs)

        return wrapper

    return decorator


class LruCache(Generic[K, V]):
    """
    A thread-safe mapping that keeps at most ``max_entries`` values and,
    when ``max_bytes`` is set, at most that many bytes as measured by
    ``sizeof``. The least recently used values are evicted first; the
    newest value is always kept, even when it alone exceeds the budget.
    """

    def __init__(
        self,
        max_entries: int,
        max_bytes: int | None = None,
        sizeof: Callable[[V], int] | None = None,
    ):
        self.max_entries = max(1, max_entries)
        self.max_bytes = max_bytes
        self._sizeof = sizeof or (lambda value: 0)
        self._entries: OrderedDict[K, tuple[V, int]] = OrderedDict()
        self._nbytes = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def keys(self) -> Iterator[K]:
        return iter(list(self._entries))

    @property
    def nbytes(self) -> int:
        """The total size of the cached values."""
        return self._nbytes

    def get(self, key: K) -> V | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            return entry[0]

    def put(self, key: K, value: V) -> None:
        size = self._sizeof(value)
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._nbytes -= old[1]
            self._entries[key] = (value, size)
            self._nbytes += size
            self._evict()

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._nbytes = 0

    def _over_budget(self) -> bool:
        if len(self._entries) > self.max_entries:
            return True
        return self.max_bytes is not None and self._nbytes > self.max_bytes

    def _evict(self) -> None:
        while len(self._entries) > 1 and self._over_budget():
            _, (_, size) = self._entries.popitem(last=False)
            self._nbytes -= size
