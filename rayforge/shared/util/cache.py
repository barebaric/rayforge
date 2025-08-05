from functools import lru_cache, wraps
from typing import Callable, Any


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
            force = kwargs.pop('force', False)
            if force:
                # If force is True, bypass the cache and call the original
                # function
                return func(*args, **kwargs)
            else:
                # Otherwise, use the cached version
                return cached_func(*args, **kwargs)

        return wrapper

    return decorator
