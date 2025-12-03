import functools


def once_per_object(func):
    seen = set()

    @functools.wraps(func)
    def wrapper(obj, *args, **kwargs):
        key = id(obj)
        if key in seen:
            return
        seen.add(key)
        return func(obj, *args, **kwargs)

    return wrapper
