import functools


def once_per_object(func):
    seen = set()

    @functools.wraps(func)
    def wrapper(obj, *args, **kwargs):
        if isinstance(obj, str):
            key = obj
        else:
            key = id(obj)
        if key in seen:
            return
        seen.add(key)
        return func(obj, *args, **kwargs)

    return wrapper
