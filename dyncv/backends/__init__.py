"""Compute backends. get("kaggle", ...) returns an object with the methods of base.Backend."""


def get(name, **kw):
    if name == "kaggle":
        from .kaggle import KaggleBackend

        return KaggleBackend(**kw)
    raise ValueError(f"unknown backend {name!r} (add dyncv/backends/{name}.py implementing base.Backend)")
