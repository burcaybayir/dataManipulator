"""The transform registry — the extension point.

Adding a technique means writing one module and decorating its class with
:func:`register`. Nothing in the UI or the scoring code needs to change.
"""

from __future__ import annotations

from typing import Any, TypeVar

from core.transforms.base import Transform, TransformError

_REGISTRY: dict[str, type[Transform]] = {}

T = TypeVar("T", bound=type[Transform])


def register(cls: T) -> T:
    """Class decorator adding a transform to the registry.

    Raises:
        TransformError: The class has no ``name``, or the name is taken.
    """
    name = getattr(cls, "name", None)
    if not name:
        raise TransformError(f"{cls.__name__} must define a name")
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise TransformError(f"transform {name!r} is already registered")
    _REGISTRY[name] = cls
    return cls


def available() -> tuple[type[Transform], ...]:
    """Every registered transform class, in registration order."""
    return tuple(_REGISTRY.values())


def get(name: str) -> type[Transform]:
    """Look up a transform class by registry name.

    Raises:
        TransformError: No transform is registered under that name.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "none"
        raise TransformError(f"unknown transform {name!r}; registered: {known}") from None


def build(name: str, **params: Any) -> Transform:
    """Construct a registered transform from its name and parameters.

    Raises:
        TransformError: The name is unknown or the parameters do not fit.
    """
    cls = get(name)
    try:
        return cls(**params)
    except TypeError as exc:
        raise TransformError(f"bad parameters for {name!r}: {exc}") from exc


def from_dict(raw: dict[str, Any]) -> Transform:
    """Rebuild a transform from its manifest entry."""
    if "transform" not in raw:
        raise TransformError("manifest entry has no 'transform' key")
    return build(str(raw["transform"]), **dict(raw.get("params") or {}))
