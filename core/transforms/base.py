"""The transform contract.

A transform is a named, parameterized, pure function over ``(DataFrame,
ChartSpec)``. Most only rewrite the spec — the distortion is in how the chart
is drawn, not in the numbers — but a few (regrouping bins, dropping outliers)
genuinely change the rows, and say so via :attr:`Transform.modifies_data`.

Every transform carries its own documentation: what it does, why it misleads,
and when it is legitimate. Most of these techniques are legitimate somewhere;
that is precisely why they work as deceptions.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar

import pandas as pd

from core.prepare import PreparedChart, PrepareError, prepare
from core.spec import ChartSpec


class TransformError(Exception):
    """Raised when a transform cannot apply to the data or spec it was given."""


class Transform(ABC):
    """Base class for every distortion technique.

    Subclasses are frozen dataclasses holding their parameters, so a transform
    is comparable, hashable, and serializable by construction.
    """

    #: Registry key, used in manifests. Stable across versions.
    name: ClassVar[str]
    #: Human-readable name for the UI.
    label: ClassVar[str]
    #: What the transform does, mechanically.
    explanation: ClassVar[str]
    #: How it changes the conclusion a reader draws.
    why_misleading: ClassVar[str]
    #: The case where an analyst would reach for it in good faith.
    legitimate_when: ClassVar[str]
    #: True when the transform changes rows rather than only the spec.
    modifies_data: ClassVar[bool] = False

    @abstractmethod
    def apply(self, frame: pd.DataFrame, spec: ChartSpec) -> tuple[pd.DataFrame, ChartSpec]:
        """Return transformed data and spec, leaving the inputs untouched."""

    @property
    def params(self) -> dict[str, Any]:
        """This transform's parameters, JSON-ready."""
        fields: tuple[dataclasses.Field[Any], ...] = (
            dataclasses.fields(self) if dataclasses.is_dataclass(self) else ()
        )
        return {f.name: _jsonable(getattr(self, f.name)) for f in fields}

    def describe(self) -> str:
        """One line naming the technique and the parameters it used."""
        if not self.params:
            return self.label
        rendered = ", ".join(f"{k}={v!r}" for k, v in self.params.items() if v is not None)
        return f"{self.label} ({rendered})" if rendered else self.label

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the reproducibility manifest (FR-8)."""
        return {"transform": self.name, "params": self.params}

    def note(self) -> str:
        """The provenance note stamped onto the spec and into exports (FR-7)."""
        return self.describe()

    def _stamp(self, spec: ChartSpec, **changes: Any) -> ChartSpec:
        """Apply ``changes`` to ``spec`` and record this transform in its notes."""
        return spec.evolve(**changes, notes=(*spec.notes, self.note()))


def prepared_or_error(frame: pd.DataFrame, spec: ChartSpec) -> PreparedChart:
    """Materialize a spec for a transform that needs to consult the data.

    Transforms report failures as :class:`TransformError` so that a stack can
    say which step failed, so a preparation failure is re-raised as one.
    """
    try:
        return prepare(frame, spec)
    except PrepareError as exc:
        raise TransformError(str(exc)) from exc


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple | list):
        return [_jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value
