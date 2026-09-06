"""An ordered, replayable sequence of transforms."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from core.spec import ChartSpec
from core.transforms.base import Transform, TransformError
from core.transforms.registry import from_dict as transform_from_dict

#: Beyond this many stacked transforms the impact scores stop being
#: interpretable, because no single technique explains the difference.
MAX_DEPTH = 5


@dataclass(frozen=True, slots=True)
class TransformStack:
    """A sequence of transforms applied in order.

    The stack is what a variant *is*: the baseline is an empty stack, and every
    chart in the gallery is one stack applied to the same source data.
    """

    transforms: tuple[Transform, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "transforms", tuple(self.transforms))
        if len(self.transforms) > MAX_DEPTH:
            raise TransformError(
                f"a stack of {len(self.transforms)} transforms exceeds the depth limit "
                f"of {MAX_DEPTH}; impact scores stop being attributable beyond it"
            )

    def __len__(self) -> int:
        return len(self.transforms)

    def __iter__(self) -> Iterator[Transform]:
        return iter(self.transforms)

    def __bool__(self) -> bool:
        return bool(self.transforms)

    @property
    def is_empty(self) -> bool:
        """True for the baseline — a stack that changes nothing."""
        return not self.transforms

    @property
    def names(self) -> tuple[str, ...]:
        """Registry names of the transforms, in application order."""
        return tuple(t.name for t in self.transforms)

    @property
    def modifies_data(self) -> bool:
        """Whether any transform in the stack changes rows rather than the spec."""
        return any(t.modifies_data for t in self.transforms)

    def push(self, *transforms: Transform) -> TransformStack:
        """Return a new stack with ``transforms`` appended."""
        return TransformStack((*self.transforms, *transforms))

    def apply(self, frame: pd.DataFrame, spec: ChartSpec) -> tuple[pd.DataFrame, ChartSpec]:
        """Apply every transform in order.

        Raises:
            TransformError: A transform could not apply, tagged with its
                position in the stack so the user can see which one failed.
        """
        for position, transform in enumerate(self.transforms):
            try:
                frame, spec = transform.apply(frame, spec)
            except TransformError as exc:
                raise TransformError(f"step {position + 1} ({transform.name}): {exc}") from exc
        return frame, spec

    def describe(self) -> tuple[str, ...]:
        """One line per transform, for captions and export footers."""
        return tuple(t.describe() for t in self.transforms)

    def to_dict(self) -> list[dict[str, Any]]:
        """Serialize for the reproducibility manifest (FR-8)."""
        return [t.to_dict() for t in self.transforms]

    @classmethod
    def from_dict(cls, raw: Sequence[dict[str, Any]]) -> TransformStack:
        """Rebuild a stack produced by :meth:`to_dict`."""
        return cls(tuple(transform_from_dict(entry) for entry in raw))
