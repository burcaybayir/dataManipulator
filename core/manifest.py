"""The reproducibility manifest (FR-8).

A manifest is everything needed to redraw a chart apart from the data itself:
the hash of the source bytes, the chart spec, and the transform stack. Two
runs of the same manifest against the same file produce the same chart, which
is what makes an exported variant auditable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core.ingest import Dataset
from core.prepare import PreparedChart, prepare
from core.spec import ChartSpec
from core.transforms.stack import TransformStack

FORMAT_VERSION = 1


class ManifestError(Exception):
    """Raised when a manifest cannot be read or does not match its data."""


@dataclass(frozen=True, slots=True)
class Manifest:
    """A reproducible description of one chart variant.

    Attributes:
        spec: The chart specification before transforms are applied.
        stack: The transforms that turn the baseline into this variant.
        source_hash: SHA-256 of the bytes the analysis was built on.
        source_name: Original filename, for display.
    """

    spec: ChartSpec
    stack: TransformStack = field(default_factory=TransformStack)
    source_hash: str = ""
    source_name: str = ""

    @classmethod
    def for_dataset(
        cls, dataset: Dataset, spec: ChartSpec, stack: TransformStack | None = None
    ) -> Manifest:
        """Build a manifest tying ``spec`` and ``stack`` to a loaded dataset."""
        return cls(
            spec=spec,
            stack=stack or TransformStack(),
            source_hash=dataset.source_hash,
            source_name=dataset.source_name,
        )

    def render(self, frame: pd.DataFrame) -> PreparedChart:
        """Apply the stack to ``frame`` and materialize the plotted values."""
        transformed, spec = self.stack.apply(frame, self.spec)
        return prepare(transformed, spec)

    def render_dataset(self, dataset: Dataset, *, check_hash: bool = True) -> PreparedChart:
        """Replay this manifest against a dataset.

        Args:
            dataset: The data to replay against.
            check_hash: Refuse to render if the dataset is not the one the
                manifest was built from.

        Raises:
            ManifestError: The hashes disagree and ``check_hash`` is set.
        """
        if check_hash and self.source_hash and dataset.source_hash != self.source_hash:
            raise ManifestError(
                f"manifest was built from {self.source_name!r} "
                f"({self.source_hash[:12]}…), but this data hashes to "
                f"{dataset.source_hash[:12]}…"
            )
        return self.render(dataset.frame)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "source_hash": self.source_hash,
            "source_name": self.source_name,
            "spec": self.spec.to_dict(),
            "transforms": self.stack.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Manifest:
        """Rebuild a manifest produced by :meth:`to_dict`.

        Raises:
            ManifestError: The payload is malformed or from a future version.
        """
        version = raw.get("format_version")
        if version != FORMAT_VERSION:
            raise ManifestError(f"unsupported manifest version: {version!r}")
        try:
            return cls(
                spec=ChartSpec.from_dict(raw["spec"]),
                stack=TransformStack.from_dict(raw.get("transforms", [])),
                source_hash=str(raw.get("source_hash", "")),
                source_name=str(raw.get("source_name", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError(f"malformed manifest: {exc}") from exc

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> Manifest:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"not valid JSON: {exc}") from exc
        return cls.from_dict(parsed)
