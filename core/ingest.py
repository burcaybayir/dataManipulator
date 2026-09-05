"""Loading, profiling and validating user data (FR-1).

The job here is to turn whatever the user uploaded into a :class:`Dataset`:
a DataFrame plus a per-column profile, a list of validation issues worth
showing them, and a content hash so an analysis can be tied back to the exact
bytes it came from.

Inference is advisory. Every inferred type can be overridden by the user via
:meth:`Dataset.retype`, because a column of zip codes is not a number no
matter what pandas thinks.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import re
import warnings
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import pandas as pd

from core.spec import Aggregation, ChartFamily, ChartSpec

MAX_BYTES: Final = 50 * 1024 * 1024
MAX_ROWS: Final = 1_000_000
DEFAULT_SAMPLE_SEED: Final = 20240101

#: Share of non-null values that must parse for a column to take that type.
_COERCION_THRESHOLD: Final = 0.9

#: A column pandas renamed to dodge a duplicate header, e.g. ``region.1``.
_MANGLED_HEADER: Final = re.compile(r"^(?P<base>.+)\.(?P<n>\d+)$")

_DELIMITERS: Final = {".csv": ",", ".tsv": "\t", ".tab": "\t", ".txt": "\t"}
_EXCEL_SUFFIXES: Final = frozenset({".xlsx", ".xlsm", ".xls"})


class IngestError(Exception):
    """Raised when input cannot be loaded at all."""


class ColumnType(StrEnum):
    """The role a column can play in a chart."""

    NUMERIC = "numeric"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


class IssueLevel(StrEnum):
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """Something the user should know about their data before charting it."""

    code: str
    level: IssueLevel
    message: str
    column: str | None = None


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    """What we know about one column."""

    name: str
    type: ColumnType
    null_count: int
    unique_count: int
    row_count: int
    mixed_types: bool = False
    inferred: bool = True

    @property
    def null_fraction(self) -> float:
        return self.null_count / self.row_count if self.row_count else 0.0

    @property
    def is_all_null(self) -> bool:
        return self.row_count > 0 and self.null_count == self.row_count

    @property
    def is_constant(self) -> bool:
        return self.unique_count <= 1 and not self.is_all_null


@dataclass(frozen=True, slots=True)
class Dataset:
    """A loaded, profiled table.

    Attributes:
        frame: The data itself, with inferred types already applied.
        profiles: One profile per column, in column order.
        issues: Validation findings to surface in the UI.
        source_hash: SHA-256 of the original bytes, for the manifest (FR-8).
        source_name: Original filename, or a label for pasted input.
        original_row_count: Rows before any sampling.
        sample_seed: Seed used if the data was sampled, else ``None``.
    """

    frame: pd.DataFrame
    profiles: tuple[ColumnProfile, ...]
    issues: tuple[ValidationIssue, ...]
    source_hash: str
    source_name: str
    original_row_count: int
    sample_seed: int | None = None

    @property
    def was_sampled(self) -> bool:
        return len(self.frame) < self.original_row_count

    @property
    def types(self) -> dict[str, ColumnType]:
        return {p.name: p.type for p in self.profiles}

    def profile(self, column: str) -> ColumnProfile:
        for p in self.profiles:
            if p.name == column:
                return p
        raise KeyError(column)

    def columns_of(self, *types: ColumnType) -> tuple[str, ...]:
        """Column names matching any of ``types``, excluding all-null columns."""
        wanted = set(types)
        return tuple(p.name for p in self.profiles if p.type in wanted and not p.is_all_null)

    def retype(self, column: str, new_type: ColumnType) -> Dataset:
        """Return a copy with ``column`` re-coerced to ``new_type`` (FR-1).

        Values that cannot be coerced become null, and the refreshed profile
        records that the type was chosen rather than inferred.
        """
        if column not in self.frame.columns:
            raise KeyError(column)
        frame = self.frame.copy()
        frame[column] = _coerce(frame[column], new_type)
        profile = _profile_column(frame[column], forced_type=new_type)
        profiles = tuple(profile if p.name == column else p for p in self.profiles)
        return replace(self, frame=frame, profiles=profiles)


def load(
    source: str | Path | bytes | io.BytesIO,
    *,
    name: str | None = None,
    max_rows: int = MAX_ROWS,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
) -> Dataset:
    """Load a CSV, TSV or Excel file from a path or raw bytes.

    Args:
        source: Filesystem path, or the file's bytes (as from an upload).
        name: Filename used to pick a parser when ``source`` is bytes.
        max_rows: Rows above which the frame is seeded-sampled down.
        sample_seed: Seed for that sampling, so results stay reproducible.

    Raises:
        IngestError: The input is empty, too large, or unparseable.
    """
    raw, source_name = _read_bytes(source, name)
    if not raw:
        raise IngestError(f"{source_name} is empty")
    if len(raw) > MAX_BYTES:
        raise IngestError(
            f"{source_name} is {len(raw) / 1024 / 1024:.1f} MB, over the "
            f"{MAX_BYTES // 1024 // 1024} MB limit"
        )

    suffix = Path(source_name).suffix.lower()
    if suffix in _EXCEL_SUFFIXES:
        frame = _read_excel(raw, source_name)
    else:
        frame = _read_delimited(raw, source_name, _DELIMITERS.get(suffix))
    return _build(frame, raw, source_name, max_rows=max_rows, sample_seed=sample_seed)


def load_table_text(
    text: str,
    *,
    name: str = "pasted table",
    max_rows: int = MAX_ROWS,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
) -> Dataset:
    """Load a table the user pasted in, sniffing the delimiter (FR-1)."""
    if not text.strip():
        raise IngestError("nothing was pasted")
    raw = text.encode("utf-8")
    frame = _read_delimited(raw, name, None)
    return _build(frame, raw, name, max_rows=max_rows, sample_seed=sample_seed)


def load_frame(
    frame: pd.DataFrame,
    *,
    name: str = "dataframe",
    max_rows: int = MAX_ROWS,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
) -> Dataset:
    """Profile an in-memory DataFrame. Used by tests and the sample loader."""
    raw = frame.to_csv(index=False).encode("utf-8")
    return _build(frame.copy(), raw, name, max_rows=max_rows, sample_seed=sample_seed)


def suggest_spec(dataset: Dataset, family: ChartFamily | None = None) -> ChartSpec:
    """Propose an honest starting chart for ``dataset``.

    Prefers a datetime x axis over a categorical one, and the numeric column
    with the most distinct values for y — the one most likely to be the
    measure rather than an id or a flag.

    Raises:
        IngestError: No pair of columns can make a chart.
    """
    datetimes = dataset.columns_of(ColumnType.DATETIME)
    categoricals = dataset.columns_of(ColumnType.CATEGORICAL, ColumnType.BOOLEAN)
    numerics = dataset.columns_of(ColumnType.NUMERIC)

    x = next(iter(datetimes + categoricals + numerics), None)
    if x is None:
        raise IngestError("no usable columns found")

    measures = [c for c in numerics if c != x]
    y = max(measures, key=lambda c: dataset.profile(c).unique_count, default=None)

    if family is None:
        family = ChartFamily.LINE if x in datetimes else ChartFamily.BAR

    series = next((c for c in categoricals if c not in {x, y} and _splits_well(dataset, c)), None)
    aggregation = Aggregation.SUM if y is not None else Aggregation.COUNT
    return ChartSpec(family=family, x=x, y=y, series=series, aggregation=aggregation)


def _splits_well(dataset: Dataset, column: str) -> bool:
    """Whether a column has few enough values to be a readable series split."""
    return 2 <= dataset.profile(column).unique_count <= 8


def _read_bytes(source: str | Path | bytes | io.BytesIO, name: str | None) -> tuple[bytes, str]:
    if isinstance(source, str | Path):
        path = Path(source)
        if not path.is_file():
            raise IngestError(f"no such file: {path}")
        return path.read_bytes(), name or path.name
    raw = source.getvalue() if isinstance(source, io.BytesIO) else source
    if name is None:
        raise IngestError("a filename is required when loading from bytes")
    return raw, name


def _read_delimited(raw: bytes, name: str, delimiter: str | None) -> pd.DataFrame:
    """Parse delimited text, trying each candidate separator when unknown.

    We try a fixed candidate list rather than pandas' ``sep=None`` sniffer,
    which is free to pick any character and will happily split ``value`` on
    its own ``v``. The separator producing the most columns wins; ties go to
    the earlier candidate, so a genuinely one-column table still loads.
    """
    candidates = [delimiter] if delimiter is not None else [",", "\t", ";", "|"]

    best: pd.DataFrame | None = None
    last_error: Exception | None = None
    for sep in candidates:
        try:
            frame = pd.read_csv(io.BytesIO(raw), sep=sep, skipinitialspace=True)
        except Exception as exc:  # try the next separator
            last_error = exc
            continue
        if delimiter is not None:
            return frame
        if best is None or frame.shape[1] > best.shape[1]:
            best = frame
    if best is not None:
        return best
    raise IngestError(f"could not parse {name}: {last_error}")


def _read_excel(raw: bytes, name: str) -> pd.DataFrame:
    try:
        return pd.read_excel(io.BytesIO(raw), sheet_name=0)
    except Exception as exc:  # surfaced to the user as-is
        raise IngestError(f"could not read {name}: {exc}") from exc


def _build(
    frame: pd.DataFrame,
    raw: bytes,
    name: str,
    *,
    max_rows: int,
    sample_seed: int,
) -> Dataset:
    if frame.empty or frame.shape[1] == 0:
        raise IngestError(f"{name} has no rows")

    issues = list(_header_issues(frame))
    frame = _dedupe_headers(frame)

    original_rows = len(frame)
    used_seed: int | None = None
    if original_rows > max_rows:
        frame = frame.sample(n=max_rows, random_state=sample_seed).sort_index()
        used_seed = sample_seed
        issues.append(
            ValidationIssue(
                code="sampled",
                level=IssueLevel.WARNING,
                message=(
                    f"{original_rows:,} rows exceeded the {max_rows:,}-row cap; "
                    f"showing a seeded random sample of {max_rows:,}. "
                    "Charts describe the sample, not the full dataset."
                ),
            )
        )

    profiles: list[ColumnProfile] = []
    for column in frame.columns:
        frame[column] = _infer_and_coerce(frame[column])
        profiles.append(_profile_column(frame[column]))

    issues.extend(_column_issues(profiles))
    return Dataset(
        frame=frame.reset_index(drop=True),
        profiles=tuple(profiles),
        issues=tuple(issues),
        source_hash=hashlib.sha256(raw).hexdigest(),
        source_name=name,
        original_row_count=original_rows,
        sample_seed=used_seed,
    )


def _header_issues(frame: pd.DataFrame) -> list[ValidationIssue]:
    """Flag duplicate headers, including ones pandas silently renamed.

    ``read_csv`` turns a repeated ``region`` column into ``region.1``, so the
    duplication is invisible by the time we see the frame. We detect both the
    mangled form and true duplicates from non-CSV sources.
    """
    issues: list[ValidationIssue] = []
    names = [str(c) for c in frame.columns]
    seen = set(names)

    for name in names:
        match = _MANGLED_HEADER.match(name)
        if match and match.group("base") in seen:
            issues.append(
                ValidationIssue(
                    code="duplicate_header",
                    level=IssueLevel.WARNING,
                    message=(
                        f"Column {match.group('base')!r} appeared more than once and was "
                        f"renamed to {name!r}. Check you are charting the one you mean."
                    ),
                    column=name,
                )
            )

    for name in {n for n in names if names.count(n) > 1}:
        issues.append(
            ValidationIssue(
                code="duplicate_header",
                level=IssueLevel.WARNING,
                message=f"Column {name!r} appears more than once; later copies were suffixed.",
                column=name,
            )
        )
    return issues


def _dedupe_headers(frame: pd.DataFrame) -> pd.DataFrame:
    """Make column names unique so downstream lookups stay unambiguous."""
    if not frame.columns.duplicated().any():
        return frame
    counts: dict[str, int] = {}
    renamed: list[str] = []
    for column in (str(c) for c in frame.columns):
        counts[column] = counts.get(column, 0) + 1
        renamed.append(column if counts[column] == 1 else f"{column}.{counts[column] - 1}")
    frame = frame.copy()
    frame.columns = pd.Index(renamed)
    return frame


def _column_issues(profiles: list[ColumnProfile]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for p in profiles:
        if p.is_all_null:
            issues.append(
                ValidationIssue(
                    code="all_null",
                    level=IssueLevel.WARNING,
                    message=f"Column {p.name!r} is entirely empty and cannot be charted.",
                    column=p.name,
                )
            )
        elif p.mixed_types:
            issues.append(
                ValidationIssue(
                    code="mixed_types",
                    level=IssueLevel.WARNING,
                    message=(
                        f"Column {p.name!r} mixes numbers and text, so it was kept as "
                        "text. Override the type if it should be numeric."
                    ),
                    column=p.name,
                )
            )
        elif p.null_fraction > 0.2:
            issues.append(
                ValidationIssue(
                    code="sparse",
                    level=IssueLevel.WARNING,
                    message=f"Column {p.name!r} is {p.null_fraction:.0%} empty.",
                    column=p.name,
                )
            )
    return issues


def _infer_and_coerce(series: pd.Series) -> pd.Series:
    """Convert an object column to its most specific plausible type."""
    if not _is_untyped(series):
        return series
    non_null = series.dropna()
    if non_null.empty:
        return series

    numeric = pd.to_numeric(non_null, errors="coerce")
    if numeric.notna().mean() >= _COERCION_THRESHOLD:
        return pd.to_numeric(series, errors="coerce")

    parsed = _try_datetime(non_null)
    if parsed is not None and parsed.notna().mean() >= _COERCION_THRESHOLD:
        converted = _try_datetime(series)
        if converted is not None:
            return converted
    return series


def _try_datetime(series: pd.Series) -> pd.Series | None:
    """Parse dates without letting pandas' format warnings reach the user."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fmt in ("mixed", None):
            with contextlib.suppress(Exception):
                parsed: pd.Series = pd.to_datetime(series, errors="coerce", format=fmt)
                return parsed
    return None


def _is_untyped(series: pd.Series) -> bool:
    """Whether a column still holds unparsed text.

    pandas 2 gives these ``object`` dtype and pandas 3 gives them the new
    string dtype, so both count as "we have not worked out what this is yet".
    """
    return series.dtype == object or pd.api.types.is_string_dtype(series)


def _profile_column(series: pd.Series, forced_type: ColumnType | None = None) -> ColumnProfile:
    column_type = forced_type or _classify(series)
    return ColumnProfile(
        name=str(series.name),
        type=column_type,
        null_count=int(series.isna().sum()),
        unique_count=int(series.nunique(dropna=True)),
        row_count=len(series),
        mixed_types=_has_mixed_types(series),
        inferred=forced_type is None,
    )


def _classify(series: pd.Series) -> ColumnType:
    if pd.api.types.is_bool_dtype(series):
        return ColumnType.BOOLEAN
    if pd.api.types.is_numeric_dtype(series):
        return ColumnType.NUMERIC
    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnType.DATETIME
    return ColumnType.CATEGORICAL


def _has_mixed_types(series: pd.Series) -> bool:
    """True when an object column holds more than one kind of scalar.

    Only meaningful for object columns; a typed column is by definition
    homogeneous.
    """
    if not _is_untyped(series):
        return False
    kinds = {_kind(v) for v in series.dropna()}
    return len(kinds) > 1


def _kind(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "number" if _looks_numeric(value) else "text"
    return type(value).__name__


def _looks_numeric(value: str) -> bool:
    try:
        float(value.replace(",", "").strip())
    except ValueError:
        return False
    return True


def _coerce(series: pd.Series, target: ColumnType) -> pd.Series:
    """Force a column to ``target``, nulling values that will not convert."""
    match target:
        case ColumnType.NUMERIC:
            return pd.to_numeric(series, errors="coerce")
        case ColumnType.DATETIME:
            parsed = _try_datetime(series)
            if parsed is None:
                raise IngestError(f"{series.name!r} cannot be read as dates")
            return parsed
        case ColumnType.BOOLEAN:
            return series.astype("boolean")
        case ColumnType.CATEGORICAL:
            return series.astype("string").astype(object)
    raise IngestError(f"unsupported target type: {target}")
