from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from core.ingest import (
    ColumnType,
    Dataset,
    IngestError,
    load,
    load_frame,
    load_table_text,
    suggest_spec,
)
from core.spec import Aggregation, ChartFamily


def issue_codes(dataset: Dataset) -> set[str]:
    return {issue.code for issue in dataset.issues}


def test_loads_a_csv_from_bytes(csv_bytes: bytes) -> None:
    dataset = load(csv_bytes, name="revenue.csv")
    assert len(dataset.frame) == 24
    assert dataset.source_name == "revenue.csv"
    assert len(dataset.source_hash) == 64


def test_loads_a_csv_from_a_path(tmp_path: Path, csv_bytes: bytes) -> None:
    path = tmp_path / "revenue.csv"
    path.write_bytes(csv_bytes)
    assert len(load(path).frame) == 24


def test_infers_column_types(csv_bytes: bytes) -> None:
    types = load(csv_bytes, name="revenue.csv").types
    assert types == {
        "month": ColumnType.DATETIME,
        "revenue": ColumnType.NUMERIC,
        "region": ColumnType.CATEGORICAL,
    }


def test_sniffs_the_delimiter_of_pasted_text() -> None:
    dataset = load_table_text("a;b\n1;2\n3;4\n")
    assert list(dataset.frame.columns) == ["a", "b"]
    assert dataset.frame["a"].tolist() == [1, 3]


def test_reads_tsv_by_extension() -> None:
    dataset = load(b"a\tb\n1\t2\n", name="table.tsv")
    assert list(dataset.frame.columns) == ["a", "b"]


def test_reads_excel(tmp_path: Path, tidy_frame: pd.DataFrame) -> None:
    path = tmp_path / "revenue.xlsx"
    tidy_frame.to_excel(path, index=False)
    dataset = load(path)
    assert len(dataset.frame) == 24
    assert dataset.types["revenue"] is ColumnType.NUMERIC


def test_accepts_a_file_like_object(csv_bytes: bytes) -> None:
    assert len(load(io.BytesIO(csv_bytes), name="revenue.csv").frame) == 24


def test_a_number_like_column_of_ids_can_be_overridden(csv_bytes: bytes) -> None:
    dataset = load(csv_bytes, name="revenue.csv")
    retyped = dataset.retype("revenue", ColumnType.CATEGORICAL)
    assert retyped.types["revenue"] is ColumnType.CATEGORICAL
    assert retyped.profile("revenue").inferred is False
    # The original is untouched, so the user can undo the override.
    assert dataset.types["revenue"] is ColumnType.NUMERIC


def test_override_nulls_values_that_will_not_convert() -> None:
    dataset = load(b"code\n10\nabc\n30\n", name="codes.csv")
    retyped = dataset.retype("code", ColumnType.NUMERIC)
    assert retyped.profile("code").null_count == 1
    assert retyped.frame["code"].dropna().tolist() == [10.0, 30.0]


def test_retyping_an_unknown_column_raises(csv_bytes: bytes) -> None:
    with pytest.raises(KeyError):
        load(csv_bytes, name="revenue.csv").retype("nope", ColumnType.NUMERIC)


def test_flags_an_all_null_column() -> None:
    dataset = load(b"a,b\n1,\n2,\n", name="t.csv")
    assert "all_null" in issue_codes(dataset)
    assert dataset.profile("b").is_all_null


def test_flags_a_mixed_type_column() -> None:
    dataset = load(b"qty\n10\n20\nunknown\n30\n", name="t.csv")
    assert "mixed_types" in issue_codes(dataset)
    assert dataset.types["qty"] is ColumnType.CATEGORICAL


def test_flags_duplicate_headers() -> None:
    dataset = load(b"region,region,value\na,a,1\nb,b,2\n", name="t.csv")
    assert "duplicate_header" in issue_codes(dataset)
    assert len(set(dataset.frame.columns)) == len(dataset.frame.columns)


def test_flags_a_sparse_column() -> None:
    rows = "".join("1,\n" if i % 2 else "1,5\n" for i in range(20))
    dataset = load(f"a,b\n{rows}".encode(), name="t.csv")
    assert "sparse" in issue_codes(dataset)


def test_a_clean_file_has_no_issues(csv_bytes: bytes) -> None:
    assert load(csv_bytes, name="revenue.csv").issues == ()


def test_samples_deterministically_above_the_row_cap() -> None:
    frame = pd.DataFrame({"i": range(500), "v": range(500)})
    first = load_frame(frame, max_rows=100)
    second = load_frame(frame, max_rows=100)

    assert first.was_sampled
    assert len(first.frame) == 100
    assert first.original_row_count == 500
    assert first.sample_seed is not None
    assert "sampled" in issue_codes(first)
    pd.testing.assert_frame_equal(first.frame, second.frame)


def test_data_under_the_cap_is_not_sampled(csv_bytes: bytes) -> None:
    dataset = load(csv_bytes, name="revenue.csv")
    assert not dataset.was_sampled
    assert dataset.sample_seed is None


def test_identical_bytes_hash_identically(csv_bytes: bytes) -> None:
    assert load(csv_bytes, name="a.csv").source_hash == load(csv_bytes, name="b.csv").source_hash


def test_different_bytes_hash_differently(csv_bytes: bytes) -> None:
    other = load(csv_bytes.replace(b"north", b"west"), name="a.csv")
    assert load(csv_bytes, name="a.csv").source_hash != other.source_hash


@pytest.mark.parametrize(
    ("payload", "name", "message"),
    [
        (b"", "empty.csv", "empty"),
        (b"a,b\n", "headers_only.csv", "no rows"),
        (b"x" * (51 * 1024 * 1024), "huge.csv", "over the"),
    ],
)
def test_unusable_input_is_rejected(payload: bytes, name: str, message: str) -> None:
    with pytest.raises(IngestError, match=message):
        load(payload, name=name)


def test_a_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="no such file"):
        load(tmp_path / "absent.csv")


def test_bytes_without_a_filename_are_rejected(csv_bytes: bytes) -> None:
    with pytest.raises(IngestError, match="filename is required"):
        load(csv_bytes)


def test_empty_paste_is_rejected() -> None:
    with pytest.raises(IngestError, match="nothing was pasted"):
        load_table_text("   \n")


def test_suggest_spec_prefers_a_time_axis(csv_bytes: bytes) -> None:
    spec = suggest_spec(load(csv_bytes, name="revenue.csv"))
    assert spec.family is ChartFamily.LINE
    assert spec.x == "month"
    assert spec.y == "revenue"
    assert spec.series == "region"
    assert spec.is_baseline


def test_suggest_spec_falls_back_to_a_categorical_axis() -> None:
    dataset = load(b"region,revenue\nnorth,10\nsouth,20\n", name="t.csv")
    spec = suggest_spec(dataset)
    assert spec.family is ChartFamily.BAR
    assert spec.x == "region"
    assert spec.y == "revenue"


def test_suggest_spec_counts_rows_when_nothing_is_measurable() -> None:
    dataset = load(b"region,city\nnorth,oslo\nsouth,rome\n", name="t.csv")
    spec = suggest_spec(dataset)
    assert spec.aggregation is Aggregation.COUNT
    assert spec.y is None


def test_suggest_spec_ignores_a_high_cardinality_series_column() -> None:
    frame = pd.DataFrame(
        {"day": range(30), "value": range(30), "label": [f"s{i}" for i in range(30)]}
    )
    assert suggest_spec(load_frame(frame)).series is None


def test_suggest_spec_honours_a_requested_family(csv_bytes: bytes) -> None:
    spec = suggest_spec(load(csv_bytes, name="revenue.csv"), family=ChartFamily.AREA)
    assert spec.family is ChartFamily.AREA


def test_unparseable_input_is_rejected() -> None:
    with pytest.raises(IngestError, match="could not parse"):
        load(b"\x00\xff\xfe\x00binary garbage\x00", name="junk.dat")


def test_a_single_column_paste_still_loads() -> None:
    dataset = load_table_text("value\n1\n2\n")
    assert list(dataset.frame.columns) == ["value"]


def test_a_corrupt_excel_file_is_rejected() -> None:
    with pytest.raises(IngestError, match="could not read"):
        load(b"not really a workbook", name="book.xlsx")


def test_true_duplicate_columns_are_renamed_and_flagged() -> None:
    frame = pd.DataFrame([[1, 2, 3]], columns=["region", "region", "value"])
    dataset = load_frame(frame)
    assert "duplicate_header" in issue_codes(dataset)
    assert list(dataset.frame.columns) == ["region", "region.1", "value"]


def test_retyping_to_datetime_parses_strings() -> None:
    dataset = load(b"label\n2024-01-05\n2024-02-05\n", name="t.csv")
    retyped = dataset.retype("label", ColumnType.DATETIME)
    assert retyped.types["label"] is ColumnType.DATETIME


def test_retyping_to_boolean() -> None:
    dataset = load(b"flag\ntrue\nfalse\n", name="t.csv")
    retyped = dataset.retype("flag", ColumnType.BOOLEAN)
    assert retyped.types["flag"] is ColumnType.BOOLEAN


def test_retyping_a_number_to_categorical_keeps_the_labels() -> None:
    dataset = load(b"zip\n02134\n90210\n", name="t.csv")
    retyped = dataset.retype("zip", ColumnType.CATEGORICAL)
    assert retyped.types["zip"] is ColumnType.CATEGORICAL
    assert retyped.frame["zip"].tolist() == ["2134", "90210"]


def test_columns_of_excludes_empty_columns() -> None:
    dataset = load(b"a,b\n1,\n2,\n", name="t.csv")
    assert "b" not in dataset.columns_of(ColumnType.NUMERIC, ColumnType.CATEGORICAL)


def test_profile_of_an_unknown_column_raises(csv_bytes: bytes) -> None:
    with pytest.raises(KeyError):
        load(csv_bytes, name="revenue.csv").profile("nope")


def test_suggest_spec_rejects_a_table_it_cannot_chart() -> None:
    dataset = load(b"a,b\n1,\n2,\n", name="t.csv").retype("a", ColumnType.CATEGORICAL)
    empty = dataset.retype("a", ColumnType.NUMERIC).frame.assign(a=None)
    with pytest.raises(IngestError, match="no usable columns"):
        suggest_spec(load_frame(empty))
