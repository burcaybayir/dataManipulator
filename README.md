# dataManipulator

**Chart Deception Detector** — take a dataset and a claim, and see how the same
numbers can be made to argue either side.

The app renders an honest baseline chart alongside deliberately distorted
variants. Every variant is labeled with the technique used (truncated axis,
cherry-picked window, aggregation swap, …), scored for how much it changes a
reader's impression, and stamped with its transform stack on export. It is a
data-literacy and chart-auditing tool, not a way to produce misleading charts
and pass them off as genuine.

Full requirements and the staged plan: [`docs/SPEC.md`](docs/SPEC.md).

## Status

| Stage | Scope | State |
|---|---|---|
| 0 | Project setup, lint/type/test tooling, CI | done |
| 1 | `ChartSpec`, ingestion + profiling, sample data | done |
| 2 | Baseline rendering | next |
| 3 | Transform engine | |
| 4 | Impact scoring | |
| 5 | Gallery and claim evaluation | |
| 6 | Export and provenance | |
| 7 | Polish and ship | |

## Getting started

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run pytest              # tests with coverage
uv run ruff check .        # lint
uv run ruff format .       # format
uv run mypy core           # types (strict)
```

Optionally, `uv run pre-commit install` to run lint and format on commit.

## Using the library

```python
from core.ingest import load, suggest_spec
from core.samples import load_sample

dataset = load("data/samples/saas_mrr.csv")   # or load_sample("saas_mrr")
print(dataset.types)                          # inferred column types
print(dataset.issues)                         # validation warnings
spec = suggest_spec(dataset)                  # an honest starting chart
assert spec.is_baseline
```

Type inference is advisory — override it when it guesses wrong:

```python
from core.ingest import ColumnType

dataset = dataset.retype("zip_code", ColumnType.CATEGORICAL)
```

## Layout

```
core/            analysis library — no UI code
  spec.py        ChartSpec: the object transforms rewrite
  ingest.py      loading, type inference, profiling, validation
  samples.py     access to the bundled demo datasets
data/samples/    demo datasets + the deterministic generator that builds them
docs/SPEC.md     requirements, architecture, staged plan
tests/           unit tests for everything in core/
```

## Sample datasets

| Sample | What it is | Built to demonstrate |
|---|---|---|
| `saas_mrr` | 36 months of MRR, growth with one bad quarter | truncated axis, cherry-picked window, cumulative |
| `support_tickets` | Two noisy years of daily volume, flat trend | smoothing, aggregation swap, cherry-picked window |
| `regional_sales` | Eight regions, heavy long tail, eight quarters | bin regrouping, aggregation swap, ratio vs. absolute |
| `messy_inventory` | Deliberately dirty export | the validation path |

They are generated deterministically; `python data/samples/generate.py`
reproduces the committed CSVs byte for byte.
