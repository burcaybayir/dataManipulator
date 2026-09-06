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
| 2 | Baseline rendering | next — needs the Altair/Plotly call |
| 3 | Transform engine | done |
| 4 | Impact scoring | done |
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

Then distort it, deliberately and on the record:

```python
from core.prepare import prepare
from core.transforms import CherryPickedWindow, TransformStack, TruncatedAxis

stack = TransformStack().push(
    CherryPickedWindow.favouring(dataset.frame, spec, direction="down", length=5),
    TruncatedAxis(),
)
frame, distorted = stack.apply(dataset.frame, spec)

prepare(dataset.frame, spec).y_domain   # (0.0, 120512.17) — the honest baseline
prepare(frame, distorted).y_domain      # (64788.2, 70992.4) — the same data, panicking
distorted.notes                         # every technique that got it there
```

The notes travel with the spec into every export, so a variant can always be
traced back to the baseline it came from.

Then score how differently it reads:

```python
from core.scoring import score_stack

variant, scores = score_stack(dataset.frame, spec, stack)
scores.slope_ratio      # 21.7 — the line as drawn, against the baseline's
scores.direction_flip   # True — the naive reading reverses
scores.data_fidelity    # 0.17 — it stands on 17% of the rows
scores.summary()        # a caption, phrased as the estimate it is
```

Four numbers, all heuristics: they model a reader who looks at where each line
starts and ends and how much of the frame the change fills. Two things they
deliberately do not catch — a shift in level with no change in slope, and a
second y axis — are documented in `core/scoring.py` and pinned by tests, so a
future metric that starts catching one fails loudly instead of quietly
changing what the scores mean.

## Transform library

Twelve techniques, each a pure `(DataFrame, ChartSpec) -> (DataFrame,
ChartSpec)` function carrying its own explanation, why it misleads, and when
it is legitimate — most of them are legitimate somewhere, which is exactly why
they work as deceptions.

`truncated_axis` · `inverted_axis` · `cherry_picked_window` ·
`aggregation_swap` · `ratio_vs_absolute` · `rebase_index` · `bin_regrouping` ·
`smoothing` · `cumulative` · `dual_axis` · `aspect_ratio` · `outlier_drop`

Adding a thirteenth means one module in `core/transforms/` decorated with
`@register`; nothing in the UI or scoring code changes. Stacks are capped at
five transforms, past which no single technique explains the difference and
the impact scores stop being attributable.

## Layout

```
core/            analysis library — no UI code
  spec.py        ChartSpec: the object transforms rewrite
  ingest.py      loading, type inference, profiling, validation
  prepare.py     spec + data -> the values a chart would draw
  transforms/    one module per distortion technique, plus the registry
  scoring.py     how differently a variant reads from its baseline
  manifest.py    reproducible source-hash + spec + stack
  samples.py     access to the bundled demo datasets
data/samples/    demo datasets + the deterministic generator that builds them
docs/SPEC.md     requirements, architecture, staged plan
tests/           unit tests for everything in core/
```

## Sample datasets

| Sample | What it is | Built to demonstrate |
|---|---|---|
| `saas_mrr` | 36 months of MRR, growth with one bad quarter | truncated axis, cherry-picked window, cumulative |
| `support_tickets` | Two noisy years of daily volume, flat trend | smoothing, cherry-picked window, dropped outliers |
| `regional_sales` | Eight regions, heavy long tail, eight quarters | bin regrouping, ratio vs. absolute |
| `messy_inventory` | Deliberately dirty export | the validation path |

Each sample names the chart its claims are calibrated against (`sample.spec()`)
— a technique only distorts the reading of a particular chart, and swapping the
measure can turn any of them into a no-op. Those claims are enforced in
`tests/test_scoring_samples.py`.

They are generated deterministically; `python data/samples/generate.py`
reproduces the committed CSVs byte for byte.
