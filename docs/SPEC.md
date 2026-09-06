# dataManipulator — Chart Deception Detector

## 1. Product summary

A Streamlit application that takes a tabular dataset and a stated claim, then
demonstrates how the *same numbers* can be rendered to support opposite
conclusions. Every distortion is applied deliberately, named, and explained.

The app is a teaching and auditing tool, not a tool for producing misleading
charts to pass off as genuine. Every exported image carries a provenance
footer naming the transforms applied.

**Primary users**

- Analysts and journalists sanity-checking a chart before publishing it.
- Readers who want to know why a chart in the wild feels off.
- Instructors teaching data literacy who need reproducible examples.

**Core loop**

Upload data → pick x/y/series → state the claim → app generates an *honest
baseline* chart plus N *distorted variants*, each labeled with the technique,
the parameter used, and the size of the effect on the reader's impression.

## 2. Functional requirements

### FR-1 Data ingestion
- Accept CSV, TSV, XLSX (first sheet), and paste-a-table input.
- Files up to 50 MB / 1M rows; rows beyond a configurable cap are sampled with
  a visible warning.
- Type inference for numeric, categorical, datetime; user can override any
  column's inferred type.
- Reject/flag: all-null columns, mixed-type columns, duplicate headers.

### FR-2 Chart specification
- User selects chart family (line, bar, area, scatter, pie), x field, y field,
  optional series field, and aggregation (sum, mean, median, count).
- Baseline chart is rendered with defensible defaults: zero-based y-axis for
  every family, full available range, absolute values, no smoothing. Line and
  scatter charts may defensibly start above zero, but only via an explicit,
  labeled `truncated_axis` transform — never silently in the baseline.

### FR-3 Transform library
Each transform is a named, parameterized, pure function over `(DataFrame,
ChartSpec) -> (DataFrame, ChartSpec)`. Minimum viable set:

| Technique | What it does | Parameter |
|---|---|---|
| `truncated_axis` | Sets y-min above zero | y-min or % of range |
| `inverted_axis` | Flips y direction | none |
| `cherry_picked_window` | Restricts to a favorable date/index range | start, end |
| `aggregation_swap` | Changes mean↔median↔sum | target aggregation |
| `ratio_vs_absolute` | Shows % change instead of raw counts (or vice versa) | mode |
| `rebase_index` | Re-indexes all series to 100 at a chosen point | anchor |
| `bin_regrouping` | Merges or splits categorical bins | grouping map |
| `smoothing` | Applies a rolling mean to hide volatility | window |
| `cumulative` | Plots the running total instead of the period value | none |
| `dual_axis` | Two series on independent scales to imply correlation | scale factors |
| `aspect_ratio` | Stretches/squashes the plot to steepen or flatten trend | w:h |
| `outlier_drop` | Removes points outside a threshold without disclosure | z-threshold |

- Transforms are composable; the app records an ordered transform stack.
- Each transform declares: display name, one-line explanation, why it misleads,
  and whether it is ever legitimate (most are — that is the point).

### FR-4 Impact scoring
For each variant, compute and display:
- **Direction flip** — does the naive reading of the chart reverse? (bool)
- **Slope ratio** — visual slope of variant ÷ baseline.
- **Magnitude ratio** — apparent effect size ÷ true effect size.
- **Data fidelity** — % of original rows still represented.

Scores are heuristics, labeled as such in the UI.

### FR-5 Claim evaluation
- User enters a plain-text claim and marks its direction (up/down/no change).
- App reports which variants support it, which contradict it, and states
  plainly whether the baseline supports it.

### FR-6 Explanation output
- Side-by-side gallery: baseline vs. variants, each captioned with technique,
  parameter, and impact scores.
- Per-variant "how to spot this" note aimed at a reader who only sees the
  finished chart.

### FR-7 Export
- PNG/SVG per chart and a combined PDF/Markdown report.
- Every export embeds a provenance footer: source file hash, transform stack,
  and a `DISTORTED — <technique>` marker on non-baseline charts. Non-optional.

### FR-8 Session persistence
- Save/load an analysis as a JSON manifest (data hash + spec + transform stack)
  so results are reproducible without re-uploading.

## 3. Non-functional requirements

- **Performance**: baseline + 6 variants render in under 2 s for 100k rows.
- **Determinism**: identical input + manifest produces byte-identical charts;
  all sampling seeded.
- **Privacy**: uploads stay in memory or a session-scoped temp dir, purged on
  session end. No third-party telemetry, no outbound data calls.
- **Accessibility**: colorblind-safe palette, ≥4.5:1 text contrast, every chart
  paired with a text summary and a data table.
- **Portability**: Python 3.11+, runs locally with `streamlit run`; Dockerfile
  provided.
- **Testability**: transforms are pure and unit-tested independently of the UI.

## 4. Architecture

```
dataManipulator/
├── app/                  # Streamlit UI only — no analysis logic
│   ├── main.py
│   └── views/            # upload, spec, gallery, export panels
├── core/
│   ├── ingest.py         # loading, type inference, validation
│   ├── spec.py           # ChartSpec dataclass
│   ├── transforms/       # one module per technique + registry.py
│   ├── scoring.py        # impact metrics
│   ├── render.py         # ChartSpec -> Altair/Plotly figure
│   └── report.py         # export + provenance stamping
├── data/samples/         # bundled demo datasets
├── tests/
└── docs/
```

Key decision: the transform registry is the extension point. Adding a
technique means one module plus one registry entry — no UI changes.

## 5. Stages

### Stage 0 — Project setup (0.5 day)
`pyproject.toml` (uv or poetry), ruff + black + mypy, pytest, pre-commit, CI
workflow, `docs/SPEC.md`.
*Exit:* `pytest` and `ruff check` pass green on an empty suite in CI.

### Stage 1 — Core data model (1–2 days)
`ChartSpec` dataclass, ingestion with type inference and validation, two or
three bundled sample datasets with known properties.
*Exit:* load any sample to a validated DataFrame + inferred spec; ingestion
unit tests pass.

### Stage 2 — Baseline rendering (1–2 days)
`render.py` producing an honest chart from a spec, with defensible defaults
enforced. Minimal Streamlit page: upload → configure → see baseline.
*Exit:* end-to-end upload-to-chart works for all five chart families.

### Stage 3 — Transform engine (3–4 days) — done
Registry, the twelve transforms above, composition, and the manifest format.
Each transform ships with its own tests and metadata.
*Exit:* any transform stack applies cleanly; round-trips through a manifest;
100% test coverage on `core/transforms/`.

Built alongside it: `core/prepare.py`, the data half of Stage 2's renderer,
pulled forward because a transform that changes nothing observable cannot be
tested. It fixes the order in which distortions hit the numbers —
`window -> aggregate -> smooth -> accumulate -> restate` — and owns the honest
axis defaults. Stage 2 now only needs to turn its output into marks.

### Stage 4 — Impact scoring (1–2 days)
Direction flip, slope ratio, magnitude ratio, fidelity. Calibrated against
hand-checked cases in the sample data.
*Exit:* scores match expected values on a fixture set of known distortions.

### Stage 5 — Gallery and claim evaluation (2–3 days)
Side-by-side variant gallery, technique captions, claim entry, support/contradict
verdicts.
*Exit:* a user can go from raw CSV to "this claim survives the baseline but
only because of a truncated axis" without touching code.

### Stage 6 — Export and provenance (1–2 days)
PNG/SVG/PDF export, Markdown report, provenance footer, source hashing.
*Exit:* every exported asset carries an unremovable transform stack; an export
can be reproduced from its manifest alone.

### Stage 7 — Polish and ship (2–3 days)
Accessibility pass, performance tuning to the 2 s budget, error states, README
with screenshots, Dockerfile, deploy.
*Exit:* non-technical tester completes the core loop unaided.

**Estimate:** roughly 12–19 working days for a solo build. Stages 0–3 are the
critical path; everything after 4 can ship incrementally.

## 6. Out of scope (v1)

- Reading a chart *image* and reverse-engineering its data (OCR/vision).
- Live database or API connectors — files only.
- Multi-user accounts, sharing, or server-side persistence.
- Automated claim extraction from free text (user states the claim manually).

## 7. Open questions

1. Altair or Plotly? Altair is declarative and matches the spec model; Plotly
   has better interactivity and PNG export. Leaning Altair with `vl-convert`.
2. Should distorted PNGs be watermarked visibly, or is the footer enough?
3. ~~Do we cap the transform stack depth to keep impact scores interpretable?~~
   Resolved in Stage 3: capped at five (`core.transforms.stack.MAX_DEPTH`).
   Beyond that no single technique explains the difference from the baseline,
   so a score cannot be attributed to one.
