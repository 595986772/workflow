# P6 Figure Audit

## Scope and provenance

- Intended use: provisional manuscript figures; the target journal and final column width are not yet fixed.
- Source table: `method_aggregates.csv`.
- Source statistics and per-seed values: `pegasus_p6_summary.json`.
- Plot implementation: `analyze_pegasus_p6.py`.
- Replication unit: scenario seed, `n=10` paired seeds (51--60).
- Main and mechanism error bars: two-sided 95% Student-t confidence intervals over seed-level values.
- Raw seed-level points are overlaid on the main baseline and mechanism completion-time panels with deterministic jitter seed `20260809`.

## Delivered files

| Figure | PNG pixels | PNG metadata | Vector fallback |
|---|---:|---:|---|
| `p6_main_baselines` | 3960 x 1470 | 300 dpi, RGBA | PDF |
| `p6_factorial_2x2` | 2010 x 1440 | 300 dpi, RGBA | PDF |
| `p6_mechanism_ablation` | 3210 x 1380 | 300 dpi, RGBA | PDF |

## Visual integrity review

- Bar charts start at zero; no broken or truncated bar baseline is used.
- Directly compared panels use explicit units and do not use dual axes, logarithmic transforms, smoothing, or omitted observations.
- Main bars show means, 95% confidence intervals, and all ten seed-level points.
- The 2x2 interaction plot uses line color plus distinct marker shape; inferential statistics are reported in `PEGASUS_P6_REPORT.md` and `PEGASUS_P6_REPORT_ZH.md` rather than encoded as significance stars.
- The mechanism plot retains the non-significant cache-dependency result and does not visually suppress it.
- No labels, legends, ticks, or data marks are clipped or incoherently overlapped in the rendered PNG files.
- The figures use an opaque white rendered background even though the PNG container has an alpha channel.
- All three one-page PDF files pass format inspection and embed their fonts as Type 0 TrueType/CID resources; no unembedded or Type 3 font resource remains.
- The PDF-only runtime export change did not alter the machine-readable analysis: `pegasus_p6_summary.json` retained the SHA-256 locked by `FINAL_LOCK.json`.

## Interpretation constraints

- `p6_main_baselines` supports a significant mean-completion-time advantage of OUR over all listed baselines. It does not support a significant P95 advantage over CoordCache-DiscreteSAC.
- `p6_factorial_2x2` is a descriptive mean interaction plot. Confidence intervals, Wilcoxon p-values, and seed wins must be supplied in the text/table when published.
- `p6_mechanism_ablation` supports task-dependency state and causal makespan reward, but not cache-dependency weighting.
- Publisher-specific width, font, color-profile, and submission-format compliance remain pending until a target journal and submission phase are chosen.
