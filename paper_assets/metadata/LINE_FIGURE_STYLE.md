# Unified Line-Figure Style

The formal paper line charts use `paper_line_style.py` as their single style
source. The style applies to:

- `fig03a_pegasus_workflows`
- `fig08_convergence`
- `fig10_server_count_sensitivity_7methods`
- `fig_service_cache_pressure_sensitivity`
- `fig_cache_capacity_heterogeneity`

## Visual Contract

- Figure size: 180 mm x 78 mm.
- Typeface: Arial, with editable SVG text and embedded TrueType PDF fonts.
- Method identity: identical color, marker, and dash pattern in every chart.
- Emphasis: OUR is the only filled marker and uses a slightly heavier line.
- Axes: left and bottom spines only, outward ticks, and horizontal dashed grid.
- Legend: borderless and centered above the data region.
- Export: SVG, PDF, and 400 dpi PNG.

## Statistical Contract

The shared style does not alter statistical semantics. Pegasus workflow and
server-scaling plots retain their original 95% confidence intervals; service
sensitivity and cache heterogeneity retain their original three-seed standard
deviations; convergence retains the measured multi-seed online-training mean.
No result values are transformed during rendering.

