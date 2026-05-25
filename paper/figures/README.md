# Paper figures

Each figure here is generated externally from `results/` data via matplotlib
(using `paper_style.mplstyle`) or hand-authored as TikZ inside the .tex.

Figures are referenced by `\includegraphics{<name>}` from sections (the
preamble sets `\graphicspath{{figures/}}`, so no path prefix is needed).

## Inventory

| # | Filename                  | Section            | Type        | Data source                                          |
|---|---------------------------|--------------------|-------------|------------------------------------------------------|
| 1 | `forest_accuracy.pdf`     | §4.2 Results       | Forest plot | `analysis/accuracy_table.csv`                        |
| 2 | `mechanism.tex` (inline)  | §3.3 Methods       | TikZ        | Hand-authored                                        |
| 3 | `discordant_flow.pdf`     | §4.3 Results       | Sankey      | `analysis/paired_rows.csv`                           |
| 4 | `flag_composition.pdf`    | §4.4 Results       | Bars (2-up) | `analysis/condition_*_verifier_summary.json`         |
| 5 | `category_heatmap.pdf`    | §4.2 Results       | Heatmap     | `analysis/per_category_accuracy.csv`                 |
| 6 | `pareto.pdf`              | §4.6 Results       | Scatter     | `analysis/cost_latency_table.csv`                    |
| 7 | `loop_trace.tex` (inline) | §5.3 Discussion    | TikZ        | One annotated example vignette from `raw/`           |

## Style

All matplotlib figures must `plt.style.use("paper/figures/paper_style.mplstyle")`
before plotting. Save as PDF for vector quality:

```python
fig.savefig("paper/figures/forest_accuracy.pdf")
```

## Color encoding

Color = condition family (consistent across every figure):

- Anchors A, B          → muted grays (`#B0B0B0`, `#4A4A4A`)
- Primary C, D, E       → blues, increasing saturation
- Exploratory B', B'+E  → warm earth tones
- Diagnostic reactive   → accent teal (`#1F5F5B`)

The reader decodes any chart by color before reading the legend.
