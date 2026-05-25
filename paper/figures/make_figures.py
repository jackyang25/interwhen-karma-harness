"""Generate all paper figures from results/analysis/* CSVs and JSONs.

Run from repo root:
    python paper/figures/make_figures.py

Outputs to paper/figures/*.pdf (vector).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
ANALYSIS = RESULTS / "analysis"
FIG_DIR = ROOT / "paper" / "figures"
plt.style.use(str(FIG_DIR / "paper_style.mplstyle"))

# Palette mirrored from paper/preamble.tex
COLOR = {
    "A":                 "#B0B0B0",
    "B":                 "#4A4A4A",
    "C":                 "#9DC3E6",
    "D":                 "#4F81BD",
    "E":                 "#1F4E79",
    "B_prime":           "#D9A066",
    "B_prime_E":         "#B45A1F",
    "B_prime_E_reactive":"#1F5F5B",
}

PRETTY = {
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "D",
    "E": "E",
    "B_prime": "B$'$",
    "B_prime_E": "B$'$+E",
    "B_prime_E_reactive": "B$'$+E (reactive)",
}
# y-axis labels for the forest plot include the rhetorical role inline so
# we don't need separate annotations below the chart.
FOREST_LABEL = {
    "A": "A",
    "B": "B (baseline)",
    "C": "C",
    "D": "D",
    "E": "E",
    "B_prime": "B$'$",
    "B_prime_E": "B$'$+E",
    "B_prime_E_reactive": "B$'$+E (reactive)",
}

CONDITION_ORDER = ["A", "B", "C", "D", "E", "B_prime", "B_prime_E", "B_prime_E_reactive"]


# ── Figure 1: Forest plot of accuracies ──────────────────────────────────
def forest_accuracy():
    df = pd.read_csv(ANALYSIS / "accuracy_table.csv").set_index("condition")

    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    n = len(CONDITION_ORDER)
    y_positions = np.arange(n)[::-1]   # top-to-bottom

    # Reference lines (draw first so markers sit on top)
    b_acc  = df.loc["B", "accuracy"]
    bp_acc = df.loc["B_prime", "accuracy"]
    ax.axvline(b_acc,  color="#4A4A4A", linestyle=":", linewidth=0.9, alpha=0.55)
    ax.axvline(bp_acc, color="#D9A066", linestyle=":", linewidth=0.9, alpha=0.55)

    # Data markers and CIs
    for y, cond in zip(y_positions, CONDITION_ORDER):
        row = df.loc[cond]
        acc, lo, hi = row["accuracy"], row["ci_low"], row["ci_high"]
        ax.errorbar(
            acc, y,
            xerr=[[acc - lo], [hi - acc]],
            fmt="o",
            color=COLOR[cond],
            ecolor=COLOR[cond],
            markersize=7,
            linewidth=2.0,
            capsize=4,
            zorder=3,
        )
        ax.text(hi + 0.008, y, f"{acc:.1%}", va="center", fontsize=9,
                color="#1A1A1A", zorder=4)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([FOREST_LABEL[c] for c in CONDITION_ORDER])
    ax.set_xlabel("Accuracy (Wilson 95% CI)")
    ax.set_xlim(0.30, 0.82)
    ax.set_ylim(-0.5, n - 0.5)
    # Title removed; caption carries the figure description.

    fig.tight_layout()
    fig.savefig(FIG_DIR / "forest_accuracy.pdf")
    plt.close(fig)
    print("[ok] forest_accuracy.pdf")


# ── Figure 2: Discordant pairs as a 2×2 contingency grid ────────────────
def discordant_flow():
    """Standard publication format for paired binary outcomes:
    a 2x2 grid where rows = baseline outcome, columns = treatment outcome,
    cells colored by meaning. The two off-diagonal cells are exactly the
    discordant counts McNemar's test operates on.

    Important: matplotlib's default text renderer is NOT LaTeX. We use
    mathtext ($...$) for math and unicode for symbols, not \\textbf{}.
    """
    cc = 775 - 118   # correct→correct  = 657 (concordant)
    cw = 118         # correct→wrong    = 118 (HARM)
    wc = 22          # wrong→correct    = 22  (GAIN)
    ww = 291 - 22    # wrong→wrong      = 269 (concordant)
    total = cc + cw + wc + ww  # 1066

    counts = np.array([[cc, cw],
                       [wc, ww]])
    cell_color = np.array([["#E8E8E5", "#B45A1F"],
                           ["#1F5F5B", "#E8E8E5"]])
    cell_text = np.array([["#3A3A3A", "white"],
                          ["white",   "#3A3A3A"]])
    # Only the two DISCORDANT cells get a role tag inside (verifier harm
    # / verifier gain). Concordant cells have no tag — their meaning is
    # already implied by their position in the grid.
    cell_tag = np.array([["", "verifier harm"],
                         ["verifier gain", ""]])

    fig, ax = plt.subplots(figsize=(6.4, 4.6))

    for i in range(2):
        for j in range(2):
            x0, y0 = j, 1 - i
            ax.add_patch(mpatches.Rectangle(
                (x0, y0), 1, 1,
                facecolor=cell_color[i, j], edgecolor="white", linewidth=3))
            # Big count number
            ax.text(x0 + 0.5, y0 + 0.60, f"{counts[i, j]}",
                    ha="center", va="center",
                    fontsize=30, fontweight="bold", color=cell_text[i, j])
            # Percentage
            ax.text(x0 + 0.5, y0 + 0.36, f"{counts[i,j]/total*100:.1f}%",
                    ha="center", va="center",
                    fontsize=11, color=cell_text[i, j])
            # Role tag (only on the two discordant cells)
            if cell_tag[i, j]:
                ax.text(x0 + 0.5, y0 + 0.16, cell_tag[i, j],
                        ha="center", va="center",
                        fontsize=10, color=cell_text[i, j],
                        fontweight="bold", fontstyle="italic")

    # Single set of axis headers (top + left). No marginal totals — those
    # are implied by the cell counts plus the column/row headers.
    ax.text(1.0, 2.18, "B$'$+E outcome", ha="center", va="bottom",
            fontsize=11.5, fontweight="bold", color="#1A1A1A")
    ax.text(0.5, 2.04, "correct", ha="center", va="bottom",
            fontsize=10, color="#3A3A3A")
    ax.text(1.5, 2.04, "wrong", ha="center", va="bottom",
            fontsize=10, color="#3A3A3A")
    ax.text(-0.32, 1.0, "B$'$ outcome", ha="center", va="center",
            rotation=90, fontsize=11.5, fontweight="bold", color="#1A1A1A")
    ax.text(-0.05, 1.5, "correct", ha="right", va="center",
            fontsize=10, color="#3A3A3A")
    ax.text(-0.05, 0.5, "wrong", ha="right", va="center",
            fontsize=10, color="#3A3A3A")

    # McNemar callout at the bottom
    ax.text(1.0, -0.18,
            "McNemar: $b=22$, $c=118$, $p \\approx 10^{-15}$",
            ha="center", va="top", fontsize=10.5, color="#1A1A1A",
            fontweight="bold")

    ax.set_xlim(-0.55, 2.30)
    ax.set_ylim(-0.42, 2.32)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "discordant_flow.pdf")
    plt.close(fig)
    print("[ok] discordant_flow.pdf")


# ── Figure 3: Flag composition (2-panel horizontal bars) ─────────────────
def flag_composition():
    with open(ANALYSIS / "verifier_characterization.json") as f:
        vc = json.load(f)

    def top_n(d, n=10):
        return sorted(d.items(), key=lambda x: -x[1])[:n]

    upfront = top_n(vc["B_prime_E"]["violations_by_field"])
    reactive = top_n(vc["B_prime_E_reactive"]["violations_by_field"])

    upfront_total = vc["B_prime_E"]["total_violations"]
    reactive_total = vc["B_prime_E_reactive"]["total_violations"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.5, 4.4), sharex=False)

    def plot_panel(ax, data, total, title, accent_color):
        fields = [f for f, _ in data][::-1]
        counts = [c for _, c in data][::-1]
        pcts = [c / total * 100 for c in counts]
        colors = [accent_color if f == "sex" else "#B5B5B5" for f in fields]
        y_pos = np.arange(len(fields))
        ax.barh(y_pos, pcts, color=colors, edgecolor="white", linewidth=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(fields, fontsize=9)
        ax.set_xlabel("Fraction of all flags (%)")
        # Panel title -- minimal labeling so caption carries the description.
        ax.set_title(title, loc="left", pad=8, fontsize=10, fontweight="normal", color="#3A3A3A")
        # Annotate counts at bar end
        for y, count, pct in zip(y_pos, counts, pcts):
            ax.text(pct + 0.5, y, f"{count}", va="center", fontsize=8,
                    color="#3A3A3A")
        ax.set_xlim(0, max(pcts) * 1.18)

    plot_panel(axL, upfront, upfront_total,
               f"Upfront B$'$+E (N={upfront_total} flags)", "#B45A1F")
    plot_panel(axR, reactive, reactive_total,
               f"Reactive B$'$+E (N={reactive_total} flags)", "#1F5F5B")

    # Suptitle removed; caption carries the description.

    fig.tight_layout()
    fig.savefig(FIG_DIR / "flag_composition.pdf")
    plt.close(fig)
    print("[ok] flag_composition.pdf")


# ── Figure 4: Per-category heatmap ───────────────────────────────────────
def category_heatmap():
    df = pd.read_csv(ANALYSIS / "per_category_accuracy.csv", index_col=0)
    # Reorder conditions to match paper narrative
    df = df[CONDITION_ORDER]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    # Use a sequential colormap; convert to numpy for imshow
    data = df.values
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    # Ticks
    ax.set_xticks(np.arange(len(CONDITION_ORDER)))
    ax.set_xticklabels([PRETTY[c] for c in CONDITION_ORDER],
                       rotation=30, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(df.index)))
    cleaned = [c.replace("_calculators", "").replace("_", " ") for c in df.index]
    ax.set_yticklabels(cleaned, fontsize=8)

    # Annotate cell values
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            color = "white" if val < 0.35 or val > 0.85 else "#222"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    # Title removed; caption carries the description.
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Accuracy", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "category_heatmap.pdf")
    plt.close(fig)
    print("[ok] category_heatmap.pdf")


# ── Figure 5: Cost-accuracy Pareto ───────────────────────────────────────
def pareto():
    """Clean scatter, no frontier line, no shading, no on-chart callouts.
    The narrative goes in the caption; the chart shows only data."""
    cost_df = pd.read_csv(ANALYSIS / "cost_latency_table.csv").set_index("condition")
    acc_df = pd.read_csv(ANALYSIS / "accuracy_table.csv").set_index("condition")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    # Hand-tuned label offsets to avoid all collisions
    LABEL_OFFSET = {
        "A":                 ( 0,  0.024, "center", "bottom"),
        "B":                 (-2,  0.022, "right",  "bottom"),
        "C":                 ( 0, -0.024, "center", "top"),
        "D":                 ( 3,  0.005, "left",   "center"),
        "E":                 ( 0, -0.024, "center", "top"),
        "B_prime":           ( 0,  0.022, "center", "bottom"),
        "B_prime_E":         ( 0, -0.024, "center", "top"),
        "B_prime_E_reactive":( 0,  0.022, "center", "bottom"),
    }

    coords = {}
    for cond in CONDITION_ORDER:
        x = cost_df.loc[cond, "mean_total_tokens"] / 1000
        y = acc_df.loc[cond, "accuracy"]
        coords[cond] = (x, y)

    for cond, (x, y) in coords.items():
        ax.scatter(x, y, s=110, color=COLOR[cond], edgecolor="white",
                   linewidth=1.6, zorder=4)

    for cond, (x, y) in coords.items():
        dx, dy, ha, va = LABEL_OFFSET[cond]
        ax.text(x + dx, y + dy, PRETTY[cond],
                ha=ha, va=va, fontsize=10, color="#1A1A1A", zorder=5)

    ax.set_xlabel("Mean total tokens per vignette (thousands)")
    ax.set_ylabel("Accuracy")
    # Title removed; caption carries the description.
    ax.set_xlim(-3, 100)
    ax.set_ylim(0.36, 0.80)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "pareto.pdf")
    plt.close(fig)
    print("[ok] pareto.pdf")


# ── Figure 7: Trajectory categories (clean horizontal bar chart) ────────
def trajectory_flow():
    """Six trajectory categories shown as a horizontal bar chart.
    Replaces an earlier alluvial which was too dense to read."""
    bp  = pd.read_csv(RESULTS / "raw" / "condition_B_prime.csv")[["id","correct"]].rename(columns={"correct":"bp"})
    bpe = pd.read_csv(RESULTS / "raw" / "condition_B_prime_E.csv")[["id","correct"]].rename(columns={"correct":"bpe"})
    bpr = pd.read_csv(RESULTS / "raw" / "condition_B_prime_E_reactive.csv")[["id","correct"]].rename(columns={"correct":"bpr"})
    df = bp.merge(bpe, on="id").merge(bpr, on="id")
    total = len(df)

    # Six trajectory categories — ordered by narrative (status quo first,
    # then verifier-related groups, then minor transitions).
    GROUPS = [
        # (label, color, count-computer)
        ("always correct",
         "#C9C9C5",
         ((df.bp==True) & (df.bpe==True) & (df.bpr==True)).sum()),
        ("always wrong",
         "#7A7A78",
         ((df.bp==False) & (df.bpe==False) & (df.bpr==False)).sum()),
        ("reactive recovers upfront harm",
         "#1F5F5B",
         ((df.bp==True) & (df.bpe==False) & (df.bpr==True)).sum()),
        ("upfront harm persists",
         "#B45A1F",
         ((df.bp==True) & (df.bpe==False) & (df.bpr==False)).sum()),
        ("reactive newly hurts",
         "#C55A1E",
         ((df.bp==True) & (df.bpe==True) & (df.bpr==False)).sum()),
        ("minor transitions",
         "#B9A8C9",
         (
            ((df.bp==False) & (df.bpe==True) & (df.bpr==True)).sum()
          + ((df.bp==False) & (df.bpe==True) & (df.bpr==False)).sum()
          + ((df.bp==False) & (df.bpe==False) & (df.bpr==True)).sum()
         )),
    ]
    labels = [g[0] for g in GROUPS]
    colors = [g[1] for g in GROUPS]
    counts = [int(g[2]) for g in GROUPS]

    fig, ax = plt.subplots(figsize=(7.6, 4.2))

    # Horizontal bars; top-to-bottom = narrative order above
    y_positions = np.arange(len(GROUPS))[::-1]
    ax.barh(y_positions, counts, color=colors, edgecolor="white",
            linewidth=1.2, height=0.72)

    # Count annotations to the right of each bar
    for y, n in zip(y_positions, counts):
        pct = n / total * 100
        ax.text(n + 12, y, f"{n}  ({pct:.1f}%)",
                ha="left", va="center", fontsize=10,
                color="#1A1A1A")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlabel(f"Vignettes (out of $N={total}$)")
    ax.set_xlim(0, max(counts) * 1.22)
    # Title removed; caption carries the description.

    fig.tight_layout()
    fig.savefig(FIG_DIR / "trajectory_flow.pdf")
    plt.close(fig)
    print("[ok] trajectory_flow.pdf")


# ── Figure 6: Token decomposition (stacked bars) ─────────────────────────
def token_decomposition():
    """Stacked bar: extractor (API) tokens vs primary (on-GPU) tokens.
    Makes the reactive cost story visible at a glance."""
    df = pd.read_csv(ANALYSIS / "cost_latency_table.csv").set_index("condition")
    df = df.reindex(CONDITION_ORDER)

    fig, ax = plt.subplots(figsize=(7.4, 4.3))

    x = np.arange(len(CONDITION_ORDER))
    bar_w = 0.6

    api_tokens = (
        df["mean_api_prompt_tokens"].fillna(0)
        + df["mean_api_completion_tokens"].fillna(0)
    ).values / 1000
    gpu_tokens = (
        df["mean_on_gpu_prompt_tokens"].fillna(0)
        + df["mean_on_gpu_completion_tokens"].fillna(0)
    ).values / 1000

    ax.bar(x, gpu_tokens, bar_w, label="Primary (Qwen3, on-GPU)",
           color="#4F81BD", edgecolor="white", linewidth=0.8)
    ax.bar(x, api_tokens, bar_w, bottom=gpu_tokens,
           label="Extractor / verifier (Sonnet, API)",
           color="#B45A1F", edgecolor="white", linewidth=0.8)

    # Total labels
    totals = gpu_tokens + api_tokens
    for xi, total in zip(x, totals):
        ax.text(xi, total + 1.5, f"{total:.1f}K", ha="center",
                fontsize=9, color="#1A1A1A", fontweight="semibold")

    # Annotate the API portion specifically on the high-cost upfront cases
    for cond, xi, api in zip(CONDITION_ORDER, x, api_tokens):
        if api > 5:
            ax.text(xi, gpu_tokens[list(CONDITION_ORDER).index(cond)] + api / 2,
                    f"{api:.1f}K\nSonnet",
                    ha="center", va="center", fontsize=8, color="white",
                    fontweight="semibold")

    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[c] for c in CONDITION_ORDER], rotation=15,
                       ha="right", fontsize=9)
    ax.set_ylabel("Mean tokens per vignette (thousands)")
    # Title removed; caption carries the description.
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, max(totals) * 1.18)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "token_decomposition.pdf")
    plt.close(fig)
    print("[ok] token_decomposition.pdf")


if __name__ == "__main__":
    forest_accuracy()
    discordant_flow()
    flag_composition()
    category_heatmap()
    pareto()
    token_decomposition()
    trajectory_flow()
    print(f"\nAll figures written to {FIG_DIR}")
