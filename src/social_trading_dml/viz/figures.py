"""
Journal-quality visualization for the causal ML analysis.

Style requirements:
- Colorblind-safe palette (2-4 colors max); interpretable in grayscale
- No top/right spines; minimal gridlines
- Professional font family (DejaVu Sans / Roboto)
- 300 dpi PNG + vector PDF to config.FIGURES_DIR (reports/figures/)
- CIs shown on every point estimate
- N annotated on each figure; panel labels for multi-panel
- No default matplotlib colors, no 3D, no chartjunk
- Distribution figures separate extensive from intensive margin
"""

import logging
import os
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch

from social_trading_dml.config import FIGURES_DIR

logger = logging.getLogger(__name__)

# ──────────────────────────── Global style ────────────────────────────

# Colorblind-safe palette (based on Wong 2011)
COLORS = {
    "primary": "#2B6CB0",     # Steel blue
    "secondary": "#C05621",   # Burnt orange
    "tertiary": "#38A169",    # Forest green
    "neutral": "#718096",     # Slate gray
    "highlight": "#D69E2E",   # Gold
    "negative": "#E53E3E",    # Red
    "positive": "#38A169",    # Green
    "ci_fill": "#BEE3F8",    # Light blue fill
    "bg": "#FAFAFA",         # Near-white background
}

# Output directory; set_figures_dir() overrides it (e.g. for smoke runs)
FIGDIR = str(FIGURES_DIR)


def set_figures_dir(path) -> None:
    """Write subsequent figures to `path` instead of config.FIGURES_DIR."""
    global FIGDIR
    FIGDIR = str(path)


def set_journal_style():
    """Set matplotlib rcParams for journal-quality figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Roboto", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "lines.linewidth": 1.5,
        "patch.linewidth": 0.5,
    })


def _save_fig(fig, name: str, figdir: Optional[str] = None):
    """Save figure as both PNG and PDF (to FIGDIR unless `figdir` is given)."""
    figdir = figdir or FIGDIR
    os.makedirs(figdir, exist_ok=True)
    png_path = os.path.join(figdir, f"{name}.png")
    pdf_path = os.path.join(figdir, f"{name}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    logger.info(f"Saved: {png_path}, {pdf_path}")
    plt.close(fig)
    return png_path, pdf_path


# ──────────────────────────── Figure 1: Sample Flow ────────────────────────────

def plot_sample_flow(
    stats: Dict[str, Any],
    figdir: Optional[str] = None,
) -> str:
    """
    Sample construction flow: raw rows → keep_country filter → analysis sample,
    with N in each box and the rows removed (and why) between boxes. Steps that
    removed nothing are not drawn.
    """
    set_journal_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axis("off")

    steps = [("Raw data", stats.get("raw_rows"), COLORS["neutral"])]
    removals = []

    if stats.get("fully_blank_rows_removed", 0):
        steps.append(("After blank-row removal", stats.get("rows_after_blank_removal"),
                      COLORS["primary"]))
        removals.append((stats["fully_blank_rows_removed"], "fully blank rows"))

    if stats.get("keep_country_rows_removed", 0):
        steps.append(("After keep_country filter", stats.get("rows_after_keep_country"),
                      COLORS["secondary"]))
        removals.append((stats["keep_country_rows_removed"], "keep_country = 0"))

    reasons = []
    if stats.get("dropped_missing_lags"):
        reasons.append(f"first week per trader (no lag): {stats['dropped_missing_lags']:,}")
    if stats.get("dropped_missing_tenure"):
        reasons.append(f"missing joining date: {stats['dropped_missing_tenure']:,}")
    if stats.get("dropped_pre_join"):
        reasons.append(f"weeks before joining: {stats['dropped_pre_join']:,}")
    steps.append(("Final analysis sample",
                  stats.get("final_analysis_rows", stats.get("final_rows")),
                  COLORS["tertiary"]))
    removals.append((stats.get("rows_dropped_for_analysis"), "\n".join(reasons)))

    n_steps = len(steps)
    y_positions = np.linspace(0.86, 0.14, n_steps)
    box_w, box_h, x0 = 0.42, 0.13, 0.10

    for i, (label, n, color) in enumerate(steps):
        ax.add_patch(FancyBboxPatch(
            (x0, y_positions[i] - box_h / 2), box_w, box_h, boxstyle="round,pad=0.01",
            facecolor=color, alpha=0.15, edgecolor=color, linewidth=1.5))
        ax.text(x0 + box_w / 2, y_positions[i] + 0.025, label,
                ha="center", va="center", fontsize=10, fontweight="bold")
        n_str = f"N = {n:,}" if isinstance(n, (int, float)) else "N = ?"
        ax.text(x0 + box_w / 2, y_positions[i] - 0.03, n_str,
                ha="center", va="center", fontsize=9.5, color=color)

        if i < n_steps - 1:
            ax.annotate("", xy=(x0 + box_w / 2, y_positions[i + 1] + box_h / 2),
                        xytext=(x0 + box_w / 2, y_positions[i] - box_h / 2),
                        arrowprops=dict(arrowstyle="->", color=COLORS["neutral"], lw=1.5))
            removed, why = removals[i]
            mid_y = (y_positions[i] + y_positions[i + 1]) / 2
            text = f"−{removed:,} rows" if isinstance(removed, (int, float)) else "−?"
            if why:
                text += f"\n{why}"
            ax.text(x0 + box_w / 2 + 0.05, mid_y, text, ha="left", va="center",
                    fontsize=8.5, color=COLORS["negative"])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Sample Construction", fontsize=13, fontweight="bold", pad=10)

    png_path, _ = _save_fig(fig, "fig1_sample_flow", figdir)
    return png_path


# ──────────────────────────── Figure 2: Causal DAG ────────────────────────────

def plot_causal_dag(figdir: Optional[str] = None) -> str:
    """
    Causal DAG showing T, Y, and the confounder set with directed edges.
    Uses networkx + matplotlib.
    """
    set_journal_style()

    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx not available — skipping DAG plot")
        return ""

    G = nx.DiGraph()

    # Nodes with labels
    G.add_node("T", label="Self-Trade\nCount (T)")
    G.add_node("Y", label="Total\nGain (Y)")
    G.add_node("L", label="Copier\nAudience (L₁)")
    G.add_node("F", label="Follower\nAudience (L₁)")
    G.add_node("G", label="Guru\nCount (L₁)")
    G.add_node("FO", label="Followed\nCount (L₁)")
    G.add_node("GD", label="Gain\nDiff (L₁)")
    G.add_node("GT", label="Reward\nThreshold (L₁)")
    G.add_node("TN", label="Tenure")
    G.add_node("C", label="Country")
    G.add_node("TT", label="Total/Self\nTrxn (L₁)")
    G.add_node("WK", label="Week\nShocks (FE)")
    G.add_node("U", label="Unobserved\nSkill (U)")
    G.add_node("TN1", label="Self-trades\nnext week")

    # Confounders → T and → Y (backdoor paths)
    confounders = ["L", "F", "G", "FO", "GD", "GT", "TN", "C", "TT", "WK"]
    for c in confounders:
        G.add_edge(c, "T")
        G.add_edge(c, "Y")

    # T → Y (causal effect of interest)
    G.add_edge("T", "Y")

    # Unobserved confounder
    G.add_edge("U", "T")
    G.add_edge("U", "Y")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Manual positions for a clean layout
    pos = {
        "T": (0.25, 0.5),
        "Y": (0.75, 0.5),
        "L": (0.1, 0.88),
        "F": (0.28, 0.88),
        "G": (0.46, 0.88),
        "FO": (0.64, 0.88),
        "GD": (0.82, 0.88),
        "GT": (0.1, 0.12),
        "TN": (0.28, 0.12),
        "C": (0.46, 0.12),
        "TT": (0.64, 0.12),
        "WK": (0.82, 0.12),
        # Off the T -> Y line so its dashed edges stay visible
        "U": (0.5, 0.66),
        "TN1": (0.5, 0.33),
    }

    labels = nx.get_node_attributes(G, "label")

    # Draw confounder nodes
    conf_nodes = [n for n in G.nodes if n not in ("T", "Y", "U", "TN1")]
    nx.draw_networkx_nodes(G, pos, nodelist=["TN1"], node_color=COLORS["highlight"],
                           alpha=0.25, node_size=2300, node_shape="s", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=conf_nodes,
                           node_color=COLORS["primary"], alpha=0.2,
                           node_size=2000, node_shape="o", ax=ax)

    # T and Y nodes
    nx.draw_networkx_nodes(G, pos, nodelist=["T"],
                           node_color=COLORS["secondary"], alpha=0.3,
                           node_size=2500, node_shape="s", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=["Y"],
                           node_color=COLORS["tertiary"], alpha=0.3,
                           node_size=2500, node_shape="s", ax=ax)

    # Unobserved node
    nx.draw_networkx_nodes(G, pos, nodelist=["U"],
                           node_color=COLORS["negative"], alpha=0.15,
                           node_size=2000, node_shape="d", ax=ax)

    # Draw edges
    causal_edges = [("T", "Y")]
    G.add_edge("Y", "TN1")
    feedback_edges = [("Y", "TN1")]
    confounder_edges = [
        (u, v) for u, v in G.edges
        if (u, v) not in causal_edges + feedback_edges and u != "U"
    ]
    unobserved_edges = [(u, v) for u, v in G.edges if u == "U"]

    nx.draw_networkx_edges(G, pos, edgelist=confounder_edges,
                           edge_color=COLORS["neutral"], alpha=0.5,
                           arrows=True, arrowsize=12, ax=ax,
                           connectionstyle="arc3,rad=0.05")
    nx.draw_networkx_edges(G, pos, edgelist=causal_edges,
                           edge_color=COLORS["secondary"], width=2.5,
                           arrows=True, arrowsize=15, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=unobserved_edges,
                           edge_color=COLORS["negative"], style="dashed",
                           width=1.5, arrows=True, arrowsize=12, ax=ax,
                           node_size=2500, min_target_margin=12)

    nx.draw_networkx_edges(G, pos, edgelist=feedback_edges, edge_color=COLORS["highlight"],
                           style="dashed", width=2, arrows=True, arrowsize=14, ax=ax,
                           connectionstyle="arc3,rad=-0.25", node_size=2500,
                           min_target_margin=14)

    # Labels
    nx.draw_networkx_labels(G, pos, labels, font_size=8,
                            font_weight="bold", ax=ax)

    ax.set_title("Assumed Causal DAG: Self-Trading → Gains\n"
                 "(trader FE absorb time-invariant U; time-varying U is not ruled out)",
                 fontsize=12, fontweight="bold")
    ax.axis("off")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS["secondary"], linewidth=2.5,
               label="Causal effect (T → Y)"),
        Line2D([0], [0], color=COLORS["neutral"], alpha=0.5, linewidth=1.5,
               label="Confounding paths"),
        Line2D([0], [0], color=COLORS["negative"], linewidth=1.5,
               linestyle="--", label="Unobserved (U)"),
        Line2D([0], [0], color=COLORS["highlight"], linewidth=2, linestyle="--",
               label="Feedback: gains → next week's self-trading"),
    ]
    # Legend below the diagram so it never covers a node
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=2, fontsize=9, frameon=False)
    ax.margins(x=0.06, y=0.08)

    png_path, _ = _save_fig(fig, "fig3_causal_dag", figdir)
    return png_path


# ──────────────────────── Figure 3: Forest Plot ──────────────────────

def plot_forest_comparison(
    results_list: List[Dict[str, Any]],
    figdir: Optional[str] = None,
) -> str:
    """
    Main estimates side by side. Thick bar: trader-clustered 95% CI; thin
    light bar: week wild-cluster-bootstrap 95% CI (where available), which is
    the relevant uncertainty when the identifying variation sits in few weeks.
    """
    set_journal_style()
    n_models = len(results_list)
    fig, ax = plt.subplots(figsize=(10, max(3, n_models * 0.9 + 1.2)))
    y_positions = list(range(n_models - 1, -1, -1))
    all_vals = []

    for i, res in enumerate(results_list):
        coef = res.get("ate", res.get("coef", 0))
        lo, hi = res.get("ate_ci_lower", coef), res.get("ate_ci_upper", coef)
        wlo, whi = res.get("ate_ci_lower_week_wild"), res.get("ate_ci_upper_week_wild")
        color = COLORS["primary"] if "DML" in res.get("model_name", "") else COLORS["secondary"]
        if wlo is not None and whi is not None:
            ax.plot([wlo, whi], [y_positions[i]] * 2, "-", color=color, alpha=0.35,
                    linewidth=7, solid_capstyle="butt", zorder=2)
            all_vals.extend([wlo, whi])
        ax.plot([lo, hi], [y_positions[i]] * 2, "-", color=color, linewidth=2.2, zorder=4)
        ax.plot(coef, y_positions[i], "o", color=color, markersize=8, zorder=5)
        all_vals.extend([lo, hi])

    ax.axvline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.8, alpha=0.7)
    span = max(all_vals) - min(all_vals)
    ax.set_xlim(min(all_vals) - 0.05 * span, max(all_vals) + 0.65 * span)

    for i, res in enumerate(results_list):
        coef = res.get("ate", res.get("coef", 0))
        text = f"{coef:.2f}  [{res.get('ate_ci_lower', coef):.2f}, {res.get('ate_ci_upper', coef):.2f}]"
        if res.get("ate_pvalue_week_wild") is not None:
            text += f"\nweek-bootstrap p = {res['ate_pvalue_week_wild']:.3f}"
        ax.text(max(all_vals) + 0.05 * span, y_positions[i], text, fontsize=8, va="center")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r.get("model_name", "") for r in results_list], fontsize=9)
    ax.set_xlabel("Effect of one extra self-trade on weekly gain", fontsize=10)
    ax.set_title("Main Estimates", fontsize=12, fontweight="bold")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color=COLORS["neutral"], linewidth=2.2, label="95% CI, trader-clustered"),
        Line2D([0], [0], color=COLORS["neutral"], linewidth=7, alpha=0.35,
               label="95% CI, week wild cluster bootstrap"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    png_path, _ = _save_fig(fig, "fig4_main_estimates", figdir)
    return png_path


# ──────────────────── Figure 4: CATE Distribution ────────────────────

def plot_cate_distribution(
    cate: np.ndarray,
    figdir: Optional[str] = None,
) -> str:
    """
    Distribution of the forest's predicted effects (CATE), with negative and
    positive predictions in different colours. Bin edges are aligned so that 0
    is an edge (no bar mixes signs), bar heights are % of trader-weeks, and the
    x-axis shows the central 99% (values beyond are counted in the shares and
    noted on the figure).
    """
    set_journal_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    cate = np.asarray(cate).ravel()
    n = len(cate)
    share_neg, share_pos = np.mean(cate < 0), np.mean(cate > 0)

    lo, hi = np.percentile(cate, [0.5, 99.5])
    width = (hi - lo) / 70
    edges = np.arange(np.floor(lo / width) * width, np.ceil(hi / width) * width + width, width)
    in_view = (cate >= edges[0]) & (cate <= edges[-1])
    weights = np.full(n, 100.0 / n)

    neg, pos = cate < 0, cate >= 0
    ax.hist(cate[neg & in_view], bins=edges, weights=weights[neg & in_view],
            color=COLORS["negative"], alpha=0.75, edgecolor="white", linewidth=0.3,
            label=f"Negative predicted effect ({share_neg:.1%})")
    ax.hist(cate[pos & in_view], bins=edges, weights=weights[pos & in_view],
            color=COLORS["positive"], alpha=0.75, edgecolor="white", linewidth=0.3,
            label=f"Positive predicted effect ({share_pos:.1%})")

    mean_cate, median_cate = np.mean(cate), np.median(cate)
    ax.axvline(0, color="black", linestyle="-", linewidth=1)
    ax.axvline(mean_cate, color=COLORS["primary"], linestyle="--", linewidth=1.5,
               label=f"Mean = {mean_cate:.2f}")
    ax.axvline(median_cate, color=COLORS["neutral"], linestyle=":", linewidth=1.8,
               label=f"Median = {median_cate:.2f}")

    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.18)
    ax.text(edges[0] + 0.02 * (edges[-1] - edges[0]), ymax * 1.08,
            f"{share_neg:.1%} below 0", color=COLORS["negative"],
            fontsize=11, fontweight="bold", ha="left", va="center")
    ax.text(edges[-1] - 0.02 * (edges[-1] - edges[0]), ymax * 1.08,
            f"{share_pos:.1%} above 0", color=COLORS["positive"],
            fontsize=11, fontweight="bold", ha="right", va="center")

    n_out = int((~in_view).sum())
    ax.text(0.99, 0.80,
            f"N = {n:,} held-out trader-weeks\n"
            f"x-axis: central 99% ({n_out:,} outside view,\nincluded in the shares)\n"
            f"Forest point predictions, not tested effects",
            transform=ax.transAxes, fontsize=8, va="top", ha="right", color=COLORS["neutral"])

    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel("Predicted effect of one extra self-trade on weekly gain (CATE)", fontsize=11)
    ax.set_ylabel("% of trader-weeks", fontsize=11)
    ax.set_title("Distribution of Predicted Treatment Effects", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.62))

    fig.tight_layout()
    png_path, _ = _save_fig(fig, "fig6_cate_distribution", figdir)
    return png_path


# ──────────────── Figure 5: Heterogeneity Panels ────────────────────



def plot_heterogeneity(
    cate: np.ndarray,
    tenure: np.ndarray,
    audience: np.ndarray,
    feature_names: Tuple[str, str] = (
        "Weeks between joining XTrader and 1 Jan 2013\n"
        "(positive = joined before 2013, up to 279; negative = joined during 2013)",
        "Copier audience in the first week of the window",
    ),
    figdir: Optional[str] = None,
) -> str:
    """
    Mean predicted effect (forest, held-out traders) by join-cohort decile and
    by starting copier audience. Both moderators are fixed before the analysis
    window. Error bars are 1.96 x SEM of the point predictions: they describe
    the spread of predictions, not a test (see GATES and the BLP tests).
    """
    set_journal_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cate_flat = np.asarray(cate).ravel()
    cohort = np.asarray(tenure).ravel()
    audience_flat = np.asarray(audience).ravel()
    if not (len(cate_flat) == len(cohort) == len(audience_flat)):
        raise ValueError("cate and moderators must come from the same rows")

    c_bins = pd.qcut(cohort, q=10, duplicates="drop")
    c_labels = [f"{b.left:.0f} to {b.right:.0f}" for b in c_bins.categories]
    a_edges = [-0.5, 0.5, 2.5, 10.5, 50.5, np.inf]
    a_labels = ["0", "1-2", "3-10", "11-50", "51+"]
    a_bins = pd.cut(audience_flat, a_edges, labels=a_labels)

    for ax_idx, (bins, labels, name, ax) in enumerate(zip(
            [pd.Categorical(c_bins), a_bins], [c_labels, a_labels], feature_names, axes)):
        grouped = pd.DataFrame({"cate": cate_flat, "bin": bins}).groupby("bin", observed=True)["cate"]
        means, sems, counts = grouped.mean(), grouped.sem().fillna(0), grouped.size()
        x = np.arange(len(means))
        ax.bar(x, means.values, yerr=1.96 * sems.values, color=COLORS["primary"], alpha=0.7,
               edgecolor="white", capsize=3, error_kw={"linewidth": 1, "color": COLORS["neutral"]})
        ax.axhline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.8, alpha=0.7)
        shown = [l for l in labels if l in set(means.index.astype(str))] or labels[:len(means)]
        ax.set_xticks(x)
        ax.set_xticklabels(shown, fontsize=8, rotation=45, ha="right")
        ax.set_xlabel(name, fontsize=9.5)
        ax.set_ylabel("Mean predicted effect" if ax_idx == 0 else "", fontsize=10)
        ax.text(-0.05, 1.05, chr(65 + ax_idx), transform=ax.transAxes, fontsize=14,
                fontweight="bold", va="bottom")
        ax.text(0.97, 0.03, f"N = {len(cate_flat):,} held-out trader-weeks",
                transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8,
                          edgecolor=COLORS["neutral"]))

    fig.suptitle("Predicted Effects by Pre-Window Trader Characteristics (held-out traders)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    png_path, _ = _save_fig(fig, "fig8_heterogeneity", figdir)
    return png_path


# ──────────────── Figure 6: Feature Importances ─────────────────────



# ──────────────── Figure 7: Specification Curve ─────────────────────

def plot_spec_curve(
    robustness_df: pd.DataFrame,
    figdir: Optional[str] = None,
) -> str:
    """
    Specification curve, one column per (treatment, outcome) scale, since
    estimates on different scales are not comparable. Top: sorted estimates
    with trader-clustered 95% CIs (dark) and week wild-bootstrap 95% CIs
    (light). Bottom: learner and sample indicators.
    """
    set_journal_style()
    df = robustness_df.dropna(subset=["ate"]).copy()
    scale_cols = [c for c in ["treatment_type", "outcome"] if c in df.columns]
    panels = list(df.groupby(scale_cols, sort=True)) if scale_cols else [(("all",), df)]
    n_panels = len(panels)
    fig, axes = plt.subplots(2, n_panels, figsize=(4.2 * n_panels, 6),
                             gridspec_kw={"height_ratios": [3, 1]}, squeeze=False)
    ind_dims = [d for d in ["nuisance_learner", "sample"] if d in df.columns]
    palette = [COLORS["primary"], COLORS["secondary"], COLORS["tertiary"],
               COLORS["highlight"], COLORS["neutral"], COLORS["negative"]]
    all_vals = [v for d in ind_dims for v in sorted(df[d].unique())]
    val_color = {v: palette[i % len(palette)] for i, v in enumerate(all_vals)}
    has_wild = "ci_lower_week_wild" in df.columns

    for p, (key, sub) in enumerate(panels):
        key = key if isinstance(key, tuple) else (key,)
        sub = sub.sort_values("ate").reset_index(drop=True)
        ax_top, ax_bot = axes[0, p], axes[1, p]
        x = np.arange(len(sub))
        if has_wild:
            ok = sub["ci_lower_week_wild"].notna()
            ax_top.vlines(x[ok], sub.loc[ok, "ci_lower_week_wild"], sub.loc[ok, "ci_upper_week_wild"],
                          color=COLORS["ci_fill"], linewidth=6, zorder=2)
        ax_top.vlines(x, sub["ci_lower"], sub["ci_upper"], color=COLORS["primary"],
                      linewidth=1.5, zorder=3)
        ax_top.plot(x, sub["ate"], "o", color=COLORS["primary"], markersize=5, zorder=5)
        ax_top.axhline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.8, alpha=0.7)
        ax_top.set_title(" / ".join(f"{k}" for k in key), fontsize=10, fontweight="bold")
        ax_top.set_xticks([])
        if p == 0:
            ax_top.set_ylabel("Estimate (95% CIs)", fontsize=10)

        for dim_idx, dim in enumerate(ind_dims):
            for val in sorted(df[dim].unique()):
                mask = (sub[dim] == val).values
                ax_bot.scatter(x[mask], [dim_idx] * mask.sum(), c=[val_color[val]],
                               marker="s", s=25, label=val if p == 0 else None)
        ax_bot.set_yticks(range(len(ind_dims)))
        ax_bot.set_yticklabels(ind_dims if p == 0 else [], fontsize=8)
        ax_bot.set_xlim(-0.5, len(sub) - 0.5)
        ax_bot.set_ylim(-0.5, len(ind_dims) - 0.5)
        ax_bot.set_xticks([])
        ax_bot.set_xlabel("Specs (sorted by estimate)", fontsize=8)

    axes[1, 0].legend(fontsize=7, frameon=False, ncol=5, loc="upper left",
                      bbox_to_anchor=(0, -0.35))
    fig.suptitle(f"Specification Curve by Scale ({len(df)} specifications, two-way FE). "
                 f"Dark: trader-clustered CI; light: week wild-bootstrap CI",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    png_path, _ = _save_fig(fig, "fig9_spec_curve", figdir)
    return png_path


# ──────────────── Diagnostic: Extensive vs Intensive Margin ─────────

def plot_margin_distributions(
    df: pd.DataFrame,
    variable: str = "total_self_trxn",
    total_trxn_col: str = "total_trxn",
    figdir: Optional[str] = None,
) -> str:
    """
    How often and how much traders self-trade.

    Panel A: every trader-week split into no trades / copy trades only /
    at least one self-trade. Panel B: for weeks with at least one self-trade,
    the number of self-trades the trader made in that week, grouped into
    ranges (share of those weeks). One trader-week = one trader in one week.
    """
    set_journal_style()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5.2),
                                     gridspec_kw={"width_ratios": [1, 1.25]})

    n_total = len(df)
    n_self = int((df[variable] > 0).sum())
    n_none = int((df[total_trxn_col] == 0).sum())
    n_copy_only = n_total - n_self - n_none

    labels_a = ["No trades", "Copy trades\nonly", "At least one\nself-trade"]
    counts_a = [n_none, n_copy_only, n_self]
    bars = ax_a.bar(labels_a, counts_a, width=0.6, alpha=0.85, edgecolor="white",
                    color=[COLORS["neutral"], COLORS["primary"], COLORS["secondary"]])
    for bar, count in zip(bars, counts_a):
        ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                  f"{count:,}\n({count / n_total:.1%})", ha="center", va="bottom", fontsize=9)
    ax_a.set_ylim(0, max(counts_a) * 1.2)
    ax_a.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M"))
    ax_a.set_ylabel("Number of trader-weeks", fontsize=10)
    ax_a.set_title("A   What traders did in a week", fontsize=11, fontweight="bold", loc="left")
    ax_a.text(0.0, -0.2, f"All {n_total:,} trader-weeks (one trader in one week).",
              transform=ax_a.transAxes, fontsize=8.5, color=COLORS["neutral"])

    values = df.loc[df[variable] > 0, variable]
    edges = [0, 1, 2, 5, 10, 20, 50, np.inf]
    range_labels = ["1", "2", "3–5", "6–10", "11–20", "21–50", f"51–{int(values.max())}"]
    shares = pd.cut(values, edges, labels=range_labels).value_counts(sort=False) / len(values)
    bars = ax_b.bar(range_labels, 100 * shares.values, color=COLORS["secondary"],
                    alpha=0.85, edgecolor="white", width=0.7)
    for bar, sh in zip(bars, shares.values):
        ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{sh:.0%}",
                  ha="center", va="bottom", fontsize=9)
    ax_b.set_ylim(0, 100 * shares.max() * 1.2)
    ax_b.set_xlabel("Number of self-trades the trader made in that week", fontsize=10)
    ax_b.set_ylabel("% of these trader-weeks", fontsize=10)
    ax_b.set_title("B   How many self-trades, in weeks with at least one",
                   fontsize=11, fontweight="bold", loc="left")
    ax_b.text(0.0, -0.2,
              f"{len(values):,} trader-weeks with ≥1 self-trade. Median {values.median():.0f}, "
              f"mean {values.mean():.1f}, maximum {int(values.max())} trades in one week.",
              transform=ax_b.transAxes, fontsize=8.5, color=COLORS["neutral"])

    fig.suptitle("Self-Trading Activity: How Often and How Much", fontsize=12,
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    png_path, _ = _save_fig(fig, "fig2_activity", figdir)
    return png_path


# ──────────────── Figure 8: GATES (held-out heterogeneity test) ─────

def plot_gates(gates: Dict[str, Any], figdir: Optional[str] = None) -> str:
    """
    Group average treatment effects by CATE quartile on held-out traders,
    with trader-clustered 95% CIs and the Wald test of equal effects.
    """
    set_journal_style()
    groups = gates["groups"]
    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = np.arange(len(groups))
    est = np.array([g["gate"] for g in groups])
    lo = np.array([g["ci_lower"] for g in groups])
    hi = np.array([g["ci_upper"] for g in groups])
    ax.errorbar(x, est, yerr=[est - lo, hi - est], fmt="o", color=COLORS["primary"],
                capsize=4, markersize=7, linewidth=1.5)
    ax.axhline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{g['group']}\n(pred. {g['cate_mean']:.2f})" for g in groups],
                       fontsize=8)
    ax.set_xlabel("Quartile of predicted CATE (held-out traders)", fontsize=10)
    ax.set_ylabel("Group average effect (95% CI, clustered)", fontsize=10)
    ax.set_title("GATES: Is the Predicted Heterogeneity Real?",
                 fontsize=12, fontweight="bold")
    ax.text(0.97, 0.05,
            f"Wald test, equal effects: p = {gates['wald_equal_pvalue']:.3g}\n"
            f"N = {gates['n']:,}",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.8, edgecolor=COLORS["neutral"]))

    fig.tight_layout()
    png_path, _ = _save_fig(fig, "fig7_gates", figdir)
    return png_path


# ──────────────── Figure 9: Week decomposition ─────────────────────

def plot_week_decomposition(
    week_dec: pd.DataFrame,
    week_blocks: Optional[pd.DataFrame] = None,
    figdir: Optional[str] = None,
    min_weight: float = 0.005,
) -> str:
    """
    Where the pooled estimate comes from.

    The pooled DML estimate equals sum_w weight_w x effect_w over calendar
    weeks. Panel A: each week's weight (share of residual self-trading
    variation), coloured by calendar block, with block totals. Panel B: each
    week's own effect with a 95% CI (weeks with weight >= `min_weight`), the
    block estimates (primary model refit within each block) as horizontal
    segments, and the pooled estimate as a dashed line.
    """
    set_journal_style()
    d = week_dec.sort_values("period").reset_index(drop=True)
    pooled = float((d["weight"] * d["slope"].fillna(0)).sum())

    block_spans = ([(r["block"], int(r["first_period"]), int(r["last_period"]), r)
                    for _, r in week_blocks.iterrows()] if week_blocks is not None else [])
    palette = [COLORS["primary"], COLORS["neutral"], COLORS["secondary"],
               COLORS["tertiary"], COLORS["highlight"]]
    block_color = {name: palette[i % len(palette)] for i, (name, *_ ) in enumerate(block_spans)}

    def color_of(period):
        for name, lo, hi, _ in block_spans:
            if lo <= period <= hi:
                return block_color[name]
        return COLORS["neutral"]

    colors = [color_of(p) for p in d["period"]]
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 1.5]})

    # Panel A: weights
    ax_a.bar(d["period"], 100 * d["weight"], color=colors, alpha=0.85, width=0.8)
    for name, lo, hi, _ in block_spans:
        share = 100 * d.loc[d["period"].between(lo, hi), "weight"].sum()
        top = 100 * d.loc[d["period"].between(lo, hi), "weight"].max()
        ax_a.text((lo + hi) / 2, top + 2.5, f"{name}\n{share:.0f}% of weight",
                  ha="center", va="bottom", fontsize=9, color=block_color[name],
                  fontweight="bold")
    ax_a.set_ylim(0, 100 * d["weight"].max() * 1.45)
    ax_a.set_ylabel("Weight in pooled\nestimate (%)", fontsize=10)
    ax_a.set_title("A   How much each week contributes to the pooled estimate",
                   fontsize=11, fontweight="bold", loc="left")

    # Panel B: week-specific effects with CIs, block estimates, pooled estimate
    show = d["weight"] >= min_weight
    has_ci = "slope_ci_lower" in d.columns
    for i in np.flatnonzero(show.to_numpy()):
        r = d.iloc[i]
        yerr = ([[r["slope"] - r["slope_ci_lower"]], [r["slope_ci_upper"] - r["slope"]]]
                if has_ci else None)
        ax_b.errorbar(r["period"], r["slope"], yerr=yerr, fmt="o", color=colors[i],
                      markersize=4 + 30 * np.sqrt(r["weight"]), capsize=2, linewidth=1)
    for name, lo, hi, r in block_spans:
        ci = (f" [{r['ci_lower']:.1f}, {r['ci_upper']:.1f}]"
              if "ci_lower" in r and pd.notna(r["ci_lower"]) else "")
        ax_b.hlines(r["ate"], lo - 0.4, hi + 0.4, color=block_color[name], linewidth=3,
                    alpha=0.6, label=f"{name.capitalize()} refit: {r['ate']:.1f}{ci}")
    ax_b.axhline(pooled, color="black", linestyle="--", linewidth=1,
                 label=f"Pooled estimate (all weeks): {pooled:.2f}")
    ax_b.axhline(0, color=COLORS["neutral"], linestyle=":", linewidth=1)
    ax_b.legend(loc="lower right", fontsize=9, frameon=True, facecolor="white",
                edgecolor=COLORS["neutral"], framealpha=0.95,
                title="Horizontal lines (95% CI, trader-clustered)", title_fontsize=9)

    ax_b.set_xlabel("Period (one week each; period 31 = 8-14 Jan 2013, period 75 = 12-18 Nov 2013)",
                    fontsize=10)
    ax_b.set_ylabel("Week contribution\n(effect per extra self-trade)", fontsize=10)
    ax_b.set_title(f"B   Week contributions (95% CI; weeks with ≥{min_weight:.1%} of "
                   f"weight) and block refits",
                   fontsize=11, fontweight="bold", loc="left")

    fig.suptitle("Where the Pooled Estimate Comes From", fontsize=13, fontweight="bold")
    fig.tight_layout()
    png_path, _ = _save_fig(fig, "fig5_week_decomposition", figdir)
    return png_path
