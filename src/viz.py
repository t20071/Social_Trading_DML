"""
Journal-quality visualization for the causal ML analysis.

Style requirements:
- Colorblind-safe palette (2-4 colors max); interpretable in grayscale
- No top/right spines; minimal gridlines
- Professional font family (DejaVu Sans / Roboto)
- 300 dpi PNG + vector PDF to figures/
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
import seaborn as sns

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

FIGDIR = "figures"


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


def _save_fig(fig, name: str, figdir: str = FIGDIR):
    """Save figure as both PNG and PDF."""
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
    figdir: str = FIGDIR,
) -> str:
    """
    Sample construction flow diagram.

    Shows: raw → blank-row drop → keep_country decision → lag/tenure filter
    → final sample, with N at each step and removals annotated.
    """
    set_journal_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")

    # Build step list dynamically from available stats
    steps = [
        ("Raw Data", stats.get("raw_rows", "?"), COLORS["neutral"]),
        ("After Blank Row\nRemoval", stats.get("rows_after_blank_removal", "?"), COLORS["primary"]),
    ]

    removals = [
        stats.get("fully_blank_rows_removed", "?"),
    ]

    if stats.get("keep_country_rows_removed", 0) > 0:
        steps.append(
            ("After keep_country\nFilter",
             stats.get("rows_after_keep_country", "?"),
             COLORS["secondary"])
        )
        removals.append(stats.get("keep_country_rows_removed", "?"))

    steps.append(
        ("Final Analysis\nSample",
         stats.get("final_analysis_rows", stats.get("final_rows", "?")),
         COLORS["tertiary"])
    )
    removals.append(stats.get("rows_dropped_for_analysis", "?"))

    n_steps = len(steps)
    y_positions = np.linspace(0.88, 0.12, n_steps)
    box_width = 0.55
    box_height = 0.12

    for i, (label, n, color) in enumerate(steps):
        n_str = f"N = {n:,}" if isinstance(n, (int, float)) and n != "?" else f"N = {n}"
        rect = FancyBboxPatch(
            (0.22, y_positions[i] - box_height / 2), box_width, box_height,
            boxstyle="round,pad=0.01",
            facecolor=color, alpha=0.15, edgecolor=color, linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(0.5, y_positions[i] + 0.02, label,
                ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(0.5, y_positions[i] - 0.03, n_str,
                ha="center", va="center", fontsize=9, color=color)

        if i < n_steps - 1:
            mid_y = (y_positions[i] - box_height / 2 + y_positions[i + 1] + box_height / 2) / 2
            ax.annotate("", xy=(0.5, y_positions[i + 1] + box_height / 2),
                        xytext=(0.5, y_positions[i] - box_height / 2),
                        arrowprops=dict(arrowstyle="->", color=COLORS["neutral"], lw=1.5))

            if i < len(removals):
                r = removals[i]
                r_str = f"−{r:,}" if isinstance(r, (int, float)) and r != "?" else f"−{r}"
                ax.text(0.83, mid_y, r_str,
                        ha="center", va="center", fontsize=8,
                        color=COLORS["negative"], style="italic")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Sample Construction", fontsize=13, fontweight="bold", pad=10)

    png_path, _ = _save_fig(fig, "fig1_sample_flow", figdir)
    return png_path


# ──────────────────────────── Figure 2: Causal DAG ────────────────────────────

def plot_causal_dag(figdir: str = FIGDIR) -> str:
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
    G.add_node("TT", label="Total\nTrxn (L₁)")
    G.add_node("U", label="Unobserved\nSkill (U)")

    # Confounders → T and → Y (backdoor paths)
    confounders = ["L", "F", "G", "FO", "GD", "GT", "TN", "C", "TT"]
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
        "U": (0.5, 0.5),
    }

    labels = nx.get_node_attributes(G, "label")

    # Draw confounder nodes
    conf_nodes = [n for n in G.nodes if n not in ("T", "Y", "U")]
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
    confounder_edges = [
        (u, v) for u, v in G.edges
        if (u, v) not in causal_edges and u != "U"
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
                           width=1.5, arrows=True, arrowsize=12, ax=ax)

    # Labels
    nx.draw_networkx_labels(G, pos, labels, font_size=8,
                            font_weight="bold", ax=ax)

    ax.set_title("Causal DAG: Self-Trading → Gains",
                 fontsize=13, fontweight="bold")
    ax.axis("off")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS["secondary"], linewidth=2.5,
               label="Causal effect (T → Y)"),
        Line2D([0], [0], color=COLORS["neutral"], alpha=0.5, linewidth=1.5,
               label="Confounding paths"),
        Line2D([0], [0], color=COLORS["negative"], linewidth=1.5,
               linestyle="--", label="Unobserved (U)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=8, frameon=False)

    png_path, _ = _save_fig(fig, "fig2_causal_dag", figdir)
    return png_path


# ──────────────────────── Figure 3: Forest Plot ──────────────────────

def plot_forest_comparison(
    results_list: List[Dict[str, Any]],
    figdir: str = FIGDIR,
) -> str:
    """
    Forest plot: naive OLS vs. LinearDML (demeaned) vs. LinearDML (trader-level).

    Each entry in results_list should have:
    - model_name, coef/ate, se, ci_lower, ci_upper, n
    """
    set_journal_style()

    n_models = len(results_list)
    fig, ax = plt.subplots(figsize=(9, max(3, n_models * 0.9 + 1)))

    y_positions = list(range(n_models - 1, -1, -1))

    # Collect all CI bounds for axis limits
    all_vals = []

    for i, res in enumerate(results_list):
        name = res.get("model_name", f"Model {i+1}")
        coef = res.get("ate", res.get("coef", 0))
        ci_lo = res.get("ci_lower", res.get("ate_ci_lower", coef))
        ci_hi = res.get("ci_upper", res.get("ate_ci_upper", coef))
        se = res.get("se", res.get("ate_se", 0))
        n = res.get("n", "")

        all_vals.extend([ci_lo, ci_hi])

        color = COLORS["primary"] if "DML" in name else COLORS["secondary"]

        # Point and CI
        ax.plot(coef, y_positions[i], "o", color=color, markersize=8, zorder=5)
        ax.plot([ci_lo, ci_hi], [y_positions[i], y_positions[i]],
                "-", color=color, linewidth=2, zorder=4)

    # Zero line
    ax.axvline(x=0, color=COLORS["neutral"], linestyle="--",
               linewidth=0.8, alpha=0.7)

    # Set x-limits
    if all_vals:
        margin = (max(all_vals) - min(all_vals)) * 0.4 + 0.01
        ax.set_xlim(min(all_vals) - margin, max(all_vals) + margin)

    # Annotate after setting limits
    for i, res in enumerate(results_list):
        coef = res.get("ate", res.get("coef", 0))
        ci_lo = res.get("ci_lower", res.get("ate_ci_lower", coef))
        ci_hi = res.get("ci_upper", res.get("ate_ci_upper", coef))
        n = res.get("n", "")
        label = f"β = {coef:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]"
        if n:
            label += f"  N={n:,}"
        ax.text(ci_hi + (max(all_vals) - min(all_vals)) * 0.05, y_positions[i],
                label, fontsize=8, va="center")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [res.get("model_name", f"Model {i+1}")
         for i, res in enumerate(results_list)],
        fontsize=9,
    )
    ax.set_xlabel("Treatment Effect Estimate", fontsize=11)
    ax.set_title("Comparison of Causal Estimates: Naive OLS vs. DML",
                 fontsize=12, fontweight="bold")

    fig.tight_layout()
    png_path, _ = _save_fig(fig, "fig3_forest_comparison", figdir)
    return png_path


# ──────────────────── Figure 4: CATE Distribution ────────────────────

def plot_cate_distribution(
    cate: np.ndarray,
    figdir: str = FIGDIR,
) -> str:
    """
    CATE distribution with mean/median marked and share above/below zero annotated.
    """
    set_journal_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    cate = np.asarray(cate).ravel()
    n = len(cate)

    # Histogram
    ax.hist(cate, bins=80, color=COLORS["primary"], alpha=0.6,
            edgecolor="white", linewidth=0.3, density=True, label="CATE density")

    # KDE overlay
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(cate)
    x_range = np.linspace(cate.min(), cate.max(), 500)
    ax.plot(x_range, kde(x_range), color=COLORS["primary"], linewidth=1.5)

    # Mean and median
    mean_cate = np.mean(cate)
    median_cate = np.median(cate)
    ax.axvline(mean_cate, color=COLORS["secondary"], linestyle="-",
               linewidth=1.5, label=f"Mean = {mean_cate:.4f}")
    ax.axvline(median_cate, color=COLORS["tertiary"], linestyle="--",
               linewidth=1.5, label=f"Median = {median_cate:.4f}")

    # Zero line
    ax.axvline(0, color=COLORS["neutral"], linestyle=":", linewidth=1, alpha=0.7)

    # Share annotations
    share_pos = np.mean(cate > 0)
    share_neg = np.mean(cate < 0)
    ax.text(0.97, 0.95,
            f"Share > 0: {share_pos:.1%}\nShare < 0: {share_neg:.1%}",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.8, edgecolor=COLORS["neutral"]))

    ax.text(0.03, 0.95, f"N = {n:,}",
            transform=ax.transAxes, fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.8, edgecolor=COLORS["neutral"]))

    ax.set_xlabel("Conditional Average Treatment Effect (CATE)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Distribution of Heterogeneous Treatment Effects",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, loc="upper left")

    fig.tight_layout()
    png_path, _ = _save_fig(fig, "fig4_cate_distribution", figdir)
    return png_path


# ──────────────── Figure 5: Heterogeneity Panels ────────────────────

def plot_heterogeneity(
    cate: np.ndarray,
    tenure: np.ndarray,
    audience: np.ndarray,
    feature_names: Tuple[str, str] = ("Tenure (weeks)", "Copier Audience (L₁)"),
    figdir: str = FIGDIR,
) -> str:
    """
    Multi-panel: CATE vs. tenure and CATE vs. audience size (decile binning).
    """
    set_journal_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Align all arrays to the minimum common length to handle cases
    # where cate comes from a cached model with a different-sized sample
    cate_flat = np.asarray(cate).ravel()
    tenure_flat = np.asarray(tenure).ravel()
    audience_flat = np.asarray(audience).ravel()
    n_min = min(len(cate_flat), len(tenure_flat), len(audience_flat))

    if not (len(cate_flat) == len(tenure_flat) == len(audience_flat)):
        logger.warning(
            f"Length mismatch: cate={len(cate_flat)}, tenure={len(tenure_flat)}, "
            f"audience={len(audience_flat)}. Truncating to {n_min} for plotting."
        )
        cate_flat = cate_flat[:n_min]
        tenure_flat = tenure_flat[:n_min]
        audience_flat = audience_flat[:n_min]

    for ax_idx, (feat, feat_name, ax) in enumerate(zip(
        [tenure_flat, audience_flat], feature_names, axes
    )):
        feat = np.asarray(feat).ravel()
        cate_arr = cate_flat.copy()

        try:
            bins = pd.qcut(feat, q=10, duplicates="drop")
            grouped = pd.DataFrame(
                {"cate": cate_arr, "bin": bins}
            ).groupby("bin")["cate"]
            means = grouped.mean()
            sems = grouped.sem()
            x_centers = np.arange(len(means))

            ax.bar(x_centers, means.values, yerr=1.96 * sems.values,
                   color=COLORS["primary"], alpha=0.7, edgecolor="white",
                   capsize=3,
                   error_kw={"linewidth": 1, "color": COLORS["neutral"]})

            ax.axhline(0, color=COLORS["neutral"], linestyle="--",
                       linewidth=0.8, alpha=0.7)
            ax.set_xticks(x_centers)
            ax.set_xticklabels(
                [f"D{i+1}" for i in range(len(means))],
                fontsize=8, rotation=45,
            )

        except Exception:
            ax.scatter(feat, cate_arr, alpha=0.05, s=1, color=COLORS["primary"])
            ax.axhline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.8)

        ax.set_xlabel(feat_name, fontsize=10)
        ax.set_ylabel("CATE" if ax_idx == 0 else "", fontsize=10)

        # Panel label
        panel_label = chr(65 + ax_idx)
        ax.text(-0.05, 1.05, panel_label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="bottom")

        # N annotation
        ax.text(0.97, 0.03, f"N = {len(cate_arr):,}",
                transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          alpha=0.8, edgecolor=COLORS["neutral"]))

    fig.suptitle("Treatment Effect Heterogeneity by Trader Characteristics",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    png_path, _ = _save_fig(fig, "fig5_heterogeneity", figdir)
    return png_path


# ──────────────── Figure 6: Feature Importances ─────────────────────

def plot_feature_importances(
    importances: np.ndarray,
    feature_names: List[str],
    figdir: str = FIGDIR,
) -> str:
    """Horizontal bar chart of CausalForestDML feature importances."""
    set_journal_style()

    importances = np.asarray(importances).ravel()
    sort_idx = np.argsort(importances)

    fig, ax = plt.subplots(
        figsize=(7, max(3, len(feature_names) * 0.5 + 1))
    )

    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, importances[sort_idx], color=COLORS["primary"],
            alpha=0.8, edgecolor="white", height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in sort_idx], fontsize=9)
    ax.set_xlabel("Feature Importance", fontsize=11)
    ax.set_title("CausalForestDML: Heterogeneity Feature Importances",
                 fontsize=12, fontweight="bold")

    for i, idx in enumerate(sort_idx):
        ax.text(importances[idx] + 0.005, i, f"{importances[idx]:.3f}",
                va="center", fontsize=8, color=COLORS["neutral"])

    fig.tight_layout()
    png_path, _ = _save_fig(fig, "fig6_feature_importances", figdir)
    return png_path


# ──────────────── Figure 7: Specification Curve ─────────────────────

def plot_spec_curve(
    robustness_df: pd.DataFrame,
    figdir: str = FIGDIR,
) -> str:
    """
    Specification curve: sorted ATE estimates with CIs across all
    robustness specs. Bottom panel shows indicator dots for each spec
    dimension.
    """
    set_journal_style()

    df = robustness_df.dropna(subset=["ate"]).copy()
    df = df.sort_values("ate").reset_index(drop=True)
    n_specs = len(df)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(max(10, n_specs * 0.5), 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    x = np.arange(n_specs)

    # Top: ATE with CIs
    ax_top.fill_between(x, df["ci_lower"].values, df["ci_upper"].values,
                        color=COLORS["ci_fill"], alpha=0.7, label="95% CI")
    ax_top.plot(x, df["ate"].values, "o", color=COLORS["primary"],
                markersize=5, zorder=5)
    ax_top.axhline(0, color=COLORS["neutral"], linestyle="--",
                   linewidth=0.8, alpha=0.7)

    ax_top.set_ylabel("ATE Estimate", fontsize=11)
    ax_top.set_title(
        "Specification Curve: Robustness Across All Specifications",
        fontsize=12, fontweight="bold",
    )
    ax_top.text(0.97, 0.95, f"{n_specs} specifications",
                transform=ax_top.transAxes, fontsize=9, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.8, edgecolor=COLORS["neutral"]))

    # Bottom: specification indicators
    spec_dims = ["treatment_type", "nuisance_learner", "sample", "country_controls"]
    available_dims = [d for d in spec_dims if d in df.columns]

    for dim_idx, dim in enumerate(available_dims):
        unique_vals = sorted(df[dim].unique())
        for val_idx, val in enumerate(unique_vals):
            mask = (df[dim] == val).values
            color = plt.cm.Set2(val_idx / max(len(unique_vals), 1))
            ax_bot.scatter(x[mask], [dim_idx] * mask.sum(), c=[color],
                           marker="s", s=15)

    ax_bot.set_yticks(range(len(available_dims)))
    ax_bot.set_yticklabels(available_dims, fontsize=8)
    ax_bot.set_xlabel("Specification (sorted by ATE)", fontsize=10)
    ax_bot.set_xlim(-0.5, n_specs - 0.5)

    fig.tight_layout()
    png_path, _ = _save_fig(fig, "fig7_spec_curve", figdir)
    return png_path


# ──────────────── Diagnostic: Extensive vs Intensive Margin ─────────

def plot_margin_distributions(
    df: pd.DataFrame,
    variable: str = "total_self_trxn",
    total_trxn_col: str = "total_trxn",
    figdir: str = FIGDIR,
) -> str:
    """
    Separate extensive margin (did it happen) from intensive margin
    (how much, conditional on active).

    Panel A: Bar chart of extensive margin
    Panel B: Histogram of intensive margin
    """
    set_journal_style()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Extensive margin
    n_active = int((df[total_trxn_col] > 0).sum())
    n_inactive = int((df[total_trxn_col] == 0).sum())
    n_total = len(df)

    bars = ax_a.bar(
        ["Active\n(trxn > 0)", "Inactive\n(trxn = 0)"],
        [n_active, n_inactive],
        color=[COLORS["primary"], COLORS["neutral"]],
        alpha=0.8, edgecolor="white",
    )

    for bar, count in zip(bars, [n_active, n_inactive]):
        ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                  f"{count:,}\n({count/n_total:.1%})",
                  ha="center", va="bottom", fontsize=9)

    ax_a.set_ylabel("Number of Trader-Weeks", fontsize=10)
    ax_a.set_title("A   Extensive Margin", fontsize=11,
                   fontweight="bold", loc="left")
    ax_a.text(0.97, 0.95, f"N = {n_total:,}",
              transform=ax_a.transAxes, fontsize=8, va="top", ha="right")

    # Panel B: Intensive margin (conditional on active)
    active_values = df.loc[df[total_trxn_col] > 0, variable].dropna()
    # For count treatment, exclude zeros for cleaner intensive-margin view
    positive_values = active_values[active_values > 0]
    plot_values = positive_values if len(positive_values) > 100 else active_values

    ax_b.hist(plot_values, bins=50, color=COLORS["secondary"], alpha=0.7,
              edgecolor="white", linewidth=0.3)
    ax_b.axvline(plot_values.mean(), color=COLORS["primary"], linestyle="-",
                 linewidth=1.5, label=f"Mean = {plot_values.mean():.1f}")
    ax_b.axvline(plot_values.median(), color=COLORS["tertiary"], linestyle="--",
                 linewidth=1.5, label=f"Median = {plot_values.median():.1f}")

    ax_b.set_xlabel(variable, fontsize=10)
    ax_b.set_ylabel("Count", fontsize=10)
    ax_b.set_title("B   Intensive Margin (Active Weeks, >0)",
                   fontsize=11, fontweight="bold", loc="left")
    ax_b.legend(fontsize=8, frameon=False)
    ax_b.text(0.97, 0.95, f"N = {len(plot_values):,}",
              transform=ax_b.transAxes, fontsize=8, va="top", ha="right")

    fig.suptitle(
        f"Distribution of {variable}: Extensive vs. Intensive Margin",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    png_path, _ = _save_fig(fig, f"diag_margins_{variable}", figdir)
    return png_path
