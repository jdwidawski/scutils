from typing import List, Optional, Union

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def volcano_plot(
    df: pd.DataFrame,
    pval_col: str = "pvals_adj",
    lfc_col: str = "logfoldchanges",
    pval_cutoff: float = 0.01,
    lfc_cutoff: float = 1.0,
    xlim: Optional[float] = None,
    ylim: Optional[float] = None,
    top_n_up: Optional[int] = 5,
    top_n_down: Optional[int] = 5,
    extra_genes: Optional[List[str]] = None,
    annot_min_lfc: Optional[float] = None,
    annot_min_pval: Optional[float] = None,
    annot_sort_by: str = "pval",
    annotation_mode: str = "offset",
    annotation_offset: float = 0.3,
    annotation_v_offset: float = 2.0,
    annotation_fontsize: float = 12,
    figsize: tuple[float, float] = (8.0, 6.0),
) -> plt.Figure:
    """Create a volcano plot from differential expression results.

    Produces a scatter plot of $-\\log_{10}$(p-value) vs log-fold-change,
    with optional gene-name annotations for the most significant up-/down-
    regulated genes.

    Args:
        df: DataFrame containing differential expression results.  Must have
            columns for p-values, log-fold-changes and, optionally, gene names
            in a column named ``"names"`` or the DataFrame index.
        pval_col: Column name for (adjusted) p-values.  Defaults to
            ``"pvals_adj"``.
        lfc_col: Column name for log-fold-changes.  Defaults to
            ``"logfoldchanges"``.
        pval_cutoff: P-value threshold for significance.  Defaults to
            ``0.01``.
        lfc_cutoff: Absolute log-fold-change threshold for significance.
            Defaults to ``1.0``.
        xlim: Symmetric x-axis limit.  When ``None`` it is computed
            automatically from the data.
        ylim: Upper y-axis limit.  When ``None`` it is computed automatically
            from the data.
        top_n_up: Number of top upregulated genes to annotate.  Set to ``0``
            or ``None`` to skip.  Defaults to ``5``.
        top_n_down: Number of top downregulated genes to annotate.  Set to
            ``0`` or ``None`` to skip.  Defaults to ``5``.
        extra_genes: Additional gene names to annotate regardless of their
            significance ranking.
        annot_min_lfc: Minimum absolute log-fold-change required for
            annotation eligibility.  ``None`` or ``0`` disables this filter.
        annot_min_pval: Maximum adjusted p-value required for annotation
            eligibility.  ``None`` or ``0`` disables this filter.
        annot_sort_by: Ranking criterion when selecting ``top_n_up`` /
            ``top_n_down`` genes.  ``"pval"`` sorts by most-significant
            p-value; ``"lfc"`` sorts by largest absolute fold-change.
            Defaults to ``"pval"``.
        annotation_mode: Label-placement strategy.  ``"offset"`` places each
            label a fixed distance from its dot; ``"vertical"`` aligns all
            labels on a single vertical line per direction.  Defaults to
            ``"offset"``.
        annotation_offset: Horizontal distance (data units) between a dot and
            its label.  Defaults to ``0.3``.
        annotation_v_offset: Vertical distance (data units) between a dot and
            its label.  Defaults to ``2.0``.
        annotation_fontsize: Font size for annotation labels.  Defaults to
            ``12``.
        figsize: ``(width, height)`` of the figure in inches.  Defaults to
            ``(8.0, 6.0)``.

    Returns:
        The matplotlib figure containing the volcano plot.

    Raises:
        ValueError: If *annot_sort_by* is not ``"pval"`` or ``"lfc"``.

    Example:
        >>> fig = volcano_plot(de_df, pval_cutoff=0.05, lfc_cutoff=0.5)
        >>> fig.savefig("volcano.png", dpi=150)
    """
    valid_sort = {"pval", "lfc"}
    if annot_sort_by not in valid_sort:
        raise ValueError(
            f"Invalid annot_sort_by={annot_sort_by!r}. "
            f"Must be one of {sorted(valid_sort)}."
        )

    valid_modes = {"vertical", "offset"}
    if annotation_mode not in valid_modes:
        raise ValueError(
            f"Invalid annotation_mode={annotation_mode!r}. "
            f"Must be one of {sorted(valid_modes)}."
        )

    if extra_genes is None:
        extra_genes = []
    top_n_up = top_n_up or 0
    top_n_down = top_n_down or 0

    df = df[(df[lfc_col] < 10) & (df[lfc_col] > -10)]
    df = df.copy()
    if "names" not in df.columns.tolist():
        df = df.reset_index().rename(columns={"index": "names"})

    fig, axs = plt.subplots(1, 1, figsize=figsize)

    # Annotation-specific cutoffs (fall back to significance cutoffs) ------
    _annot_lfc = annot_min_lfc if annot_min_lfc else lfc_cutoff
    _annot_pval = annot_min_pval if annot_min_pval else pval_cutoff

    # Define top DE genes --------------------------------------------------
    up_mask = (df[lfc_col] > _annot_lfc) & (df[pval_col] < _annot_pval)
    down_mask = (df[lfc_col] < -_annot_lfc) & (df[pval_col] < _annot_pval)

    if annot_sort_by == "pval":
        top_up: List[str] = (
            df[up_mask]
            .sort_values(by=pval_col, ascending=True)["names"]
            .tolist()[:top_n_up]
        )
        top_down: List[str] = (
            df[down_mask]
            .sort_values(by=pval_col, ascending=True)["names"]
            .tolist()[:top_n_down]
        )
    else:  # "lfc"
        top_up: List[str] = (
            df[up_mask]
            .sort_values(by=lfc_col, ascending=False)["names"]
            .tolist()[:top_n_up]
        )
        top_down: List[str] = (
            df[down_mask]
            .sort_values(by=lfc_col, ascending=True)["names"]
            .tolist()[:top_n_down]
        )

    # Merge with extra genes (avoid duplicates, keep order) ---------------
    genes_to_annotate: List[str] = list(dict.fromkeys(top_up + top_down + extra_genes))

    # Assign DE category ---------------------------------------------------
    df["de_category"] = [
        "Upregulated"
        if (pval < pval_cutoff and lfc > lfc_cutoff)
        else (
            "Downregulated"
            if (pval < pval_cutoff and lfc < -lfc_cutoff)
            else "Not significant"
        )
        for pval, lfc in df[[pval_col, lfc_col]].values
    ]
    df["de_category"] = df["de_category"].astype("category")

    # Transform p-values to -log10 scale -----------------------------------
    # Replace exact zeros with a tiny floor so -log10 stays finite
    df[pval_col] = df[pval_col].replace(0, 1e-300)
    df[pval_col] = -np.log10(df[pval_col])

    # Plot scatter ---------------------------------------------------------
    ax = sns.scatterplot(
        data=df,
        y=pval_col,
        x=lfc_col,
        hue="de_category",
        s=75,
        palette={
            "Downregulated": "blue",
            "Not significant": "gray",
            "Upregulated": "red",
        },
        ax=axs,
    )

    # Figure styling -------------------------------------------------------
    automatic_xlim = df[lfc_col].abs().fillna(0).max() * 1.25
    automatic_ylim = df[pval_col].dropna().abs().max() * 1.25

    if xlim is None:
        xlim = automatic_xlim
    if ylim is None:
        ylim = automatic_ylim

    ax.set_ylim(0, ylim)
    ax.set_xlim(-xlim, xlim)

    # Annotate genes -------------------------------------------------------
    arrow_kw = dict(arrowstyle="->", color="black", lw=0.8, zorder=4)

    # Use ax.annotate() so that the arrow *always* originates at the
    # scatter point (xy) regardless of where the text (xytext) ends up.
    # Overlap is resolved by repelling y-positions per side while
    # preserving the original y-ordering (higher y_val → higher label).
    annot_right: list = []  # positive fold-change labels
    annot_left: list = []   # negative fold-change labels

    for gene in genes_to_annotate:
        row = df.loc[df["names"] == gene, ["names", lfc_col, pval_col]]
        if row.empty:
            continue
        name, x_val, y_val = row.values[0]
        sign = 1 if x_val >= 0 else -1

        if annotation_mode == "vertical":
            text_x = xlim * 0.75 if x_val >= 0 else -xlim * 0.75
        else:  # "offset"
            text_x = x_val + sign * annotation_offset

        text_y = y_val + annotation_v_offset

        entry = {
            "name": name,
            "x_val": float(x_val),
            "y_val": float(y_val),
            "text_x": float(text_x),
            "text_y": float(text_y),
            "sign": sign,
        }
        (annot_right if sign > 0 else annot_left).append(entry)

    # Repel y-positions within each side so labels don't overlap.
    # Sort by *scatter-point* y_val so that higher points keep higher
    # labels, then push apart any labels that are too close.
    min_gap = ylim * 0.045
    for group in (annot_left, annot_right):
        group.sort(key=lambda a: a["y_val"])
        for i in range(1, len(group)):
            if group[i]["text_y"] - group[i - 1]["text_y"] < min_gap:
                group[i]["text_y"] = group[i - 1]["text_y"] + min_gap

    # Determine the maximum text_y / text_x to expand axes if needed
    all_entries = annot_left + annot_right
    if all_entries:
        max_text_y = max(e["text_y"] for e in all_entries)
        max_text_x = max(abs(e["text_x"]) for e in all_entries)
        # Add a small padding so text doesn't sit at the very edge
        y_pad = ylim * 0.08
        x_pad = xlim * 0.08
        if max_text_y + y_pad > ylim:
            ylim = max_text_y + y_pad
            ax.set_ylim(0, ylim)
        if max_text_x + x_pad > xlim:
            xlim = max_text_x + x_pad
            ax.set_xlim(-xlim, xlim)

    for entry in all_entries:
        ax.annotate(
            entry["name"],
            xy=(entry["x_val"], entry["y_val"]),
            xytext=(entry["text_x"], entry["text_y"]),
            arrowprops=arrow_kw,
            fontsize=annotation_fontsize,
            ha="left" if entry["sign"] > 0 else "right",
            va="center",
            zorder=5,
        )

    # Legend & labels -------------------------------------------------------
    ax.get_legend().remove()
    ax.set_ylabel("-log10(FDR)", fontsize=18)
    ax.set_xlabel("Log2(Fold Change)", fontsize=18)
    plt.tight_layout()

    color_map = {
        "Upregulated": "red",
        "Downregulated": "blue",
        "Not significant": "gray",
    }

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor=color,
            color=color,
            label=label,
        )
        for label, color in color_map.items()
    ]

    fig.axes[0].legend(
        handles=legend_elements,
        loc="best",
        bbox_to_anchor=(1.35, 1),
        shadow=True,
        title="Annotation legend",
        title_fontsize=15,
        fontsize=12,
        labelspacing=0.75,
        borderpad=0.75,
    )

    return fig