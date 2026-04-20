"""Figure generation for the paper's numerical experiments.

Produces PNG files matching the paper's scatter plots and visualizations.
Plot styling follows the old code: scatter with alpha=0.5, fontsize=14,
no titles.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.config import OUTPUT_DIR


def _savefig(fig, filename):
    """Save a figure to OUTPUT_DIR and close it."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


# ======================================================================
# Figure 3: Prototypical boxes scatter (selected only)
# ======================================================================

def plot_figure_3(selected_boxes):
    """Scatter of selected prototypical boxes: F-threshold vs P-threshold.

    Paper: Figure 3 (fig:scatter_boxes0), boxes_scatter_new.png.
    """
    x = [b.f_threshold for b in selected_boxes]
    y = [b.p_threshold for b in selected_boxes]

    fig, ax = plt.subplots()
    ax.scatter(x, y, alpha=0.5, color='red', marker='^')
    ax.set_xlabel(r'$\sigma^F$', fontsize=14)
    ax.set_ylabel(r'$\sigma^P$', fontsize=14)
    _savefig(fig, 'figure_3_prototypical_boxes.png')


# ======================================================================
# Figure EC.8: Prototypical box selection (all candidates + selected)
# ======================================================================

def plot_figure_EC8(selected_boxes, all_candidates):
    """Scatter showing all candidate boxes (faint blue) and selected (red).

    Paper: Figure EC.8 (fig:scatter_boxes), boxes_scatter.png.
    """
    all_x = [b.f_threshold for b in all_candidates]
    all_y = [b.p_threshold for b in all_candidates]

    sel_x = [b.f_threshold for b in selected_boxes]
    sel_y = [b.p_threshold for b in selected_boxes]

    fig, ax = plt.subplots()
    ax.scatter(all_x, all_y, alpha=0.1)
    ax.scatter(sel_x, sel_y, alpha=0.5, color='red', marker='^')
    ax.set_xlabel(r'$\sigma^F$', fontsize=14)
    ax.set_ylabel(r'$\sigma^P$', fontsize=14)
    _savefig(fig, 'figure_EC8_prototypical_boxes_selection.png')


# ======================================================================
# Figure EC.9a: P-ratio vs proportion with σ^P > σ^F
# ======================================================================

def plot_figure_EC9a(p_dominant_fractions, p_ratios):
    """Scatter: P-ratio of OPT vs proportion of boxes with σ^P > σ^F.

    Paper: Figure EC.9 left (fig:when_P_important), when_P_is_important.png.
    """
    fig, ax = plt.subplots()
    ax.scatter(p_dominant_fractions, p_ratios, alpha=0.5)
    ax.set_xlabel('Proportion of boxes with larger P-thresholds', fontsize=14)
    ax.set_ylabel('P-F ratio of OPT', fontsize=14)
    _savefig(fig, 'figure_EC9a_p_ratio_vs_proportion.png')


# ======================================================================
# Figure EC.9b: Weitzman suboptimality vs proportion
# ======================================================================

def plot_figure_EC9b(p_dominant_fractions, weitzman_ratios):
    """Scatter: Weitzman performance / OPT vs proportion with σ^P > σ^F.

    Paper: Figure EC.9 right (fig:when_P_important), Weitzman_performance.png.
    """
    fig, ax = plt.subplots()
    ax.scatter(p_dominant_fractions, weitzman_ratios, alpha=0.5)
    ax.set_xlabel('Proportion of boxes with larger P-thresholds', fontsize=14)
    ax.set_ylabel('Performance of WEITZMAN over OPT', fontsize=14)
    _savefig(fig, 'figure_EC9b_weitzman_vs_proportion.png')


# ======================================================================
# Figure EC.10a: Low-dispersion example
# ======================================================================

def plot_figure_EC10a(box_list):
    """Scatter of one instance's box thresholds (low dispersion).

    Paper: Figure EC.10 top-left, boxes_scatter_small_dispersion.png.
    """
    x = [b.f_threshold for b in box_list]
    y = [b.p_threshold for b in box_list]

    fig, ax = plt.subplots()
    ax.scatter(x, y, alpha=0.5, color='red', marker='^')
    ax.set_xlabel(r'$\sigma^F$', fontsize=14)
    ax.set_ylabel(r'$\sigma^P$', fontsize=14)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    _savefig(fig, 'figure_EC10a_low_dispersion.png')


# ======================================================================
# Figure EC.10b: High-dispersion example
# ======================================================================

def plot_figure_EC10b(box_list):
    """Scatter of one instance's box thresholds (high dispersion).

    Paper: Figure EC.10 top-right, boxes_scatter_big_dispersion.png.
    """
    x = [b.f_threshold for b in box_list]
    y = [b.p_threshold for b in box_list]

    fig, ax = plt.subplots()
    ax.scatter(x, y, alpha=0.5, color='red', marker='^')
    ax.set_xlabel(r'$\sigma^F$', fontsize=14)
    ax.set_ylabel(r'$\sigma^P$', fontsize=14)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    _savefig(fig, 'figure_EC10b_high_dispersion.png')


# ======================================================================
# Figure EC.10c: P-ratio vs dispersion
# ======================================================================

def plot_figure_EC10c(dispersions, p_ratios):
    """Scatter: P-ratio of OPT vs dispersion.

    Paper: Figure EC.10 bottom-left, P_F_dispersion.png.
    """
    fig, ax = plt.subplots()
    ax.scatter(dispersions, p_ratios, alpha=0.5)
    ax.set_xlabel('Dispersion', fontsize=14)
    ax.set_ylabel('P-F ratio of OPT', fontsize=14)
    _savefig(fig, 'figure_EC10c_p_ratio_vs_dispersion.png')


# ======================================================================
# Figure EC.10d: P-ratio vs number of boxes (i.i.d. experiment)
# ======================================================================

def plot_figure_EC10d(n_values, results_by_cp):
    """Line plot: P-ratio vs number of i.i.d. boxes, one line per c_P.

    Paper: Figure EC.10 bottom-right, more_boxes.png.
    Old code: OneBoxExperiment.ipynb.

    Parameters
    ----------
    n_values : list of int
        The N values used (x-axis).
    results_by_cp : dict
        Mapping {c_P: list_of_p_ratios} where each list aligns with n_values.
    """
    fig, ax = plt.subplots()
    for c_P, p_ratios in sorted(results_by_cp.items()):
        ax.plot(n_values, p_ratios, marker='o', label=f'$c_P={c_P:.1f}$')
    ax.set_xlabel('Number of boxes', fontsize=14)
    ax.set_ylabel('P-ratio', fontsize=14)
    ax.legend(fontsize=10)
    _savefig(fig, 'figure_EC10d_p_ratio_vs_num_boxes.png')
