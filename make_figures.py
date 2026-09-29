# Rebuild the plots from whatever is sitting in results/.

from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

rows = json.loads((ROOT / 'results' / 'metrics.json').read_text())
R = {r['key']: r for r in rows}
control = json.loads((ROOT / 'results' / 'matched_budget.json').read_text())

# shared plot palette
ink = '#1F2A36'
blue = '#0069B4'
slate = '#5B6875'
grey = '#AEB9C4'
grid_c = '#E3E8ED'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['FreeSans', 'Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 23,
    'ytick.labelsize': 23,
    'text.color': ink,
    'axes.labelcolor': ink,
    'xtick.color': slate,
    'ytick.color': slate,
    'axes.edgecolor': grid_c,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

MM = 1 / 25.4


def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color=grid_c, linewidth=0.8)
    ax.tick_params(length=0, pad=8)


def save_fig(fig, name):
    fig.savefig(FIG_DIR / (name + '.pdf'))
    fig.savefig(FIG_DIR / (name + '.png'), dpi=150)
    plt.close(fig)

def plot_overview():
    # left = pair acc (LR / SVM), right = orig vs revised acc for LR bigrams
    fig = plt.figure(figsize=(350 * MM, 170 * MM))
    ax1 = fig.add_axes([0.085, 0.19, 0.38, 0.65])
    ax2 = fig.add_axes([0.60, 0.19, 0.37, 0.65])

    legend = [
        Patch(facecolor=grey, label='Trained on originals'),
        Patch(facecolor=blue, label='Trained on originals + revisions'),
    ]
    fig.legend(
        handles=legend,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=23,
        handlelength=1,
        handletextpad=0.5,
        columnspacing=1.8,
    )

    for j, reg in enumerate(['original', 'augmented']):
        vals = [100 * R[f'{alg}_bigrams_{reg}']['pair_accuracy'] for alg in ['LR', 'SVM']]
        cis = np.array([
            R[f'{alg}_bigrams_{reg}']['pair_accuracy_ci'] for alg in ['LR', 'SVM']
        ]) * 100
        x = np.arange(2) + (j - 0.5) * 0.36
        ax1.bar(x, vals, width=0.31, color=[grey, blue][j], zorder=3)
        lo = np.array(vals) - cis[:, 0]
        hi = cis[:, 1] - np.array(vals)
        ax1.errorbar(
            x, vals, yerr=np.vstack([lo, hi]),
            fmt='none', ecolor=ink, capsize=4, linewidth=1.3, zorder=4,
        )
        for xx, v, top in zip(x, vals, cis[:, 1]):
            ax1.text(xx, top + 4, f'{v:.1f}', ha='center', fontsize=24, color=ink)

    ax1.set_xticks([0, 1], ['Logistic\nregression', 'Linear\nSVM'])
    ax1.set_ylim(0, 100)
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.set_ylabel('Both reviews correct (%)', labelpad=10)

    for reg, color, marker, ls, dy in [
        ('original', grey, 's', '--', -8),
        ('augmented', blue, 'o', '-', 5),
    ]:
        r = R[f'LR_bigrams_{reg}']
        vals = [100 * r['orig_accuracy'], 100 * r['revised_accuracy']]
        ax2.plot([0, 1], vals, marker=marker, linestyle=ls, color=color, lw=2.6, ms=10)
        for x, y in enumerate(vals):
            col = slate if reg == 'original' else blue
            ax2.text(x, y + dy, f'{y:.1f}', ha='center', color=col, fontsize=24)

    ax2.set_xlim(-0.22, 1.22)
    ax2.set_ylim(40, 100)
    ax2.set_xticks([0, 1], ['Original\nreviews', 'Revised\nreviews'])
    ax2.set_yticks([40, 60, 80, 100])
    ax2.set_ylabel('Review accuracy (%)', labelpad=10)

    style_axes(ax1)
    style_axes(ax2)
    save_fig(fig, 'results_overview')


def plot_ablation():
    fig, ax = plt.subplots(figsize=(166 * MM, 140 * MM))
    fig.subplots_adjust(left=0.47, right=0.99, bottom=0.05, top=0.79)

    feats = ['unigrams', 'negation', 'bigrams']
    m = np.array([
        [R[f'LR_{f}_{reg}']['pair_accuracy'] * 100 for reg in ['original', 'augmented']]
        for f in feats
    ])

    cmap = LinearSegmentedColormap.from_list(
        'poster', ['#C0D2E4', '#ABC5DD', '#0165AD', '#035EA1']
    )
    ax.imshow(m, cmap=cmap, vmin=38.7, vmax=78.5, aspect='auto')
    ax.set_xticks([0, 1], ['Originals', 'Originals\n+ revisions'])
    ax.xaxis.tick_top()
    ax.tick_params(axis='both', length=0, pad=10)
    ax.set_yticks([0, 1, 2], ['Unigrams', '+ negation\nmarkers', 'Unigrams\n+ bigrams'])

    for i in range(3):
        for j in range(2):
            col = 'white' if m[i, j] > 65 else ink
            ax.text(j, i, f'{m[i, j]:.1f}', ha='center', va='center', fontsize=28, color=col)

    for sp in ax.spines.values():
        sp.set_visible(False)
    save_fig(fig, 'ablation')


def plot_matched_budget():
    # each blue dot is one of the 5 subset seeds
    fig, ax = plt.subplots(figsize=(166 * MM, 140 * MM))
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.24, top=0.92)

    base = np.array([r['pair_accuracy'] * 100 for r in control if r['training'] == 'original'])
    mix = np.array([r['pair_accuracy'] * 100 for r in control if r['training'] == 'mixed'])

    ax.scatter([0], [base[0]], s=110, color=grey, marker='s', zorder=4)
    ax.scatter(1 + np.linspace(-0.09, 0.09, len(mix)), mix, s=80, color=blue, zorder=4)
    ax.hlines(mix.mean(), 0.78, 1.22, color=ink, lw=2.4, zorder=5)
    ax.text(0, base[0] + 6, f'{base[0]:.1f}', ha='center', color=slate)
    ax.text(1, mix.max() + 6, f'{mix.mean():.1f} (mean)', ha='center', color=blue)

    ax.set_xlim(-0.42, 1.50)
    ax.set_ylim(25, 93)
    ax.set_yticks([25, 50, 75])
    ax.set_xticks([0, 1], ['1,706\noriginals', '853 pairs\n(1,706 texts)'])
    ax.set_ylabel('Both reviews correct (%)', labelpad=9)
    style_axes(ax)
    save_fig(fig, 'matched_budget')


if __name__ == '__main__':
    plot_overview()
    plot_ablation()
    plot_matched_budget()
    print('Saved vector figures to', FIG_DIR)
