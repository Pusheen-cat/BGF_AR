"""FFT analysis and figures of the recorded gradient signals.

Compares the RAW gradient signal of the Adam run against the FILTERED
(actually applied) gradient signal of the BGF run - both recorded with
``run_grad_training.py`` (500 sampled coordinates, every training step) - in
the frequency domain, for two training phases:

  * Training phase:        the --window-step (default 10,000) interval starting
                           at step --train_start (default 100).
  * Generalization phase:  the --window-step interval starting at the first
                           step at which the trailing --ma_window-step (50)
                           moving average of the TRAINING accuracy reaches
                           --acc_threshold (95%).  The onset is computed
                           separately for each run from its own log, so the
                           two runs generally use different window starts.

Four Adam-vs-BGF figures are written to --out_dir (window starts embedded in
the names, Adam first, BGF second):

  FFT_{s}_{s}.pdf           training phase, per-coordinate FFT line plots
                            (Baseline vs BGF) for the --coords coordinates
  FFT_{sb}_{so}.pdf         the same for the generalization phase
  boxFFTnorm_{s}_{s}.pdf    training phase, box plots of the per-coordinate
                            band power fractions (Low / High / Very-high),
                            each band normalised by its Adam median
  boxFFTnorm_{sb}_{so}.pdf  the same for the generalization phase

Three further figures compare the TRAINING phase against the GENERALIZATION
phase WITHIN the Adam run alone (its raw gradient signal; BGF is not involved).
Window starts in the names are training first, generalization second:

  js_overlay_{tb}_{tg}.pdf            per-coordinate |FFT| overlays (red =
                                      training, blue = generalization) plus an
                                      all-coordinate aggregate panel
  js_box_norm_{tb}_{tg}.pdf           band-power box plots, each band
                                      normalised by its Training median
  js_thresholds_simple_{tb}_{tg}.pdf  mean power-fraction spectra of the two
                                      phases with the per-log-bin gen/train
                                      power ratio drawn as gray bars behind
                                      them (bar baseline = 1 aligned with the
                                      level where the two spectra cross)

The js_* figures follow the original single-run pipeline: they operate on the
raw Adam signal with per-window z-scoring only (no cross-coordinate L2 step -
that step exists to make two DIFFERENT runs comparable, which is unnecessary
within one run).

Analysis pipeline (identical to the paper's):
  1. per-step cross-coordinate L2 normalisation of each signal matrix
     (removes the overall gradient-magnitude trend over training);
  2. per-coordinate z-scoring of each analysis window (equal total spectral
     power for every signal, by Parseval - the comparison becomes purely
     about how energy is distributed across frequency);
  3. line plots show the 10-tap-smoothed FFT magnitude; box plots use the
     FRACTION of total spectral POWER summed inside three frequency bands
     Low < 10^t1 | High 10^t1..10^t2 | Very-high >= 10^t2  (t1 = -1.7,
     t2 = -1.1), with paired Wilcoxon significance across the coordinates.

Prerequisite:  python run_grad_training.py --cuda 0    (both runs finished)
Then:          python plot_grad_fft.py
"""
import argparse
import glob
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))

BANDS = ["Low", "High", "Very-high"]
C_ADAM = "#D55E00"   # vermillion (Okabe-Ito, colour-blind safe)
C_BGF = "#0072B2"    # blue

# The 12 coordinates shown in the paper's per-parameter FFT figure (indices
# into the 500 sampled coordinates; the sampling is deterministic, so they
# denote the same parameters in every run of the same architecture).
COORDS_12 = [35, 225, 337, 184, 467, 14, 62, 45, 125, 233, 390, 391]

parser = argparse.ArgumentParser(description='FFT figures of the recorded gradient signals (Adam vs BGF)')
parser.add_argument('--save_dir', default=os.path.join(HERE, 'results_grad'), type=str,
                    help='results root written by run_grad_training.py')
parser.add_argument('--adam_run', default=None, type=str,
                    help='Adam run directory (default: unique baseline-* run under --save_dir)')
parser.add_argument('--bgf_run', default=None, type=str,
                    help='BGF run directory (default: unique ours_balance-* run under --save_dir)')
parser.add_argument('--out_dir', default=os.path.join(HERE, 'figures'), type=str)
parser.add_argument('--window', default=10_000, type=int, help='analysis-window length in steps')
parser.add_argument('--train_start', default=100, type=int,
                    help='window start (step) of the training phase')
parser.add_argument('--ma_window', default=50, type=int,
                    help='moving-average window (steps) for the generalization onset')
parser.add_argument('--acc_threshold', default=0.95, type=float,
                    help='training-accuracy threshold defining the generalization onset')
parser.add_argument('--gen_start_adam', default=None, type=int,
                    help='override the computed generalization window start of the Adam run')
parser.add_argument('--gen_start_bgf', default=None, type=int,
                    help='override the computed generalization window start of the BGF run')
parser.add_argument('--t1', default=-1.7, type=float, help='Low | High band boundary (log10 Hz)')
parser.add_argument('--t2', default=-1.1, type=float, help='High | Very-high band boundary (log10 Hz)')
parser.add_argument('--coords', nargs='+', type=int, default=COORDS_12,
                    help='coordinates shown in the FFT line-plot figures')
parser.add_argument('--smooth', default=10, type=int, help='smoothing taps of the FFT line plots')
args = parser.parse_args()


# --------------------------------------------------------------------------- #
# Loading and alignment
# --------------------------------------------------------------------------- #
def find_run(prefix, override):
    """The unique run directory under --save_dir whose folder starts with prefix."""
    if override is not None:
        return os.path.normpath(os.path.abspath(override))
    hits = [os.path.dirname(p) for p in
            glob.glob(os.path.join(args.save_dir, prefix + '*', '*', '*', '*', 'grad_signal.pkl'))]
    if len(hits) != 1:
        raise SystemExit(
            f'Expected exactly one {prefix}* run with a grad_signal.pkl under {args.save_dir}, '
            f'found {len(hits)}: {hits}\nUse --adam_run / --bgf_run to select one explicitly.')
    return hits[0]


def norml2(x):
    """Per-training-step (column) L2 normalisation across the 500 coordinates.

    Removes the overall gradient-magnitude envelope (which decays by orders of
    magnitude over training and differs between optimizers) while leaving the
    per-coordinate temporal pattern - the thing we FFT - intact.
    """
    col_l2 = np.linalg.norm(x, ord=2, axis=0, keepdims=True)
    col_l2[col_l2 == 0] = 1
    return x / col_l2


def load_signals(adam_run, bgf_run):
    """Returns (adam_raw, bgf_filtered, adam_full, positions, align).

    ``adam_raw`` and ``bgf_filtered`` are L2-normalised and aligned so column 0
    == training step ``align`` (used by the Adam-vs-BGF figures).  ``adam_full``
    is the UN-normalised full Adam raw matrix with column i == training step i
    (used by the single-run js_* figures), and ``positions`` its coordinate
    sampling positions (for labelling).
    """
    with open(os.path.join(adam_run, 'grad_signal.pkl'), 'rb') as f:
        adam_pkl = pickle.load(f)
    with open(os.path.join(bgf_run, 'grad_signal.pkl'), 'rb') as f:
        bgf_pkl = pickle.load(f)
    if bgf_pkl['filtered'] is None:
        raise SystemExit(f'{bgf_run}/grad_signal.pkl has no filtered signal - is it really a BGF run?')
    align = bgf_pkl['filtered_start_step']       # filter warm-up (default 100)
    adam_full = np.asarray(adam_pkl['raw'], dtype=np.float64)
    adam = norml2(adam_full)[:, align:]
    bgf = norml2(np.asarray(bgf_pkl['filtered'], dtype=np.float64))
    return adam, bgf, adam_full, adam_pkl['positions'], align


def training_accuracy_series(run_dir):
    """(steps, t_a) arrays from the run's logs file, ordered by step."""
    with open(os.path.join(run_dir, 'logs')) as f:
        logs = json.load(f)
    step_log = logs['step_log']
    steps = sorted(int(k) for k in step_log if str(k).isdigit())
    t_a = np.array([step_log[str(s)]['t_a'] for s in steps])
    return np.array(steps), t_a


def find_generalization_start(run_dir):
    """First step at which the trailing --ma_window-step moving average of the
    training accuracy reaches --acc_threshold."""
    steps, t_a = training_accuracy_series(run_dir)
    if len(t_a) < args.ma_window:
        return None
    ma = np.convolve(t_a, np.ones(args.ma_window) / args.ma_window, mode='valid')
    hits = np.where(ma >= args.acc_threshold)[0]
    if len(hits) == 0:
        return None
    return int(steps[hits[0] + args.ma_window - 1])


def window(signal, start_step, align, name):
    """The (n_coords, --window) analysis window starting at training step start_step."""
    lo = start_step - align
    hi = lo + args.window
    if lo < 0 or hi > signal.shape[1]:
        raise SystemExit(
            f'{name}: window [{start_step}, {start_step + args.window}) is outside the recorded '
            f'signal (steps [{align}, {align + signal.shape[1]})). Use shorter --window or '
            f'--gen_start_* overrides.')
    return signal[:, lo:hi]


# --------------------------------------------------------------------------- #
# Spectra
# --------------------------------------------------------------------------- #
def zscore(W):
    """Per-coordinate z-scoring: equal total spectral power for every signal."""
    return (W - W.mean(1, keepdims=True)) / (W.std(1, keepdims=True) + 1e-5)


def power_fractions(W):
    """Per-coordinate fraction-of-total-spectral-power spectra of a window.

    Z-score -> real FFT -> power |X|^2 (DC dropped) -> divide by the total, so
    each spectrum is a probability distribution over frequency and the three
    band fractions below sum to 1 (a genuine energy partition, by Parseval).
    """
    F = np.fft.rfft(zscore(W), axis=1)
    P = (np.abs(F) ** 2)[:, 1:]
    tot = P.sum(1, keepdims=True)
    tot[tot == 0] = 1.0
    return P / tot


def band_fractions(pfrac, log10f):
    """(n_coords, 3): share of spectral power in Low / High / Very-high."""
    low = log10f < args.t1
    high = (log10f >= args.t1) & (log10f < args.t2)
    vhf = log10f >= args.t2
    return np.stack([pfrac[:, low].sum(1), pfrac[:, high].sum(1),
                     pfrac[:, vhf].sum(1)], axis=1)


def wilcoxon_p(a, b):
    """Two-sided paired Wilcoxon signed-rank p-value across coordinates."""
    if np.allclose(np.asarray(a) - np.asarray(b), 0):
        return 1.0
    try:
        return wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue
    except ValueError:
        return 1.0


def pstars(p):
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _overlay_bands(ax, xmin, xmax):
    """Shade Low | High | Very-high and draw the t1/t2 boundaries."""
    ax.axvspan(xmin, args.t1, color="green", alpha=0.07)
    ax.axvspan(args.t1, args.t2, color="orange", alpha=0.07)
    ax.axvspan(args.t2, xmax, color="grey", alpha=0.12)
    for t in (args.t1, args.t2):
        ax.axvline(t, color="black", ls="--", lw=1.0)


def plot_fft_lines(phase, W_adam, W_bgf, start_adam, start_bgf, path):
    """Per-coordinate Baseline-vs-BGF FFT magnitude line plots (with band overlay)."""
    n = len(args.coords)
    rows = (n + 1) // 2
    fig, axs = plt.subplots(rows, 2, figsize=(14, 2.85 * rows), squeeze=False)
    axs = axs.flatten()
    ker = np.ones(args.smooth) / args.smooth
    for ax_i, coord in enumerate(args.coords):
        front = zscore(W_adam[coord:coord + 1])[0]
        end = zscore(W_bgf[coord:coord + 1])[0]
        fmag = np.convolve(np.abs(np.fft.fft(front)), ker, mode="same")
        emag = np.convolve(np.abs(np.fft.fft(end)), ker, mode="same")
        freq = np.fft.fftfreq(len(fmag), 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_freq = np.log10(freq)

        ax = axs[ax_i]
        xmin, xmax = np.nanmin(log_freq[1:]), np.nanmax(log_freq)
        _overlay_bands(ax, xmin, xmax)
        ax.plot(log_freq, fmag, label=f"Baseline-{phase} phase", color="red", alpha=0.5)
        ax.plot(log_freq, emag, label=f"BGF-{phase} phase", color="blue", alpha=0.5)
        ax.set_ylabel("Intensity")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_xlim(xmin, xmax)
        ax.get_xaxis().set_major_formatter(FormatStrFormatter("10^(%0.1f)"))
        ax.legend(fontsize=8)
        ax.set_title(f"FFT of Normalized Gradient Signal {coord}")
    for ax in axs[n:]:
        ax.set_visible(False)
    fig.suptitle(f"{phase} phase  (window starts: Baseline {start_adam}, BGF {start_bgf})   "
                 f"bands: Low < 10^{args.t1} | High 10^{args.t1}..10^{args.t2} | "
                 f"VHF >= 10^{args.t2}", fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _style_boxes(bp, colour):
    for patch in bp["boxes"]:
        patch.set_facecolor(colour)
        patch.set_alpha(0.55)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.6)
    for elem in bp["whiskers"] + bp["caps"]:
        elem.set_color("black")
        elem.set_linewidth(0.8)
    for fl in bp["fliers"]:
        fl.set(marker="o", markersize=2.2, markerfacecolor=colour,
               markeredgecolor="none", alpha=0.25)


def _box_norm(base_bands, comp_bands, base_name, base_label, comp_label,
              ratio_prefix, title, path):
    """Box plots of the band power fractions, each band normalised by the
    BASELINE median (baseline = Adam in the Adam-vs-BGF figures, Training in
    the js_* train-vs-generalization figures).

    The absolute band fractions sit at very different heights (Low << High <<
    Very-high), which hides the comparison; dividing every coordinate's band
    fraction by the baseline MEDIAN of that band centres the baseline box of
    every band on 1 (dashed reference), so the comparison box directly shows
    its distribution RELATIVE to the baseline.  Normalising by a per-band
    constant leaves the paired Wilcoxon p-values unchanged.
    """
    stats = [dict(p=wilcoxon_p(comp_bands[:, k], base_bands[:, k])) for k in range(3)]

    fig, ax = plt.subplots(figsize=(9, 5.6))
    x = np.arange(3)
    off = 0.19
    width = 0.32

    ref = np.median(base_bands, axis=0)
    ref[ref == 0] = 1.0
    base_n = base_bands / ref
    comp_n = comp_bands / ref

    bp_a = ax.boxplot([base_n[:, k] for k in range(3)], positions=x - off,
                      widths=width, patch_artist=True, manage_ticks=False)
    bp_b = ax.boxplot([comp_n[:, k] for k in range(3)], positions=x + off,
                      widths=width, patch_artist=True, manage_ticks=False)
    _style_boxes(bp_a, C_ADAM)
    _style_boxes(bp_b, C_BGF)
    ax.axhline(1.0, color="black", ls="--", lw=1.0, alpha=0.6)

    # robust y-limits from the 1.5*IQR whisker extents (ignore extreme fliers)
    def _whisk(arr, hi):
        q1, q3 = np.percentile(arr, [25, 75])
        return (q3 + 1.5 * (q3 - q1)) if hi else (q1 - 1.5 * (q3 - q1))
    tops = [max(_whisk(base_n[:, k], True), _whisk(comp_n[:, k], True)) for k in range(3)]
    bots = [min(_whisk(base_n[:, k], False), _whisk(comp_n[:, k], False)) for k in range(3)]
    ax.set_ylim(max(0.0, min(bots) * 0.9), max(tops) * 1.30)

    for k, st in enumerate(stats):
        med_r = np.median(comp_n[:, k])
        ax.text(k, tops[k] * 1.03, f"{ratio_prefix} x{med_r:.2f}  {pstars(st['p'])}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(BANDS)
    ax.set_ylabel(f"Power fraction / {base_name} band median  (per coordinate)")
    ax.set_title(title)
    ax.grid(axis="y", ls=":", alpha=0.4)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=C_ADAM, alpha=0.55,
                             edgecolor="black", label=base_label),
               plt.Rectangle((0, 0), 1, 1, facecolor=C_BGF, alpha=0.55,
                             edgecolor="black", label=comp_label)]
    ax.legend(handles=handles, fontsize=9, loc="upper left")
    band_txt = (f"each band divided by its {base_name} median (dashed line = {base_name} = 1); "
                f"bands: Low < 10^{args.t1:.2f} | High 10^{args.t1:.2f}..10^{args.t2:.2f} | "
                f"VHF >= 10^{args.t2:.2f} Hz    (x = median {ratio_prefix} ratio; "
                f"* p<.05 ** p<.01 *** p<.001, paired Wilcoxon)")
    fig.text(0.5, 0.005, band_txt, ha="center", va="bottom", fontsize=7.3, color="0.35")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_box_norm(phase, adam_bands, bgf_bands, path):
    """boxFFTnorm_*: Adam-vs-BGF band power fractions, Adam median = 1."""
    _box_norm(adam_bands, bgf_bands, base_name="Adam",
              base_label="Adam (= 1 by normalisation)",
              comp_label="BGF (filtered)", ratio_prefix="BGF",
              title=f"{phase} phase - band power RELATIVE TO ADAM across "
                    f"{adam_bands.shape[0]} coordinates",
              path=path)


# --------------------------------------------------------------------------- #
# js_* figures: training vs generalization WITHIN the Adam run (no BGF)
# --------------------------------------------------------------------------- #
def coordinate_labels(positions):
    """Human-readable label per sampled coordinate: 'layer/name w [flat index]'."""
    labels = []
    for path, idx in positions:
        for j in idx:
            labels.append(f"{'/'.join(path[:-1])} '{path[-1]}' [{int(j)}]")
    return labels


def js_sampled_indices(n_coords):
    """Evenly-spaced coordinate indices shown in the overlay's single panels."""
    step = max(1, n_coords // 33)
    return [t for t in range(0, n_coords, step)][:34]


def plot_js_overlay(arr, labels, train_start, gen_start, path):
    """Per-coordinate Training-vs-Generalization |FFT| overlays + aggregate.

    For each shown coordinate the raw Adam gradient signal is z-scored globally
    over the whole trajectory, then each phase window is z-scored again and its
    FFT MAGNITUDE plotted against log10 frequency - red = training window over
    blue = generalization window.  The first five sampled coordinates get a
    panel each; the sixth panel is the mean |FFT| over ALL coordinates (the
    aggregate view).  Dashed lines mark the t1/t2 band boundaries.  (The box
    figures use POWER fractions; magnitude is kept here to match the original
    per-coordinate overlay figure.)
    """
    n_coords = arr.shape[0]
    sig_idxs = js_sampled_indices(n_coords)
    allz = zscore(arr)                                     # global z-score per coordinate
    freq = np.fft.fftfreq(args.window, 1.0)
    pos = freq > 0
    logf = np.log10(freq[pos])

    def _phase_mag(sig1d, start):
        w = sig1d[start:start + args.window]
        w = (w - w.mean()) / (w.std() + 1e-5)              # per-window z-score
        return np.abs(np.fft.fft(w))[pos]

    fig, axs = plt.subplots(3, 2, figsize=(16, 12))
    axs = axs.flatten()
    for i in range(6):
        ax = axs[i]
        if i == 5:                                         # aggregate panel
            mt = np.mean([_phase_mag(allz[c], train_start) for c in range(n_coords)], 0)
            mg = np.mean([_phase_mag(allz[c], gen_start) for c in range(n_coords)], 0)
            ax.plot(logf, mt, color=C_ADAM, alpha=0.75, lw=1.2,
                    label=f"Training  (mean |FFT|, all {n_coords})")
            ax.plot(logf, mg, color=C_BGF, alpha=0.75, lw=1.2,
                    label="Generalization")
            ax.set_title(f"AGGREGATE mean |FFT| over ALL {n_coords} coordinates",
                         fontsize=9)
        else:
            sv = allz[sig_idxs[i]]
            ax.plot(logf, _phase_mag(sv, train_start), color=C_ADAM, alpha=0.5,
                    label="Training phase")
            ax.plot(logf, _phase_mag(sv, gen_start), color=C_BGF, alpha=0.5,
                    label="Generalization phase")
            ax.set_title(f"coord {sig_idxs[i]}: {labels[sig_idxs[i]]}", fontsize=9)
        for xv in (args.t1, args.t2):                      # band boundaries
            ax.axvline(xv, color="green", ls="--", lw=1)
        ax.set_xlim(np.nanmin(logf), np.nanmax(logf))
        ax.get_xaxis().set_major_formatter(FormatStrFormatter("10^%0.1f"))
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Intensity  (|FFT|)")
        ax.legend(fontsize=7)
        ax.grid(ls=":", alpha=0.35)
    fig.suptitle("Adam run: Training (red) vs Generalization (blue) FFT overlay "
                 f"(window starts {train_start} / {gen_start})  |  green dashes = "
                 f"Low|High|VHF bands (10^{args.t1:.2f}, 10^{args.t2:.2f})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_js_box_norm(train_bands, gen_bands, train_start, gen_start, path):
    """js_box_norm_*: Training-vs-Generalization band power fractions of the
    Adam run, Training median = 1."""
    W = args.window
    _box_norm(train_bands, gen_bands, base_name="Training",
              base_label=f"Training [{train_start}-{train_start + W}] "
                         f"(= 1 by normalisation)",
              comp_label=f"Generalization [{gen_start}-{gen_start + W}]",
              ratio_prefix="gen/tr",
              title=f"Adam run - band power RELATIVE TO TRAINING across "
                    f"{train_bands.shape[0]} coordinates",
              path=path)


def logbinned_ratio(pf_train, pf_gen, log10f, nbins=40):
    """Per-log-bin gen/train mean-spectrum power ratio.

    Linear FFT bins are dense at high frequency and sparse at low frequency;
    summing the mean spectra inside log-spaced bins gives an approximately
    even-variance ratio curve.  Empty bins stay NaN (they leave gaps).
    """
    edges = np.linspace(log10f.min(), log10f.max(), nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    tmean = pf_train.mean(0)
    gmean = pf_gen.mean(0)
    ratio = np.full(nbins, np.nan)
    for k in range(nbins):
        hi = log10f <= edges[k + 1] if k == nbins - 1 else log10f < edges[k + 1]
        m = (log10f >= edges[k]) & hi
        if not m.any():
            continue
        T = tmean[m].sum()
        if T > 0:
            ratio[k] = gmean[m].sum() / T
    return centers, ratio


JS_SMOOTH_K = 25            # boxcar width for the mean spectra
JS_RIGHT_CUT = JS_SMOOTH_K // 2   # width of the edge-drooped high-frequency tail


def _smooth_same(y, k=JS_SMOOTH_K):
    """Boxcar smoothing with np.convolve(mode='same') - NOT edge-renormalised,
    so both array ends droop; the high-frequency end (droop + the anomalous
    Nyquist bin) is trimmed at plot time by JS_RIGHT_CUT points."""
    return np.convolve(y, np.ones(k) / k, mode="same")


def _crossing_y(tr, gn):
    """Geometric-mean power level where the two smoothed spectra cross.

    Used to place the ratio = 1 baseline of the gray bars exactly at the
    height where the training and generalization curves meet.
    """
    d = gn - tr
    s = np.sign(d)
    ys = []
    for i in range(len(d) - 1):
        if s[i] != 0 and s[i + 1] != 0 and s[i] != s[i + 1]:
            f = (0.0 - d[i]) / (d[i + 1] - d[i])
            ys.append(tr[i] + f * (tr[i + 1] - tr[i]))
    ys = np.asarray([v for v in ys if v > 0])
    return float(np.exp(np.log(ys).mean())) if ys.size else float(np.sqrt(tr.mean() * gn.mean()))


def plot_js_thresholds_simple(pf_train, pf_gen, log10f, train_start, gen_start, path):
    """js_thresholds_simple_*: the two phases' mean power-fraction spectra with
    the per-log-bin gen/train power ratio as gray bars BEHIND them.

    The bar baseline (ratio = 1, 'gen = train') sits, in display coordinates,
    exactly at the power level where the two curves cross, so 'bar pokes above
    the baseline' and 'generalization > training at this frequency' can be
    read off directly against the curves.  Band regions are shaded and the
    t1/t2 boundaries drawn as in every other figure of this set.
    """
    tr_full = _smooth_same(pf_train.mean(0))
    gn_full = _smooth_same(pf_gen.mean(0))
    logf = log10f[:-JS_RIGHT_CUT]              # drop the right-edge droop/Nyquist tail
    tr = tr_full[:-JS_RIGHT_CUT]
    gn = gn_full[:-JS_RIGHT_CUT]

    centers, ratio = logbinned_ratio(pf_train, pf_gen, log10f)
    keep = np.isfinite(ratio)
    bx, br = centers[keep], ratio[keep]

    y_cross = _crossing_y(tr, gn)

    # primary log-power axis limits, with headroom above the curves
    p_lo, p_hi = min(tr.min(), gn.min()) * 0.80, max(tr.max(), gn.max()) * 2.0
    # solve the twin (linear ratio) axis limits so that ratio = 1 lands on the
    # display height of y_cross and the whole ratio range fits with a margin
    L = np.log10
    f_cross = (L(y_cross) - L(p_lo)) / (L(p_hi) - L(p_lo))
    need_below = 1.0 - br.min()
    need_above = br.max() - 1.0
    S = 1.12 * max(need_below / f_cross, need_above / (1.0 - f_cross), 1e-9)
    r_lo = 1.0 - f_cross * S
    r_hi = 1.0 + (1.0 - f_cross) * S

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.axvspan(logf.min(), args.t1, color="green", alpha=0.07)
    ax.axvspan(args.t1, args.t2, color="orange", alpha=0.07)
    ax.axvspan(args.t2, logf.max(), color="grey", alpha=0.12)

    # twin axis carries the gray ratio bars, drawn BEHIND the spectra curves
    ax2 = ax.twinx()
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    bw = (centers[1] - centers[0]) * 0.9
    ax2.bar(bx, br - 1.0, bottom=1.0, width=bw, color="0.6", alpha=0.55,
            edgecolor="none", zorder=1, label="gen/train power ratio")
    ax2.axhline(1.0, color="0.35", lw=1.0, ls="-", zorder=1.1)
    ax2.set_ylim(r_lo, r_hi)
    ax2.set_ylabel("Generalization / Training  power ratio", fontsize=9, color="0.35")
    ax2.tick_params(axis="y", labelcolor="0.4", labelsize=8)

    W = args.window
    ax.plot(logf, tr, color=C_ADAM, lw=1.6,
            label=f"Training [{train_start}-{train_start + W}]", zorder=5)
    ax.plot(logf, gn, color=C_BGF, lw=1.6,
            label=f"Generalization [{gen_start}-{gen_start + W}]", zorder=5)
    for t in (args.t1, args.t2):
        ax.axvline(t, color="black", ls="--", lw=1.0, zorder=4)
    ax.set_yscale("log")
    ax.set_ylim(p_lo, p_hi)
    ax.set_xlabel("log10 frequency [Hz]")
    ax.set_ylabel("mean power fraction / bin")
    ax.set_title("Adam run: mean spectra with gen/train ratio bars  "
                 "(Low | High | Very-high)")
    ax.grid(ls=":", alpha=0.4, zorder=0)
    ax.annotate("baseline: curves cross  (gen/train = 1)",
                xy=(logf.min(), y_cross), xytext=(6, 6),
                textcoords="offset points", fontsize=7.5, color="0.3", zorder=6)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left", framealpha=0.9)
    fig.text(0.5, 0.005,
             f"bands: Low < 10^{args.t1:.1f} | High 10^{args.t1:.1f}..10^{args.t2:.1f} | "
             f"VHF >= 10^{args.t2:.1f} Hz.   Gray bars = per-log-bin gen/train power "
             f"ratio (all populated bins); baseline (=1) at the curve crossing "
             f"y={y_cross:.1e}.  Curves boxcar-smoothed (k={JS_SMOOTH_K}); "
             f"high-frequency edge tail trimmed.",
             ha="center", va="bottom", fontsize=6.6, color="0.4")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
adam_run = find_run('baseline', args.adam_run)
bgf_run = find_run('ours', args.bgf_run)
print(f'Adam run: {adam_run}')
print(f'BGF  run: {bgf_run}')

adam_sig, bgf_sig, adam_full, adam_positions, align = load_signals(adam_run, bgf_run)
print(f'signals: Adam raw {adam_sig.shape}, BGF filtered {bgf_sig.shape} '
      f'(column 0 == training step {align})')

if args.train_start < align:
    raise SystemExit(f'--train_start must be >= {align} (the recorded signals start there)')

gen_start_adam = args.gen_start_adam or find_generalization_start(adam_run)
gen_start_bgf = args.gen_start_bgf or find_generalization_start(bgf_run)
for name, s in [('Adam', gen_start_adam), ('BGF', gen_start_bgf)]:
    if s is None:
        raise SystemExit(
            f'{name} run: the {args.ma_window}-step moving average of the training accuracy '
            f'never reaches {args.acc_threshold:.0%}, so no generalization phase exists. '
            f'Train longer or set --gen_start_adam/--gen_start_bgf manually.')
print(f'generalization onsets ({args.ma_window}-step MA of training accuracy >= '
      f'{args.acc_threshold:.0%}): Adam {gen_start_adam}, BGF {gen_start_bgf}')

PHASES = [
    dict(name='Training', start_adam=args.train_start, start_bgf=args.train_start),
    dict(name='Generalization', start_adam=gen_start_adam, start_bgf=gen_start_bgf),
]

log10f = np.log10(np.fft.rfftfreq(args.window, d=1.0)[1:])
os.makedirs(args.out_dir, exist_ok=True)

for ph in PHASES:
    sa, sb = ph['start_adam'], ph['start_bgf']
    W_adam = window(adam_sig, sa, align, f"Adam {ph['name']}")
    W_bgf = window(bgf_sig, sb, align, f"BGF {ph['name']}")

    fft_path = os.path.join(args.out_dir, f'FFT_{sa}_{sb}.pdf')
    plot_fft_lines(ph['name'], W_adam, W_bgf, sa, sb, fft_path)
    print(f'wrote {fft_path}')

    adam_bands = band_fractions(power_fractions(W_adam), log10f)
    bgf_bands = band_fractions(power_fractions(W_bgf), log10f)
    box_path = os.path.join(args.out_dir, f'boxFFTnorm_{sa}_{sb}.pdf')
    plot_box_norm(ph['name'], adam_bands, bgf_bands, box_path)
    print(f'wrote {box_path}')

    print(f"---- {ph['name']} phase  (Adam start {sa}, BGF start {sb}) ----")
    print(f"{'band':>10} | {'Adam':>8} {'BGF':>8} | {'BGF/Adam':>8} {'p':>9} sig")
    for k in range(3):
        a, b = adam_bands[:, k], bgf_bands[:, k]
        ratio = b.mean() / a.mean() if a.mean() > 0 else float('nan')
        p = wilcoxon_p(b, a)
        print(f"{BANDS[k]:>10} | {a.mean():8.4f} {b.mean():8.4f} | "
              f"{ratio:8.3f} {p:9.2e} {pstars(p)}")

# ---- js_* figures: training vs generalization WITHIN the Adam run ---------- #
tb, tg = args.train_start, gen_start_adam
# raw un-normalised Adam matrix; column i == training step i (also validates
# that both windows fit inside the recorded signal)
W_train_js = window(adam_full, tb, 0, 'Adam Training (js)')
W_gen_js = window(adam_full, tg, 0, 'Adam Generalization (js)')

overlay_path = os.path.join(args.out_dir, f'js_overlay_{tb}_{tg}.pdf')
plot_js_overlay(adam_full, coordinate_labels(adam_positions), tb, tg, overlay_path)
print(f'wrote {overlay_path}')

pf_train_js = power_fractions(W_train_js)
pf_gen_js = power_fractions(W_gen_js)
train_bands_js = band_fractions(pf_train_js, log10f)
gen_bands_js = band_fractions(pf_gen_js, log10f)

box_path = os.path.join(args.out_dir, f'js_box_norm_{tb}_{tg}.pdf')
plot_js_box_norm(train_bands_js, gen_bands_js, tb, tg, box_path)
print(f'wrote {box_path}')

thr_path = os.path.join(args.out_dir, f'js_thresholds_simple_{tb}_{tg}.pdf')
plot_js_thresholds_simple(pf_train_js, pf_gen_js, log10f, tb, tg, thr_path)
print(f'wrote {thr_path}')

print(f"---- Adam run: Training (start {tb}) vs Generalization (start {tg}) ----")
print(f"{'band':>10} | {'Train':>8} {'Gen':>8} | {'gen/tr':>8} {'p':>9} sig")
for k in range(3):
    a, b = train_bands_js[:, k], gen_bands_js[:, k]
    ratio = b.mean() / a.mean() if a.mean() > 0 else float('nan')
    p = wilcoxon_p(b, a)
    print(f"{BANDS[k]:>10} | {a.mean():8.4f} {b.mean():8.4f} | "
          f"{ratio:8.3f} {p:9.2e} {pstars(p)}")
print('Done.')
