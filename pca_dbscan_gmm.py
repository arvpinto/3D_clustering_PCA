#!/usr/bin/env python3
"""
pca_dbscan_gmm.py - density-based clustering of PCA projections and choice of
representative frames.

Reads the npz file written by traj_pca.py and:
  1. flags outliers with DBSCAN (frames in sparsely populated regions),
  2. clusters the remaining frames with a Gaussian mixture model (GMM),
  3. picks the representative frames of each cluster:
       - every frame is compared with the frames of its cluster using the RMSD over
         the whole PCA selection (computed from the projections on ALL the PCs kept
         by traj_pca.py, not only the ones used for clustering),
       - its local density is the mean RMSD to its nearest neighbours,
       - the densest frames form the core of the cluster (--core-fraction),
       - the core frames are ranked by their mean RMSD to the whole cluster, and the
         best ones are kept, at least --min-sep frames apart.
     The first representative is the medoid of the core: a real frame, in the most
     populated region of the cluster, with the lowest mean RMSD to the other frames.

Example:
    python pca_dbscan_gmm.py -i pca.npz                      # first look, automatic parameters
    python pca_dbscan_gmm.py -i pca.npz --eps 1.5 --n-clusters 4

Outputs (PREFIX is set with -o):
    PREFIX_assignments.csv      cluster, membership probability and density of every frame
    PREFIX_representatives.csv  representative frames (input of extract_frames.py)
    PREFIX_3d.html              interactive 3D plot
    PREFIX_overview.png         2D maps and density curves of the clusters
    PREFIX_timeline.png         cluster of each frame along the trajectory
    PREFIX_parameters.png       k-distance curve (eps) and BIC / ICL curves (number of clusters)

Frame numbers are the 0-based trajectory frames stored by traj_pca.py.
Outliers are given cluster -1. Clusters are numbered from the most populated (0).
"""

import argparse
import csv
import sys
import warnings

import numpy as np

# Colours of the clusters, in this order (colour-blind safe on a white background).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "v"]          # a new marker shape every 8 clusters
OUTLIER = "#9b9a94"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def colour(c):
    return PALETTE[c % len(PALETTE)]


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="DBSCAN + GMM clustering of PCA projections, with representative frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    io = p.add_argument_group("input and output")
    io.add_argument("-i", "--input", required=True, help="npz file written by traj_pca.py")
    io.add_argument("-o", "--output", default="cluster", help="prefix of the output files")
    io.add_argument("--dims", type=int, default=3, help="number of PCs used for clustering")

    ou = p.add_argument_group("outliers (DBSCAN)")
    ou.add_argument("--eps", default="auto",
                    help="neighbourhood radius in angstrom; 'auto' takes the knee of the k-distance curve, "
                         "'none' skips the outlier search")
    ou.add_argument("--min-samples", type=int, default=10,
                    help="frames needed inside eps for a region to count as populated")

    cl = p.add_argument_group("clusters (Gaussian mixture model)")
    cl.add_argument("--n-clusters", default="auto",
                    help="number of clusters; 'auto' scans 1 to --max-clusters and takes the lowest ICL")
    cl.add_argument("--max-clusters", type=int, default=8, help="largest number tried by 'auto'")
    cl.add_argument("--covariance", choices=["full", "tied", "diag", "spherical"], default="full",
                    help="shape of the Gaussians")
    cl.add_argument("--n-init", type=int, default=10, help="GMM restarts (the best fit is kept)")
    cl.add_argument("--seed", type=int, default=0, help="random seed, for reproducible results")

    rp = p.add_argument_group("representative frames")
    rp.add_argument("--n-rep", type=int, default=5, help="representative frames per cluster")
    rp.add_argument("--core-fraction", type=float, default=0.10,
                    help="fraction of each cluster, taken from its densest frames, that forms the core; "
                         "the same fraction sets the number of neighbours used to measure the density")
    rp.add_argument("--min-sep", type=int, default=10,
                    help="minimum separation between representatives of a cluster, in analysed frames "
                         "(avoids picking consecutive, nearly identical frames)")
    rp.add_argument("--max-ref", type=int, default=5000,
                    help="clusters larger than this are compared with a random subset of this many frames "
                         "(keeps memory and time in check for long trajectories)")

    pl = p.add_argument_group("plots")
    pl.add_argument("--no-show", action="store_true", help="save the figures without opening windows")
    pl.add_argument("--dpi", type=int, default=600, help="resolution of the saved figures")
    pl.add_argument("--plotly-js", choices=["cdn", "embed"], default="cdn",
                    help="'cdn' gives a small html file that loads plotly from the internet, "
                         "'embed' a larger file that works offline")
    return p.parse_args()


# --------------------------------------------------------------------------
# Outliers
# --------------------------------------------------------------------------

def k_distances(x, k):
    """Distance of every frame to its k-th neighbour (itself included), sorted."""
    from sklearn.neighbors import NearestNeighbors
    dist, _ = NearestNeighbors(n_neighbors=k).fit(x).kneighbors(x)
    return np.sort(dist[:, -1])


def knee(values):
    """Knee of an increasing curve: the point farthest below the line that joins its ends."""
    n = len(values)
    span = values[-1] - values[0]
    if n < 3 or span <= 0:
        return values[-1]
    xs = np.linspace(0.0, 1.0, n)
    ys = (values - values[0]) / span
    return values[int(np.argmax(xs - ys))]


def find_outliers(x, eps, min_samples):
    from sklearn.cluster import DBSCAN
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(x)
    return labels == -1


# --------------------------------------------------------------------------
# Clusters
# --------------------------------------------------------------------------

def fit_gmm(x, k, args):
    from sklearn.mixture import GaussianMixture
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return GaussianMixture(n_components=k, covariance_type=args.covariance, n_init=args.n_init,
                               tol=1e-5, max_iter=1000, random_state=args.seed).fit(x)


def icl(model, x):
    """Integrated completed likelihood: BIC plus a penalty for overlapping clusters."""
    tau = np.clip(model.predict_proba(x), 1e-300, 1.0)
    return model.bic(x) - 2.0 * np.sum(tau * np.log(tau))


def scan_clusters(x, args):
    ks = list(range(1, min(args.max_clusters, len(x) - 1) + 1))
    models, bic, icl_values = [], [], []
    print("  clusters        BIC          ICL")
    for k in ks:
        m = fit_gmm(x, k, args)
        models.append(m)
        bic.append(m.bic(x))
        icl_values.append(icl(m, x))
        print(f"  {k:8d} {bic[-1]:12.1f} {icl_values[-1]:12.1f}")
    best = int(np.argmin(icl_values))
    return models[best], ks, np.array(bic), np.array(icl_values)


# --------------------------------------------------------------------------
# Representative frames
# --------------------------------------------------------------------------

def rmsd_statistics(full, n_atoms, k_fraction, rng, max_ref=5000, chunk=1000):
    """RMSD statistics of the frames of one cluster, computed from their projections on
    all the PCs kept by traj_pca.py (the distance between two frames in that space,
    divided by sqrt(n_atoms), is their RMSD over the PCA selection after the common fit).

    Returns, for every frame:
      local - mean RMSD to its k nearest frames of the cluster (k = k_fraction of the
              cluster, at least 10): low values mean a densely populated region;
      mean  - mean RMSD to all the frames of the cluster.
    Clusters larger than max_ref frames are compared with a random subset of max_ref frames."""
    n = len(full)
    if n < 2:
        return np.zeros(n), np.zeros(n)
    ref = np.arange(n) if n <= max_ref else np.sort(rng.choice(n, max_ref, replace=False))
    m = len(ref)
    k = int(min(m - 1, max(10, np.ceil(k_fraction * m))))
    r = full[ref]
    r2 = (r ** 2).sum(axis=1)
    local, mean = np.empty(n), np.empty(n)
    for i in range(0, n, chunk):
        idx = np.arange(i, min(n, i + chunk))
        f = full[idx]
        d = np.sqrt(np.clip((f ** 2).sum(axis=1)[:, None] + r2[None, :] - 2.0 * f @ r.T, 0.0, None) / n_atoms)
        # a frame is not its own neighbour
        pos = np.searchsorted(ref, idx)
        pos[pos >= m] = 0
        self_hit = ref[pos] == idx
        d[np.where(self_hit)[0], pos[self_hit]] = np.inf
        mean[idx] = np.where(np.isinf(d), 0.0, d).sum(axis=1) / (m - self_hit)
        local[idx] = np.partition(d, k - 1, axis=1)[:, :k].mean(axis=1)
    return local, mean


def pick_representatives(rows, local, mean, n_rep, core_fraction, min_sep):
    """rows: indices (in the analysed trajectory) of the frames of one cluster.
    local, mean: their RMSD statistics (see rmsd_statistics).
    The core is the densest core_fraction of the cluster (at least 10 frames). The core
    frames are ranked by their mean RMSD to the whole cluster (the medoid of the core
    comes first), followed by the other frames if the core cannot supply enough
    representatives at least min_sep frames apart. Returns the positions chosen and the core mask."""
    n = len(rows)
    n_core = int(min(n, max(np.ceil(core_fraction * n), 10)))
    core = np.zeros(n, dtype=bool)
    core[np.argsort(local, kind="stable")[:n_core]] = True
    order = np.concatenate([np.where(core)[0][np.argsort(mean[core], kind="stable")],
                            np.where(~core)[0][np.argsort(mean[~core], kind="stable")]])
    chosen = []
    for i in order:
        if all(abs(int(rows[i]) - int(rows[j])) >= min_sep for j in chosen):
            chosen.append(i)
            if len(chosen) == n_rep:
                break
    return np.array(chosen, dtype=int), core


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def setup_matplotlib():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlesize": 10, "axes.titlecolor": INK})
    return plt


def save(fig, name, dpi):
    fig.savefig(name, dpi=dpi, facecolor="white")
    print(f"  {name}")


def plot_parameters(kdist, eps, n_out, scan, k_used, min_samples, prefix, dpi):
    plt = setup_matplotlib()
    panels = [p for p in ("kdist" if kdist is not None else None, "scan" if scan is not None else None) if p]
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 3.8), squeeze=False)
    for ax, panel in zip(axes[0], panels):
        if panel == "kdist":
            ax.plot(np.arange(len(kdist)), kdist, color=PALETTE[0], lw=2)
            ax.axhline(eps, color=INK, lw=1, ls="--")
            ax.text(0.02 * len(kdist), eps, f" eps = {eps:.2f} Å  ({n_out} outliers)", va="bottom", color=INK)
            ax.set_xlabel("Frames, sorted")
            ax.set_ylabel(f"Distance to the {min_samples}th neighbour (Å)")
            ax.set_title("Outlier threshold (DBSCAN)", loc="left")
        else:
            ks, bic, icl_values = scan
            ax.plot(ks, bic, color=PALETTE[0], lw=2, marker="o", ms=5, label="BIC")
            ax.plot(ks, icl_values, color=PALETTE[1], lw=2, marker="s", ms=5, label="ICL")
            ax.axvline(k_used, color=INK, lw=1, ls="--")
            ax.set_xticks(ks)
            ax.set_xlabel("Number of clusters")
            ax.set_ylabel("Criterion (lower is better)")
            ax.set_title(f"Number of clusters (GMM): {k_used} used", loc="left")
            ax.legend(frameon=False)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, f"{prefix}_parameters.png", dpi)
    return fig


def plot_overview(x, labels, reps, ratio, prefix, dpi, rng):
    """Top: 2D maps of the clusters. Bottom: density of each cluster along each PC,
    scaled by its population. Diamonds mark the representative frames."""
    from itertools import combinations
    from scipy.stats import gaussian_kde
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    plt = setup_matplotlib()

    n_show = min(x.shape[1], 3)
    pairs = list(combinations(range(n_show), 2))
    n_col = max(len(pairs), n_show)
    fig, axes = plt.subplots(2, n_col, figsize=(4.2 * n_col, 7.6), squeeze=False)
    clusters = [c for c in np.unique(labels) if c >= 0]
    n_clustered = int((labels >= 0).sum())
    halo = [pe.withStroke(linewidth=2.5, foreground="white")]

    def axis_label(k):
        return f"PC{k + 1} ({100 * ratio[k]:.1f}%)  /  Å"

    for ax, (a, b) in zip(axes[0], pairs):
        out = labels == -1
        ax.scatter(x[out, a], x[out, b], s=10, marker="x", color=OUTLIER, lw=0.8, rasterized=True)
        for c in clusters:
            m = labels == c
            ax.scatter(x[m, a], x[m, b], s=7, color=colour(c), marker=MARKERS[(c // 8) % 4],
                       alpha=0.45, lw=0, rasterized=True)
        for c in clusters:
            rows = reps[c]["rows"]
            if len(rows) == 0:
                continue
            ax.scatter(x[rows[1:], a], x[rows[1:], b], s=26, marker="D", color=colour(c),
                       edgecolor="white", lw=0.9, zorder=4)
            ax.scatter(x[rows[0], a], x[rows[0], b], s=70, marker="D", color=colour(c),
                       edgecolor=INK, lw=1.1, zorder=5)
            ax.annotate(str(c), (x[rows[0], a], x[rows[0], b]), xytext=(6, 6), textcoords="offset points",
                        fontsize=10, fontweight="bold", color=INK, path_effects=halo, zorder=6)
        ax.set_xlabel(axis_label(a))
        ax.set_ylabel(axis_label(b))
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    for ax in axes[0][len(pairs):]:
        ax.set_visible(False)

    for k, ax in zip(range(n_show), axes[1]):
        lo, hi = x[labels >= 0, k].min(), x[labels >= 0, k].max()
        pad = 0.08 * (hi - lo)
        grid = np.linspace(lo - pad, hi + pad, 300)
        for c in clusters:
            vals = x[labels == c, k]
            if len(vals) < 3 or np.ptp(vals) == 0:
                continue
            fit = vals if len(vals) <= 5000 else rng.choice(vals, 5000, replace=False)
            kde = gaussian_kde(fit)
            weight = len(vals) / n_clustered
            ax.plot(grid, weight * kde(grid), color=colour(c), lw=2)
            ax.fill_between(grid, weight * kde(grid), color=colour(c), alpha=0.10, lw=0)
            rows = reps[c]["rows"]
            if len(rows):
                ax.scatter(x[rows[1:], k], weight * kde(x[rows[1:], k]), s=26, marker="D", color=colour(c),
                           edgecolor="white", lw=0.9, zorder=4)
                ax.scatter(x[rows[0], k], weight * kde(x[rows[0], k]), s=70, marker="D", color=colour(c),
                           edgecolor=INK, lw=1.1, zorder=5)
        ax.set_xlabel(axis_label(k))
        ax.set_ylabel("Density, scaled by population")
        ax.set_ylim(bottom=0)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    for ax in axes[1][n_show:]:
        ax.set_visible(False)

    handles = [Line2D([], [], marker=MARKERS[(c // 8) % 4], ls="", color=colour(c), ms=7,
                      label=f"Cluster {c} ({int((labels == c).sum())} frames)") for c in clusters]
    if (labels == -1).any():
        handles.append(Line2D([], [], marker="x", ls="", color=OUTLIER, ms=6,
                              label=f"Outliers ({int((labels == -1).sum())})"))
    handles.append(Line2D([], [], marker="D", ls="", color="#c9c8c2", markeredgecolor=INK, ms=8,
                          label="Representative frame (best)"))
    handles.append(Line2D([], [], marker="D", ls="", color="#c9c8c2", markeredgecolor="white", ms=6,
                          label="Other representatives"))
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 5), frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    rows_legend = int(np.ceil(len(handles) / min(len(handles), 5)))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.035 * rows_legend))
    save(fig, f"{prefix}_overview.png", dpi)
    return fig


def plot_timeline(times, labels, reps, prefix, dpi):
    plt = setup_matplotlib()
    in_ns = times.max() >= 1000.0
    t = times / 1000.0 if in_ns else times
    clusters = [c for c in np.unique(labels) if c >= 0]
    rows = clusters + ([-1] if (labels == -1).any() else [])
    fig, ax = plt.subplots(figsize=(9.5, 1.2 + 0.42 * len(rows)))
    for y, c in enumerate(rows):
        m = labels == c
        col = colour(c) if c >= 0 else OUTLIER
        ax.scatter(t[m], np.full(m.sum(), y), marker="|", s=160, lw=0.6, color=col, rasterized=True)
        if c >= 0 and len(reps[c]["rows"]):
            r = reps[c]["rows"]
            ax.scatter(t[r[1:]], np.full(len(r) - 1, y), s=26, marker="D", color=col, edgecolor="white", lw=0.9, zorder=4)
            ax.scatter(t[r[0]], y, s=70, marker="D", color=col, edgecolor=INK, lw=1.1, zorder=5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"Cluster {c}" if c >= 0 else "Outliers" for c in rows])
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlabel(f"Time ({'ns' if in_ns else 'ps'})")
    ax.set_xlim(t.min(), t.max())
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.set_title("Cluster of each frame along the trajectory (diamonds: representative frames)", loc="left")
    fig.tight_layout()
    save(fig, f"{prefix}_timeline.png", dpi)
    return fig


def plot_3d(x, labels, prob, rank, frames, times, reps, ratio, prefix, plotly_js):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  (plotly is not installed: the interactive plot was skipped)")
        return
    n_show = min(x.shape[1], 3)
    clusters = [c for c in np.unique(labels) if c >= 0]
    n_total = len(labels)
    is3d = n_show == 3
    Scatter = go.Scatter3d if is3d else go.Scatter
    fig = go.Figure()

    def xyz(mask):
        d = dict(x=x[mask, 0], y=x[mask, 1])
        if is3d:
            d["z"] = x[mask, 2]
        return d

    hover = "frame %{customdata[0]}<br>time %{customdata[1]:.0f} ps"
    out = labels == -1
    if out.any():
        fig.add_trace(Scatter(**xyz(out), mode="markers", name=f"Outliers ({int(out.sum())})",
                              marker=dict(size=2.5 if is3d else 5, color=OUTLIER, symbol="x"),
                              customdata=np.column_stack([frames[out], times[out]]),
                              hovertemplate=hover + "<extra>outlier</extra>"))
    for c in clusters:
        m = labels == c
        size = (2.0 + 4.0 * rank[m]) if is3d else (4.0 + 5.0 * rank[m])
        fig.add_trace(Scatter(**xyz(m), mode="markers", legendgroup=f"c{c}",
                              name=f"Cluster {c} ({int(m.sum())} frames, {100 * m.sum() / n_total:.1f}%)",
                              marker=dict(size=size, color=colour(c), opacity=0.55, line=dict(width=0)),
                              customdata=np.column_stack([frames[m], times[m], prob[m]]),
                              hovertemplate=hover + "<br>probability %{customdata[2]:.2f}"
                                                    f"<extra>cluster {c}</extra>"))
    for c in clusters:
        r = reps[c]["rows"]
        if len(r) == 0:
            continue
        order = np.arange(1, len(r) + 1)
        fig.add_trace(Scatter(**xyz(r), mode="markers", legendgroup=f"c{c}", showlegend=False,
                              name=f"Representatives of cluster {c}",
                              marker=dict(size=np.where(order == 1, 9, 6) if is3d else np.where(order == 1, 14, 10),
                                          color=colour(c), symbol="diamond", opacity=1.0,
                                          line=dict(color=INK, width=2 if is3d else 1.5)),
                              customdata=np.column_stack([frames[r], times[r], order, reps[c]["rmsd"]]),
                              hovertemplate="<b>representative %{customdata[2]}</b><br>" + hover +
                                            "<br>mean RMSD to the cluster %{customdata[3]:.2f} Å"
                                            f"<extra>cluster {c}</extra>"))
    # legend entry that explains the diamonds
    fig.add_trace(Scatter(**{k: [None] for k in ("x", "y", "z")[:n_show]}, mode="markers",
                          name="Representative frames", hoverinfo="skip",
                          marker=dict(size=8 if is3d else 12, color="#c9c8c2", symbol="diamond",
                                      line=dict(color=INK, width=2 if is3d else 1.5))))

    titles = [f"PC{k + 1} ({100 * ratio[k]:.1f}%) / Å" for k in range(n_show)]
    axis = dict(backgroundcolor="white", gridcolor=GRID, zerolinecolor=GRID, linecolor=MUTED,
                tickfont=dict(color=MUTED), title_font=dict(color=INK))
    layout = dict(
        template="plotly_white",
        title=dict(text="Clusters of the PCA projections", x=0.02, xanchor="left", font=dict(color=INK, size=16)),
        font=dict(family="system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif", color=INK),
        legend=dict(itemsizing="constant", x=0.01, y=0.94, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=0, r=0, t=48, b=0),
        paper_bgcolor="white",
    )
    if is3d:
        layout["scene"] = dict(xaxis=dict(title=titles[0], **axis), yaxis=dict(title=titles[1], **axis),
                               zaxis=dict(title=titles[2], **axis), aspectmode="data")
    else:
        layout.update(xaxis=dict(title=titles[0], gridcolor=GRID), yaxis=dict(title=titles[1], gridcolor=GRID),
                      margin=dict(l=60, r=10, t=48, b=50))
    fig.update_layout(**layout)
    name = f"{prefix}_3d.html"
    fig.write_html(name, include_plotlyjs="cdn" if plotly_js == "cdn" else True, full_html=True)
    print(f"  {name}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    args = parse_args()
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    rng = np.random.default_rng(args.seed)

    try:
        data = np.load(args.input, allow_pickle=False)
        data["projections"], data["n_atoms"]
    except (OSError, KeyError, ValueError) as err:
        sys.exit(f"ERROR: could not read {args.input} as an npz file written by traj_pca.py ({err}).")
    full = data["projections"].astype(np.float64)
    frames, times = data["frames"], data["times"]
    ratio, n_atoms = data["variance_ratio"], int(data["n_atoms"])
    n_frames, n_kept = full.shape
    if not 2 <= args.dims <= n_kept:
        sys.exit(f"ERROR: --dims must be between 2 and {n_kept} (PCs stored in {args.input}).")
    x = full[:, :args.dims]
    print(f"{args.input}: {n_frames} frames, selection \"{data['select']}\" ({n_atoms} atoms)")
    print(f"Clustering on the first {args.dims} PCs ({100 * ratio[:args.dims].sum():.1f}% of the variance); "
          f"structures compared with {n_kept} PCs ({100 * ratio.sum():.1f}%)")

    # ---- 1. outliers
    kdist, eps = None, None
    outlier = np.zeros(n_frames, dtype=bool)
    if str(args.eps).lower() != "none":
        kdist = k_distances(x, args.min_samples)
        if str(args.eps).lower() == "auto":
            eps, how = knee(kdist), "knee of the k-distance curve"
        else:
            try:
                eps, how = float(args.eps), "given"
            except ValueError:
                sys.exit("ERROR: --eps takes a number, 'auto' or 'none'.")
        outlier = find_outliers(x, eps, args.min_samples)
        print(f"DBSCAN: eps {eps:.3f} A ({how}), min_samples {args.min_samples}: "
              f"{outlier.sum()} outliers ({100 * outlier.mean():.1f}% of the frames)")
        if outlier.mean() > 0.5:
            sys.exit("ERROR: more than half of the frames are outliers. Increase --eps or lower --min-samples.")
    else:
        print("DBSCAN: skipped (--eps none)")
    keep = np.where(~outlier)[0]
    xk = x[keep]

    # ---- 2. clusters
    scan = None
    if str(args.n_clusters).lower() == "auto":
        print(f"GMM: scanning 1 to {args.max_clusters} clusters")
        model, ks, bic, icl_values = scan_clusters(xk, args)
        scan = (ks, bic, icl_values)
        print(f"GMM: {model.n_components} clusters (lowest ICL)")
    else:
        try:
            k = int(args.n_clusters)
        except ValueError:
            sys.exit("ERROR: --n-clusters takes a whole number or 'auto'.")
        if k < 1:
            sys.exit("ERROR: --n-clusters must be at least 1.")
        model = fit_gmm(xk, k, args)
        print(f"GMM: {k} clusters (given), BIC {model.bic(xk):.1f}, ICL {icl(model, xk):.1f}")
    if not model.converged_:
        print("  WARNING: the GMM did not converge; try another --seed or fewer clusters.")

    proba = model.predict_proba(xk)
    raw = proba.argmax(axis=1)
    sizes = np.bincount(raw, minlength=model.n_components)
    order = [c for c in np.argsort(-sizes, kind="stable") if sizes[c] > 0]   # most populated first
    relabel = {old: new for new, old in enumerate(order)}
    labels = np.full(n_frames, -1, dtype=int)
    labels[keep] = [relabel[c] for c in raw]
    prob = np.full(n_frames, np.nan)
    prob[keep] = proba.max(axis=1)
    clusters = list(range(len(order)))

    # ---- 3. representative frames
    rank = np.full(n_frames, np.nan)
    local_all = np.full(n_frames, np.nan)
    mean_all = np.full(n_frames, np.nan)
    reps = {}
    print(f"Representative frames: medoid of the densest {100 * args.core_fraction:.0f}% of each cluster "
          f"(RMSD over {n_kept} PCs), at least {args.min_sep} frames apart")
    print("  cluster  frames      %   mean prob.  mean RMSD (A)   representative frames (best first)")
    for c in clusters:
        rows = np.where(labels == c)[0]
        local, mean = rmsd_statistics(full[rows], n_atoms, args.core_fraction, rng, args.max_ref)
        local_all[rows], mean_all[rows] = local, mean
        rank[rows] = 1.0 - np.argsort(np.argsort(local, kind="stable")) / max(1, len(rows) - 1)
        chosen, core = pick_representatives(rows, local, mean, args.n_rep, args.core_fraction, args.min_sep)
        rep_rows = rows[chosen]
        reps[c] = dict(rows=rep_rows, rmsd=mean[chosen], local=local[chosen], core=core[chosen])
        print(f"  {c:7d} {len(rows):7d} {100 * len(rows) / n_frames:6.1f} {np.nanmean(prob[rows]):12.2f} "
              f"{mean[chosen[0]]:14.2f}   {' '.join(str(f) for f in frames[rep_rows])}")
        if len(rep_rows) < args.n_rep:
            print(f"          (only {len(rep_rows)} frames are at least {args.min_sep} frames apart in this cluster)")
    print("  (mean RMSD: mean RMSD of the frames of the cluster to its best representative)")

    # ---- files
    print("Files written")
    name = f"{args.output}_assignments.csv"
    with open(name, "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["frame", "time_ps", "cluster", "probability", "density_rank", "local_rmsd_A", "mean_rmsd_A"] +
                   [f"PC{k + 1}" for k in range(args.dims)])
        for i in range(n_frames):
            w.writerow([int(frames[i]), f"{times[i]:.3f}", int(labels[i]),
                        "" if np.isnan(prob[i]) else f"{prob[i]:.4f}",
                        "" if np.isnan(rank[i]) else f"{rank[i]:.4f}",
                        "" if np.isnan(local_all[i]) else f"{local_all[i]:.3f}",
                        "" if np.isnan(mean_all[i]) else f"{mean_all[i]:.3f}"] + [f"{v:.4f}" for v in x[i]])
    print(f"  {name}")

    name = f"{args.output}_representatives.csv"
    with open(name, "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["cluster", "rank", "frame", "time_ps", "mean_rmsd_A", "local_rmsd_A", "probability", "in_core"])
        for c in clusters:
            for i, row in enumerate(reps[c]["rows"]):
                w.writerow([c, i + 1, int(frames[row]), f"{times[row]:.3f}", f"{reps[c]['rmsd'][i]:.3f}",
                            f"{reps[c]['local'][i]:.3f}", f"{prob[row]:.4f}", int(reps[c]["core"][i])])
    print(f"  {name}")

    plot_3d(x, labels, np.nan_to_num(prob), np.nan_to_num(rank), frames, times, reps, ratio,
            args.output, args.plotly_js)
    for make in (
        lambda: plot_parameters(kdist, eps, int(outlier.sum()), scan, len(clusters), args.min_samples, args.output, args.dpi),
        lambda: plot_overview(x, labels, reps, ratio, args.output, args.dpi, rng),
        lambda: plot_timeline(times, labels, reps, args.output, args.dpi),
    ):
        make()
    if not args.no_show and "agg" not in matplotlib.get_backend().lower():
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
