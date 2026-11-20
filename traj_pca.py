#!/usr/bin/env python3
"""
traj_pca.py - Cartesian PCA of an MD trajectory with MDAnalysis.

Replaces the gmx trjconv / gmx covar / gmx anaeig steps: the trajectory is read
with MDAnalysis, the molecules that hold the selected atoms are made whole
across the periodic boundaries, every frame is fitted (rotation + translation)
on a selection of your choice, and a principal component analysis is run on the
Cartesian coordinates of the selection you want to analyse.

Example:
    python traj_pca.py -s ref.tpr -f trajectory.xtc \
        --select "backbone and same residue as around 10 (resid 160 185 237)" \
        -o pca

Outputs (PREFIX is set with -o):
    PREFIX.npz              projections, variances, frame numbers and metadata
                            (input of pca_dbscan_gmm.py)
    PREFIX_projections.csv  frame, time and the first PCs in plain text
    PREFIX_overview.png     explained variance and PC1/PC2 map (600 dpi)
    PREFIX_pcN.pdb          optional, motion along each PC (--modes)

Frame numbers are 0-based and count the frames of the trajectory as it is read
(all the -f files together), which is how MDAnalysis numbers them.
Distances are in angstrom.
"""

import argparse
import sys
import warnings

import numpy as np


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Cartesian PCA of an MD trajectory on a chosen atom selection (MDAnalysis).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    io = p.add_argument_group("input and output")
    io.add_argument("-s", "--topology", required=True,
                    help="topology or structure file (tpr, gro, pdb, psf, ...) with the same atoms as the trajectory")
    io.add_argument("-f", "--trajectory", required=True, nargs="+",
                    help="trajectory file(s) (xtc, trr, dcd, nc, ...); several files are read one after the other")
    io.add_argument("-o", "--output", default="pca", help="prefix of the output files")

    sel = p.add_argument_group("atom selections (MDAnalysis selection language)")
    sel.add_argument("--select", default="backbone",
                     help="atoms used in the PCA")
    sel.add_argument("--fit-select", default=None,
                     help="atoms used to fit the frames (default: the same as --select)")
    sel.add_argument("--check-selection", action="store_true",
                     help="print the selections, write them to PREFIX_selection.pdb and stop")

    fr = p.add_argument_group("frames")
    fr.add_argument("-b", "--start", type=int, default=None, help="first frame to read (0-based)")
    fr.add_argument("-e", "--stop", type=int, default=None, help="stop before this frame")
    fr.add_argument("-dt", "--step", type=int, default=1, help="read every N-th frame")

    an = p.add_argument_group("analysis")
    an.add_argument("--pbc", choices=["auto", "whole", "none"], default="auto",
                    help="periodic boundaries: 'whole' makes the molecules that hold the selected atoms whole "
                         "and keeps them together, 'none' reads the coordinates as they are, "
                         "'auto' uses 'whole' when the trajectory has a box")
    an.add_argument("--ref", choices=["average", "first"], default="average",
                    help="structure the frames are fitted to: the average structure (found iteratively) "
                         "or the first frame read")
    an.add_argument("--n-pcs", type=int, default=3,
                    help="number of PCs written to the csv file and listed on screen")
    an.add_argument("--keep-variance", type=float, default=0.99,
                    help="the npz file keeps the PCs needed to reach this fraction of the total variance "
                         "(they are used later to compare whole structures, not only the first PCs)")
    an.add_argument("--modes", type=int, default=0,
                    help="write the motion along the first N PCs to multi-model pdb files")
    an.add_argument("--mode-frames", type=int, default=21,
                    help="number of models in each of those pdb files")

    pl = p.add_argument_group("plots")
    pl.add_argument("--no-show", action="store_true", help="save the figure without opening a window")
    pl.add_argument("--dpi", type=int, default=600, help="resolution of the saved figure")
    return p.parse_args()


# --------------------------------------------------------------------------
# Periodic boundaries
# --------------------------------------------------------------------------

def ensure_bonds(universe, atoms):
    """Making molecules whole needs bonds. A tpr has them; for a gro or pdb they
    are guessed (from the first frame) for the protein and the selected residues."""
    try:
        if len(universe.bonds) > 0:
            return
    except Exception:
        pass
    group = universe.select_atoms("protein") | atoms.residues.atoms
    print(f"  the topology has no bonds: guessing them for {group.n_atoms} atoms "
          "(protein and selected residues)")
    try:
        group.guess_bonds()
    except Exception as err:
        sys.exit(f"ERROR: could not guess the bonds ({err}).\n"
                 "Use a tpr file as topology, or make the molecules whole beforehand and run with --pbc none.")


class Whole:
    """Makes molecules whole along their bonds, for any box shape.

    A spanning tree of the bond graph is built once. In every frame each bond
    vector is reduced to its minimum image and the positions are rebuilt from
    the root of each molecule by summing those vectors along the tree (pointer
    jumping, so it takes log2(depth) vectorised steps). The result only needs
    the bonds to be shorter than half the box, whatever the size of the molecule."""

    def __init__(self, atoms):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import breadth_first_order, connected_components
        self.idx = np.sort(atoms.indices)
        n = len(self.idx)
        pairs = np.array([b.indices for b in atoms.bonds], dtype=np.int64).reshape(-1, 2)
        if len(pairs):
            inside = np.isin(pairs, self.idx).all(axis=1)
            pairs = np.searchsorted(self.idx, pairs[inside])
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)).tocsr()
        n_comp, comp = connected_components(graph, directed=False)
        roots = np.unique(comp, return_index=True)[1]
        # one extra node joined to the first atom of every molecule: a single BFS covers them all
        extra = coo_matrix((np.ones(len(roots)), (np.full(len(roots), n), roots)), shape=(n + 1, n + 1))
        big = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n + 1, n + 1)) + extra
        _, pred = breadth_first_order(big.tocsr(), n, directed=False, return_predecessors=True)
        parent = pred[:n].astype(np.int64)
        parent[roots] = roots
        self.parent = parent
        self.child = np.where(parent != np.arange(n))[0]
        self.n_molecules = n_comp

    def apply(self, positions, box):
        """positions: all the positions of the frame (modified in place)."""
        from MDAnalysis.lib.distances import minimize_vectors
        if len(self.child) == 0:
            return positions
        x = positions[self.idx].astype(np.float64)
        acc = np.zeros_like(x)
        vec = (x[self.child] - x[self.parent[self.child]]).astype(np.float32)
        acc[self.child] = minimize_vectors(vec, box)
        anc = self.parent.copy()
        while True:
            nxt = anc[anc]
            if np.array_equal(nxt, anc):
                break
            acc = acc + acc[anc]
            anc = nxt
        positions[self.idx] = x[anc] + acc
        return positions


class MakeWhole:
    """On-the-fly transformation: makes whole every molecule that holds at least
    one of the given atoms, then moves those molecules to the periodic image
    closest to the largest one, so that a complex is not split over the box."""

    def __init__(self, atoms):
        fragments = sorted(atoms.fragments, key=lambda f: f.n_atoms, reverse=True)
        self.fragments = fragments
        self.group = fragments[0]
        for frag in fragments[1:]:
            self.group = self.group | frag
        self.whole = Whole(self.group)

    def __call__(self, ts):
        from MDAnalysis.lib.distances import minimize_vectors
        if ts.dimensions is None or not np.all(ts.dimensions[:3] > 0):
            return ts
        positions = ts.positions
        self.whole.apply(positions, ts.dimensions)
        ts.positions = positions
        if len(self.fragments) > 1:
            anchor = self.fragments[0].center_of_geometry()
            for frag in self.fragments[1:]:
                delta = (frag.center_of_geometry() - anchor).astype(np.float32)
                shift = minimize_vectors(delta[None, :], ts.dimensions)[0] - delta
                if np.any(np.abs(shift) > 1e-3):
                    frag.translate(shift)
        return ts


# --------------------------------------------------------------------------
# Fitting and PCA
# --------------------------------------------------------------------------

def fit_rotations(mobile, ref):
    """Rotation matrices that superpose each frame of mobile (F, N, 3) on ref (N, 3).
    Both are centred on their centroid. The fitted coordinates are mobile @ R."""
    h = np.einsum("fni,nj->fij", mobile, ref)
    u, _, vt = np.linalg.svd(h)
    det = np.sign(np.linalg.det(u @ vt))
    u[:, :, -1] *= det[:, None]          # no mirror images
    return u @ vt


def align(fit_xyz, sel_xyz, ref_mode, tol=1e-4, max_iter=50, chunk=2000):
    """Fits all frames on the reference (first frame or average structure) using
    fit_xyz, and applies the same translation and rotation to sel_xyz (in place).
    Returns the number of iterations and the RMSD of each frame to the reference."""
    n_frames = fit_xyz.shape[0]
    centroids = fit_xyz.mean(axis=1, keepdims=True)
    same = sel_xyz is fit_xyz
    fit_xyz -= centroids
    if not same:
        sel_xyz -= centroids

    ref = fit_xyz[0].astype(np.float64)
    rotations = np.empty((n_frames, 3, 3))
    iterations = 0
    for iterations in range(1, max_iter + 1):
        mean = np.zeros_like(ref)
        for i in range(0, n_frames, chunk):
            block = fit_xyz[i:i + chunk].astype(np.float64)
            rotations[i:i + chunk] = fit_rotations(block, ref)
            mean += (block @ rotations[i:i + chunk]).sum(axis=0)
        if ref_mode == "first":
            break
        mean /= n_frames
        change = np.sqrt(((mean - ref) ** 2).sum(axis=1).mean())
        ref = mean
        if change < tol:
            break

    rmsd = np.empty(n_frames)
    for i in range(0, n_frames, chunk):
        rot = rotations[i:i + chunk]
        fitted = fit_xyz[i:i + chunk].astype(np.float64) @ rot
        rmsd[i:i + chunk] = np.sqrt(((fitted - ref) ** 2).sum(axis=2).mean(axis=1))
        if same:
            fit_xyz[i:i + chunk] = fitted
        else:
            sel_xyz[i:i + chunk] = sel_xyz[i:i + chunk].astype(np.float64) @ rot
    return iterations, rmsd


def pca(x):
    """PCA of the centred matrix x (frames, coordinates).
    Returns the eigenvalues (variances, largest first) and the eigenvectors as columns."""
    n_frames, n_dim = x.shape
    if n_dim <= n_frames:
        cov = x.T @ x / (n_frames - 1)
        values, vectors = np.linalg.eigh(cov)
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
    else:
        # fewer frames than coordinates: diagonalise the (frames x frames) matrix instead
        gram = x @ x.T / (n_frames - 1)
        values, u = np.linalg.eigh(gram)
        order = np.argsort(values)[::-1]
        values, u = values[order], u[:, order]
        keep = values > 1e-10 * values[0]
        values, u = values[keep], u[:, keep]
        vectors = x.T @ u / np.sqrt(values * (n_frames - 1))
    values = np.clip(values, 0.0, None)
    # fixed sign: the largest component of each eigenvector is positive
    idx = np.abs(vectors).argmax(axis=0)
    vectors *= np.sign(vectors[idx, np.arange(vectors.shape[1])])
    return values, vectors


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def describe(label, text, group):
    res = group.residues
    if res.n_residues == 0:
        span = "no residues"
    else:
        span = f"{res[0].resname}{res[0].resid} to {res[-1].resname}{res[-1].resid}"
    print(f"  {label:<14} \"{text}\"")
    print(f"  {'':<14} {group.n_atoms} atoms in {res.n_residues} residues ({span})")


def write_modes(atoms, mean, vectors, proj, n_modes, n_models, prefix):
    import MDAnalysis as mda
    for k in range(min(n_modes, vectors.shape[1])):
        name = f"{prefix}_pc{k + 1}.pdb"
        amplitudes = np.linspace(proj[:, k].min(), proj[:, k].max(), n_models)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with mda.Writer(name, atoms.n_atoms, multiframe=True) as out:
                for a in amplitudes:
                    atoms.positions = (mean + a * vectors[:, k]).reshape(-1, 3)
                    out.write(atoms)
        print(f"  {name}  (PC{k + 1}, {n_models} models from the lowest to the highest projection)")


def plot_overview(ratio, proj, times, prefix, dpi, show):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    ink, muted, grid, blue = "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))

    n_show = min(15, len(ratio))
    x = np.arange(1, n_show + 1)
    ax1.bar(x, 100 * ratio[:n_show], width=0.6, color=blue, label="each PC")
    ax1.plot(x, 100 * np.cumsum(ratio)[:n_show], color=ink, lw=1.5, marker="o", ms=3.5, label="cumulative")
    ax1.set_xticks(x)
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Variance explained (%)")
    ax1.set_ylim(0, 100)
    ax1.grid(axis="y", color=grid, lw=0.6)
    ax1.set_axisbelow(True)
    ax1.legend(frameon=False, loc="center right")
    ax1.set_title("Explained variance", loc="left", fontsize=10, color=ink)

    in_ns = times.max() >= 1000.0
    t = times / 1000.0 if in_ns else times
    cmap = LinearSegmentedColormap.from_list("time", ["#c9dcf4", "#2a78d6", "#0d2f5c"])
    order = np.arange(len(t))
    sc = ax2.scatter(proj[order, 0], proj[order, 1], c=t[order], cmap=cmap, s=5, lw=0, rasterized=True)
    cb = fig.colorbar(sc, ax=ax2, pad=0.02)
    cb.set_label(f"Time ({'ns' if in_ns else 'ps'})")
    cb.outline.set_visible(False)
    ax2.set_xlabel(f"PC1 ({100 * ratio[0]:.1f}%)  /  Å")
    ax2.set_ylabel(f"PC2 ({100 * ratio[1]:.1f}%)  /  Å")
    ax2.grid(color=grid, lw=0.6)
    ax2.set_axisbelow(True)
    ax2.set_title("Projection on PC1 and PC2", loc="left", fontsize=10, color=ink)

    fig.tight_layout()
    name = f"{prefix}_overview.png"
    fig.savefig(name, dpi=dpi, facecolor="white")
    print(f"  {name}")
    if show and "agg" not in matplotlib.get_backend().lower():
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    import MDAnalysis as mda

    warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="MDAnalysis")

    print(f"Reading {args.topology} and {len(args.trajectory)} trajectory file(s)")
    universe = mda.Universe(args.topology, *args.trajectory)
    n_total = universe.trajectory.n_frames
    print(f"  {universe.atoms.n_atoms} atoms, {n_total} frames")

    # ---- selections (evaluated once, on the first frame)
    fit_text = args.fit_select or args.select
    try:
        sel = universe.select_atoms(args.select)
        fit = universe.select_atoms(fit_text)
    except Exception as err:
        sys.exit(f"ERROR: selection not understood ({err})")
    print("Selections")
    describe("PCA:", args.select, sel)
    describe("fit:", fit_text, fit)
    if sel.n_atoms < 2 or fit.n_atoms < 3:
        sys.exit("ERROR: the PCA selection needs at least 2 atoms and the fit selection at least 3.")
    same = sel.n_atoms == fit.n_atoms and np.array_equal(sel.indices, fit.indices)

    if args.check_selection:
        name = f"{args.output}_selection.pdb"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (sel | fit).write(name)
        print(f"Selections written to {name} (first frame). Nothing else was done.")
        return

    # ---- periodic boundaries
    has_box = universe.trajectory.ts.dimensions is not None and np.all(universe.trajectory.ts.dimensions[:3] > 0)
    pbc = args.pbc
    if pbc == "auto":
        pbc = "whole" if has_box else "none"
    if pbc == "whole" and not has_box:
        sys.exit("ERROR: --pbc whole needs a trajectory with box dimensions.")
    print(f"Periodic boundaries: {pbc}")
    if pbc == "whole":
        ensure_bonds(universe, sel | fit)
        whole = MakeWhole(sel | fit)
        print(f"  {len(whole.fragments)} molecule(s), {whole.group.n_atoms} atoms, made whole in every frame")
        universe.trajectory.add_transformations(whole)

    # ---- read the coordinates of the selected atoms
    frames = np.arange(n_total)[slice(args.start, args.stop, args.step)]
    n_frames = len(frames)
    if n_frames < 3:
        sys.exit("ERROR: fewer than 3 frames to analyse.")
    print(f"Reading {n_frames} frames ({frames[0]} to {frames[-1]}, every {args.step})")
    sel_xyz = np.empty((n_frames, sel.n_atoms, 3), dtype=np.float32)
    fit_xyz = sel_xyz if same else np.empty((n_frames, fit.n_atoms, 3), dtype=np.float32)
    times = np.empty(n_frames)
    progress = sys.stdout.isatty()
    report = max(1, n_frames // 100)
    for i, ts in enumerate(universe.trajectory[args.start:args.stop:args.step]):
        sel_xyz[i] = sel.positions
        if not same:
            fit_xyz[i] = fit.positions
        times[i] = ts.time
        if progress and ((i + 1) % report == 0 or i + 1 == n_frames):
            print(f"\r  frame {i + 1} of {n_frames}", end="", flush=True)
    if progress:
        print()

    # a selection split over periodic images shows up as a sudden jump in its size
    if has_box:
        centred = sel_xyz - sel_xyz.mean(axis=1, keepdims=True)
        rgyr = np.sqrt((centred.astype(np.float64) ** 2).sum(axis=2).mean(axis=1))
        del centred
        jump = np.abs(np.diff(rgyr)).max()
        if jump > 0.3 * np.median(rgyr):
            print("  WARNING: the size of the selection jumps between consecutive frames (radius of gyration "
                  f"changes by up to {jump:.1f} A, median {np.median(rgyr):.1f} A).\n"
                  "           It may be split over periodic images: check --pbc and the topology.")

    # ---- fit
    iterations, rmsd = align(fit_xyz, sel_xyz, args.ref)
    if args.ref == "average":
        print(f"Fit on the average structure (converged in {iterations} iterations)")
    else:
        print(f"Fit on the first frame read (frame {frames[0]})")
    print(f"  RMSD of the fit atoms to the reference: mean {rmsd.mean():.2f} A, max {rmsd.max():.2f} A")

    # ---- PCA
    x = sel_xyz.reshape(n_frames, -1).astype(np.float64)
    mean = x.mean(axis=0)
    x -= mean
    values, vectors = pca(x)
    ratio = values / values.sum()
    cumulative = np.cumsum(ratio)
    n_keep = int(min(len(values), max(np.searchsorted(cumulative, args.keep_variance) + 1,
                                      min(len(values), max(10, args.n_pcs)))))
    proj = x @ vectors[:, :n_keep]

    n_list = min(args.n_pcs, n_keep)
    print(f"PCA on {sel.n_atoms} atoms ({x.shape[1]} coordinates)")
    print("  PC   variance (A^2)   explained   cumulative")
    for k in range(n_list):
        print(f"  {k + 1:<4d} {values[k]:12.3f}   {100 * ratio[k]:8.1f}%   {100 * cumulative[k]:8.1f}%")
    print(f"  {n_keep} PCs kept in the npz file ({100 * cumulative[n_keep - 1]:.1f}% of the variance)")

    # ---- outputs
    print("Files written")
    name = f"{args.output}.npz"
    n_vec = min(10, n_keep)
    np.savez_compressed(
        name,
        projections=proj.astype(np.float32),
        frames=frames.astype(np.int64),
        times=times,
        variance=values[:n_keep],
        variance_ratio=ratio[:n_keep],
        total_variance=values.sum(),
        n_atoms=sel.n_atoms,
        mean=mean.reshape(-1, 3),
        eigenvectors=vectors[:, :n_vec].T,
        select=args.select,
        fit_select=fit_text,
        topology=args.topology,
        trajectory=np.array(args.trajectory),
    )
    print(f"  {name}")

    name = f"{args.output}_projections.csv"
    header = "frame,time_ps," + ",".join(f"PC{k + 1}" for k in range(n_list))
    table = np.column_stack([frames, times, proj[:, :n_list]])
    np.savetxt(name, table, delimiter=",", header=header, comments="",
               fmt=["%d", "%.3f"] + ["%.4f"] * n_list)
    print(f"  {name}")

    if args.modes > 0:
        write_modes(sel, mean, vectors, proj, args.modes, args.mode_frames, args.output)

    plot_overview(ratio, proj, times, args.output, args.dpi, not args.no_show)


if __name__ == "__main__":
    main()
