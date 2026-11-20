#!/usr/bin/env python3
"""
extract_frames.py - write the representative frames of each cluster to
separate structure files with MDAnalysis.

Reads the PREFIX_representatives.csv file written by pca_dbscan_gmm.py (or a
list of frame numbers) and extracts those frames from the trajectory. Give the
same trajectory file(s), in the same order, that were given to traj_pca.py:
the frame numbers are 0-based and count the frames of the trajectory as it was
read then.

Example:
    python extract_frames.py -s ref.tpr -f trajectory.xtc -r cluster_representatives.csv
    python extract_frames.py -s ref.tpr -f trajectory.xtc --frames 120 1840 --select protein --format pdb

Output: PREFIX_c<cluster>_r<rank>_f<frame>.gro (or PREFIX_f<frame>.gro with --frames).
"""

import argparse
import csv
import sys
import warnings

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract representative frames from a trajectory (MDAnalysis).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    io = p.add_argument_group("input and output")
    io.add_argument("-s", "--topology", required=True,
                    help="topology or structure file (tpr, gro, pdb, psf, ...) with the same atoms as the trajectory")
    io.add_argument("-f", "--trajectory", required=True, nargs="+",
                    help="trajectory file(s), the same and in the same order as in traj_pca.py")
    io.add_argument("-o", "--output", default="rep", help="prefix of the output files")
    io.add_argument("--format", default="gro", help="format of the output files (gro, pdb, xyz, ...)")

    fr = p.add_argument_group("frames to extract")
    fr.add_argument("-r", "--representatives", default=None,
                    help="csv file written by pca_dbscan_gmm.py (columns cluster, rank, frame)")
    fr.add_argument("--frames", type=int, nargs="+", default=None,
                    help="frame numbers (0-based) to extract instead of, or in addition to, the csv file")
    fr.add_argument("--max-rank", type=int, default=None,
                    help="only the N best representatives of each cluster (1 = the best one only)")
    fr.add_argument("--clusters", type=int, nargs="+", default=None,
                    help="only these clusters")

    at = p.add_argument_group("atoms and periodic boundaries")
    at.add_argument("--select", default="all", help="atoms written to the files (MDAnalysis selection language)")
    at.add_argument("--pbc", choices=["auto", "whole", "none"], default="auto",
                    help="'whole' makes the solute whole, centres it in the box and wraps the other "
                         "molecules around it; 'none' writes the coordinates as they are in the trajectory; "
                         "'auto' uses 'whole' when the trajectory has a box")
    at.add_argument("--solute", default="protein",
                    help="atoms made whole and centred with --pbc whole (MDAnalysis selection language)")
    return p.parse_args()


def ensure_bonds(universe, atoms):
    """Making molecules whole needs bonds. A tpr has them; for a gro or pdb they
    are guessed (from the first frame) for the protein and the selected residues.
    Returns True when the topology came with its own bonds."""
    try:
        if len(universe.bonds) > 0:
            return True
    except Exception:
        pass
    group = universe.select_atoms("protein") | atoms.residues.atoms
    print(f"  the topology has no bonds: guessing them for {group.n_atoms} atoms")
    try:
        group.guess_bonds()
    except Exception as err:
        sys.exit(f"ERROR: could not guess the bonds ({err}).\n"
                 "Use a tpr file as topology, or run with --pbc none.")
    return False


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


class CentreSolute:
    """Makes the molecules of the solute whole, keeps them together, moves them
    to the centre of the box and wraps the remaining molecules around them.
    With bonds from the topology (tpr) the other molecules are kept whole;
    without them their atoms are wrapped one by one."""

    def __init__(self, universe, solute, bonded):
        fragments = sorted(solute.fragments, key=lambda f: f.n_atoms, reverse=True)
        self.fragments = fragments
        self.group = fragments[0]
        for frag in fragments[1:]:
            self.group = self.group | frag
        self.all = universe.atoms
        self.rest = universe.atoms - self.group
        self.bonded = bonded
        self.whole = Whole(self.group)
        self.whole_rest = Whole(self.rest) if (bonded and self.rest.n_atoms) else None

    def __call__(self, ts):
        from MDAnalysis.lib.distances import minimize_vectors
        from MDAnalysis.lib.mdamath import triclinic_vectors
        positions = ts.positions
        self.whole.apply(positions, ts.dimensions)
        if self.whole_rest is not None:
            self.whole_rest.apply(positions, ts.dimensions)
        ts.positions = positions
        if len(self.fragments) > 1:
            anchor = self.fragments[0].center_of_geometry()
            for frag in self.fragments[1:]:
                delta = (frag.center_of_geometry() - anchor).astype(np.float32)
                shift = minimize_vectors(delta[None, :], ts.dimensions)[0] - delta
                if np.any(np.abs(shift) > 1e-3):
                    frag.translate(shift)
        box_centre = triclinic_vectors(ts.dimensions).sum(axis=0) / 2.0
        self.all.translate(box_centre - self.group.center_of_geometry())
        if self.rest.n_atoms and self.bonded:
            self.rest.wrap(compound="fragments", center="cog", inplace=True)
        elif self.rest.n_atoms:
            self.rest.wrap(compound="atoms", inplace=True)
        return ts


def read_representatives(path, max_rank, clusters):
    """Returns a list of (frame, label) from the csv file of pca_dbscan_gmm.py."""
    wanted = []
    try:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                cluster, rank, frame = int(row["cluster"]), int(row["rank"]), int(row["frame"])
                if max_rank is not None and rank > max_rank:
                    continue
                if clusters is not None and cluster not in clusters:
                    continue
                wanted.append((frame, f"c{cluster}_r{rank}_f{frame}"))
    except (OSError, KeyError, ValueError) as err:
        sys.exit(f"ERROR: could not read {path} ({err!r}). It needs the columns cluster, rank and frame.")
    return wanted


def main():
    args = parse_args()
    import MDAnalysis as mda
    warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="MDAnalysis")

    wanted = []
    if args.representatives:
        wanted += read_representatives(args.representatives, args.max_rank, args.clusters)
    if args.frames:
        wanted += [(f, f"f{f}") for f in args.frames]
    if not wanted:
        sys.exit("ERROR: no frames to extract. Give -r FILE and/or --frames.")

    print(f"Reading {args.topology} and {len(args.trajectory)} trajectory file(s)")
    universe = mda.Universe(args.topology, *args.trajectory)
    n_total = universe.trajectory.n_frames
    print(f"  {universe.atoms.n_atoms} atoms, {n_total} frames")
    bad = [f for f, _ in wanted if not 0 <= f < n_total]
    if bad:
        sys.exit(f"ERROR: frames {bad} are not in this trajectory (it has frames 0 to {n_total - 1}).\n"
                 "Give the same trajectory file(s), in the same order, as in traj_pca.py.")

    try:
        out_atoms = universe.select_atoms(args.select)
    except Exception as err:
        sys.exit(f"ERROR: selection not understood ({err})")
    if out_atoms.n_atoms == 0:
        sys.exit(f"ERROR: the selection \"{args.select}\" has no atoms.")
    print(f"  writing \"{args.select}\": {out_atoms.n_atoms} atoms")

    dims = universe.trajectory.ts.dimensions
    has_box = dims is not None and np.all(dims[:3] > 0)
    pbc = args.pbc
    if pbc == "auto":
        pbc = "whole" if has_box else "none"
    if pbc == "whole" and not has_box:
        sys.exit("ERROR: --pbc whole needs a trajectory with box dimensions.")
    print(f"Periodic boundaries: {pbc}")
    if pbc == "whole":
        try:
            solute = universe.select_atoms(args.solute)
        except Exception as err:
            sys.exit(f"ERROR: --solute selection not understood ({err})")
        if solute.n_atoms == 0:
            sys.exit(f"ERROR: the --solute selection \"{args.solute}\" has no atoms. "
                     "Set --solute to the molecules to centre, or use --pbc none.")
        bonded = ensure_bonds(universe, solute)
        centre = CentreSolute(universe, solute, bonded)
        print(f"  solute \"{args.solute}\": {len(centre.fragments)} molecule(s), {centre.group.n_atoms} atoms, "
              "made whole and centred")
        if centre.rest.n_atoms and not bonded:
            print("  the other molecules are wrapped atom by atom (use a tpr topology to keep them whole)")
        universe.trajectory.add_transformations(centre)

    print("Files written")
    for frame, label in sorted(wanted):
        ts = universe.trajectory[frame]
        name = f"{args.output}_{label}.{args.format}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out_atoms.write(name)
        print(f"  {name}   (frame {frame}, time {ts.time:.1f} ps)")


if __name__ == "__main__":
    main()
