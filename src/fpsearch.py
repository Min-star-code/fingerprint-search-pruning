"""Exact Tanimoto similarity search: four ways to get the same answer.

1. rdkit_brute   - score every molecule with RDKit's BulkTanimotoSimilarity.
2. rdkit_pruned  - popcount-bound pruning (Swamidass & Baldi, 2007), then RDKit.
3. numpy_brute   - fingerprints packed into 32 x uint64 words; AND + popcount
                   over the whole database in one vectorised numpy call.
4. numpy_pruned  - numpy scoring on the pruned slice only.

Pruning bound: with a = |A| and b = |B| set bits and c shared bits,
Tanimoto(A, B) = c / (a + b - c) <= min(a, b) / max(a, b).
For threshold t, molecules with b < t*a or b > a/t can never reach t. With the
database sorted by popcount those form two tails that two binary searches cut
off. Every method returns exactly the same hits (checked in tests/ and in the
benchmark).

All methods return (indices, similarities) as numpy arrays sorted by index,
where indices refer to the database's original order.
"""

from __future__ import annotations

import csv
import math

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

RADIUS = 2
N_BITS = 2048
_EPS = 1e-9


def load_smiles(path: str, limit: int | None = None) -> list[str]:
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.append(row["smiles"].strip())
            if limit is not None and len(out) >= limit:
                break
    return out


def fingerprints(smiles: list[str]):
    """Morgan (ECFP4-like) fingerprints. Returns (fps, kept_indices)."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    fps, kept = [], []
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        fp = gen.GetFingerprint(mol)
        if fp.GetNumOnBits() == 0:
            continue
        fps.append(fp)
        kept.append(i)
    return fps, kept


def pack(fp) -> np.ndarray:
    """RDKit bit vector -> 32 uint64 words."""
    bits = np.zeros((N_BITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, bits)
    return np.packbits(bits).view(np.uint64)


def popcount_window(a: int, t: float) -> tuple[int, int]:
    """Smallest and largest popcount b that could still give Tanimoto >= t."""
    if t <= 0:
        return 0, N_BITS
    lo = math.ceil(t * a - _EPS)
    hi = math.floor(a / t + _EPS)
    return lo, hi


def _keep(sims: np.ndarray, t: float, order: np.ndarray | None = None, offset: int = 0):
    """Filter scores >= t; map positions back to original indices."""
    pos = np.flatnonzero(sims >= t)
    idx = pos + offset if order is None else order[pos + offset]
    s = sims[pos]
    k = np.argsort(idx, kind="stable")
    return idx[k].astype(np.int64), s[k]


class Database:
    """Fingerprints held two ways (RDKit objects and packed uint64), sorted by popcount."""

    def __init__(self, fps):
        counts = np.array([fp.GetNumOnBits() for fp in fps], dtype=np.int32)
        self.order = np.argsort(counts, kind="stable")       # sorted pos -> original idx
        self.counts_sorted = counts[self.order]
        # Fresh copies, allocated in the order they will be scanned. RDKit keeps
        # each fingerprint as a separate heap object; scanning a list whose order
        # differs from allocation order was measured at ~3.5x slower (cache
        # misses), so both lists get their own contiguous copies. This gives the
        # RDKit methods their best case.
        self.fps = [DataStructs.ExplicitBitVect(fp.ToBinary()) for fp in fps]          # original order
        self.fps_sorted = [DataStructs.ExplicitBitVect(fps[i].ToBinary()) for i in self.order]
        self.packed = np.stack([pack(fp) for fp in fps])       # original order
        self.packed_sorted = self.packed[self.order]
        self.counts = counts
        self.n = len(fps)

    def window(self, a: int, t: float) -> tuple[int, int]:
        """Slice [start, stop) of the sorted database worth scoring."""
        lo, hi = popcount_window(a, t)
        start = int(np.searchsorted(self.counts_sorted, lo, side="left"))
        stop = int(np.searchsorted(self.counts_sorted, hi, side="right"))
        return start, max(start, stop)

    # --- 1. RDKit, every molecule -------------------------------------------
    def rdkit_brute(self, query, t: float):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(query, self.fps))
        return _keep(sims, t) + (self.n,)

    # --- 2. RDKit on the pruned slice ---------------------------------------
    def rdkit_pruned(self, query, t: float):
        start, stop = self.window(query.GetNumOnBits(), t)
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(query, self.fps_sorted[start:stop]))
        return _keep(sims, t, self.order, start) + (stop - start,)

    # --- 3. numpy packed bits, every molecule -------------------------------
    def numpy_brute(self, qpacked: np.ndarray, qcount: int, t: float):
        c = np.bitwise_count(self.packed & qpacked).sum(axis=1, dtype=np.int32)
        sims = c / (qcount + self.counts - c)
        return _keep(sims, t) + (self.n,)

    # --- 4. numpy on the pruned slice ---------------------------------------
    def numpy_pruned(self, qpacked: np.ndarray, qcount: int, t: float):
        start, stop = self.window(qcount, t)
        block = self.packed_sorted[start:stop]
        c = np.bitwise_count(block & qpacked).sum(axis=1, dtype=np.int32)
        sims = c / (qcount + self.counts_sorted[start:stop] - c)
        return _keep(sims, t, self.order, start) + (stop - start,)


METHODS = ["rdkit_brute", "rdkit_pruned", "numpy_brute", "numpy_pruned"]


def run(db: Database, method: str, query, qpacked, qcount, t):
    if method.startswith("rdkit"):
        return getattr(db, method)(query, t)
    return getattr(db, method)(qpacked, qcount, t)
