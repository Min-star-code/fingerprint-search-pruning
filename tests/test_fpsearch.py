import os
import random
import sys

import numpy as np
import pytest
from rdkit import DataStructs

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import fpsearch as fs  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "zinc250k.csv")


@pytest.fixture(scope="module")
def small():
    smiles = fs.load_smiles(DATA, limit=3000)
    fps, _ = fs.fingerprints(smiles)
    db_fps, queries = fps[:2500], fps[2500:]
    return fs.Database(db_fps), db_fps, queries


def reference(query, db_fps, t):
    """Plain Python loop, the slowest and most obviously correct version."""
    out = [(i, DataStructs.TanimotoSimilarity(query, fp)) for i, fp in enumerate(db_fps)]
    return {i: s for i, s in out if s >= t}


@pytest.mark.parametrize("method", fs.METHODS)
@pytest.mark.parametrize("t", [0.2, 0.35, 0.5, 0.7, 0.9, 1.0])
def test_every_method_matches_reference(small, method, t):
    db, db_fps, queries = small
    for q in queries[:25] + db_fps[:5]:  # held-out queries plus exact self-matches
        expected = reference(q, db_fps, t)
        idx, sims, scored = fs.run(db, method, q, fs.pack(q), q.GetNumOnBits(), t)
        assert list(idx) == sorted(expected)
        assert np.allclose(sims, [expected[i] for i in idx], rtol=0, atol=1e-12)
        assert 0 <= scored <= db.n


def test_pruned_methods_score_fewer_molecules_at_high_threshold(small):
    db, db_fps, queries = small
    q = queries[0]
    *_, scored = db.rdkit_pruned(q, 0.9)
    assert scored < db.n


def test_self_match_is_found_at_threshold_one(small):
    db, db_fps, _ = small
    for method in fs.METHODS:
        q = db_fps[7]
        idx, _, _ = fs.run(db, method, q, fs.pack(q), q.GetNumOnBits(), 1.0)
        assert 7 in idx


def test_popcount_bound_holds(small):
    """Tanimoto never exceeds min(a,b)/max(a,b) on real fingerprints."""
    _, db_fps, _ = small
    rng = random.Random(0)
    for _ in range(2000):
        x, y = rng.sample(db_fps, 2)
        a, b = x.GetNumOnBits(), y.GetNumOnBits()
        assert DataStructs.TanimotoSimilarity(x, y) <= min(a, b) / max(a, b) + 1e-12


def test_window_edges():
    assert fs.popcount_window(40, 0.5) == (20, 80)
    assert fs.popcount_window(40, 1.0) == (40, 40)
    assert fs.popcount_window(30, 0.7) == (21, 42)  # 0.7*30 = 21 exactly; 30/0.7 = 42.86


def test_pack_preserves_bits(small):
    _, db_fps, _ = small
    fp = db_fps[3]
    assert int(np.bitwise_count(fs.pack(fp)).sum()) == fp.GetNumOnBits()


def test_sorted_view_is_consistent(small):
    db, _, _ = small
    c = db.counts_sorted
    assert np.all(c[:-1] <= c[1:])
    assert sorted(db.order.tolist()) == list(range(db.n))
    assert np.array_equal(db.packed_sorted, db.packed[db.order])
