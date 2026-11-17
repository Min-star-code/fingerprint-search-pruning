"""Benchmark brute-force vs popcount-pruned Tanimoto search on ZINC 250k.

Writes results/benchmark.json, used by the interactive page and the report.
Run:  .venv/bin/python src/benchmark.py
"""

import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time

import numpy as np
import rdkit

sys.path.insert(0, os.path.dirname(__file__))
import fpsearch as fs  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data", "zinc250k.csv")
OUT = os.path.join(ROOT, "results", "benchmark.json")

SEED = 42
N_HELDOUT = 1000      # molecules kept out of the database to use as queries
N_QUERIES = 100
THRESHOLDS = [round(0.30 + 0.05 * k, 2) for k in range(14)]  # 0.30 .. 0.95
REPEATS = 3           # timing = best of 3 per query, to reduce noise


def best_time(fn, repeats=REPEATS):
    best, out = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def cpu_name():
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        return platform.processor()


def main():
    t0 = time.perf_counter()
    smiles = fs.load_smiles(DATA)
    fps, kept = fs.fingerprints(smiles)
    fp_seconds = time.perf_counter() - t0
    print(f"{len(smiles)} SMILES -> {len(fps)} fingerprints in {fp_seconds:.1f}s")

    rng = random.Random(SEED)
    idx = list(range(len(fps)))
    rng.shuffle(idx)
    held, db_idx = idx[:N_HELDOUT], idx[N_HELDOUT:]
    t0 = time.perf_counter()
    db = fs.Database([fps[i] for i in db_idx])
    index_seconds = time.perf_counter() - t0
    queries = [fps[i] for i in held[:N_QUERIES]]
    qpacked = [fs.pack(q) for q in queries]
    qcounts = [q.GetNumOnBits() for q in queries]
    counts = db.counts_sorted

    # Side experiment: same RDKit call, same fingerprints, different memory layout.
    from rdkit import DataStructs
    scattered = [fps[i] for i in db_idx]  # original objects, list order != allocation order
    lay_s, lay_c = [], []
    for q in queries[:30]:
        lay_s.append(best_time(lambda: DataStructs.BulkTanimotoSimilarity(q, scattered))[0])
        lay_c.append(best_time(lambda: DataStructs.BulkTanimotoSimilarity(q, db.fps))[0])
    layout = {"queries": 30,
              "scattered_ms": 1000 * statistics.mean(lay_s),
              "contiguous_ms": 1000 * statistics.mean(lay_c)}
    print("layout", layout, flush=True)

    rows = []
    mismatches = 0
    for t in THRESHOLDS:
        times = {m: 0.0 for m in fs.METHODS}
        scanned = {m: [] for m in fs.METHODS}
        hits = []
        for qi, q in enumerate(queries):
            ref = None
            for m in fs.METHODS:
                sec, (ix, sims, n_scored) = best_time(
                    lambda: fs.run(db, m, q, qpacked[qi], qcounts[qi], t))
                times[m] += sec
                scanned[m].append(n_scored / db.n)
                if ref is None:
                    ref = (ix, sims)
                elif not (np.array_equal(ix, ref[0]) and np.array_equal(sims, ref[1])):
                    mismatches += 1
            hits.append(len(ref[0]))
        row = {"threshold": t, "mean_hits": statistics.mean(hits),
               "median_hits": statistics.median(hits)}
        for m in fs.METHODS:
            row[f"{m}_ms"] = 1000 * times[m] / len(queries)
            row[f"{m}_scanned"] = statistics.mean(scanned[m])
            row[f"{m}_speedup"] = times["rdkit_brute"] / times[m]
        rows.append(row)
        print(f"t={t:.2f} " + "  ".join(f"{m} {row[m + '_ms']:6.2f}ms" for m in fs.METHODS)
              + f"  scanned {100 * row['rdkit_pruned_scanned']:5.1f}%  hits {row['mean_hits']:.1f}", flush=True)

    hist_edges = list(range(0, int(counts.max()) + 2))
    hist, _ = np.histogram(counts, bins=hist_edges)

    result = {
        "dataset": {
            "name": "ZINC 250k (drug-like subset used in Gomez-Bombarelli et al., 2018)",
            "source": "https://github.com/aspuru-guzik-group/chemical_vae (250k_rndm_zinc_drugs_clean_3.csv)",
            "molecules_read": len(smiles),
            "fingerprints": len(fps),
            "database_size": db.n,
            "queries": len(queries),
            "heldout_pool": N_HELDOUT,
        },
        "fingerprint": {"type": "Morgan", "radius": fs.RADIUS, "bits": fs.N_BITS},
        "timing": {
            "fingerprint_seconds": fp_seconds,
            "index_build_seconds": index_seconds,
            "method": f"best of {REPEATS} runs per query, single thread",
            "packed_db_megabytes": db.packed.nbytes / 1e6,
        },
        "correctness": {
            "checks": len(THRESHOLDS) * len(queries) * (len(fs.METHODS) - 1),
            "compared_against": "rdkit_brute (exact equality of indices and float scores)",
            "mismatched_result_sets": mismatches,
        },
        "popcount": {
            "db_min": int(counts.min()), "db_max": int(counts.max()),
            "db_median": float(np.median(counts)),
            "histogram": hist.tolist(),
            "query_counts": qcounts,
        },
        "rows": rows,
        "layout": layout,
        "methods": fs.METHODS,
        "environment": {
            "python": platform.python_version(),
            "rdkit": rdkit.__version__,
            "numpy": np.__version__,
            "cpu": cpu_name(),
            "os": platform.platform(),
        },
        "seed": SEED,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=1)
    print(f"mismatches: {mismatches} / {len(THRESHOLDS) * len(queries) * 3}  -> {OUT}")


if __name__ == "__main__":
    main()
