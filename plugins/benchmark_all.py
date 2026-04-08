"""
Master benchmark: compare ALL caching algorithms on multiple workloads.

Algorithms tested:
  Baselines:  FIFO, SIEVE (existing)
  Classic:    LRU, CLOCK, LRU-2
  Adaptive:   ARC, 2Q, SLRU
  Modern:     S3-FIFO, S3FIFO+SIEVE, W-TinyLFU, LFU-DA

Workloads:
  1. cloudPhysicsIO.vscsi     – real storage I/O trace (512-byte blocks)
  2. Zipf α=1.2 (10k objs)   – highly skewed web/CDN-like workload
  3. Zipf α=1.0 (10k objs)   – standard internet Zipf distribution
  4. Zipf α=0.7 (10k objs)   – moderate skew (harder for frequency-based)

Run:  python3 benchmark_all.py
"""

import struct, os, sys
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Ensure parent directory is importable so sibling plugin_*.py modules work
sys.path.insert(0, str(Path(__file__).parent))

from libcachesim import PluginCache, TraceReader, TraceType

# ──────────────────────────────────────────────────────────────────────────────
# Import all algorithm hooks
# ──────────────────────────────────────────────────────────────────────────────

import plugin_fifo
import plugin_seive
import plugin_lru
import plugin_clock
import plugin_arc
import plugin_2q
import plugin_s3fifo
import plugin_s3fifo_sieve
import plugin_slru
import plugin_tinylfu
import plugin_lru2
import plugin_lfuda

ALGORITHMS = [
    ("FIFO",          plugin_fifo.cache_init_hook,    plugin_fifo.cache_hit_hook,    plugin_fifo.cache_miss_hook,    plugin_fifo.cache_eviction_hook,    plugin_fifo.cache_remove_hook,    plugin_fifo.cache_free_hook),
    ("SIEVE",         plugin_seive.init_hook,         plugin_seive.hit_hook,         plugin_seive.miss_hook,         plugin_seive.eviction_hook,         plugin_seive.remove_hook,         plugin_seive.free_hook),
    ("LRU",           plugin_lru.init_hook,           plugin_lru.hit_hook,           plugin_lru.miss_hook,           plugin_lru.eviction_hook,           plugin_lru.remove_hook,           plugin_lru.free_hook),
    ("CLOCK",         plugin_clock.init_hook,         plugin_clock.hit_hook,         plugin_clock.miss_hook,         plugin_clock.eviction_hook,         plugin_clock.remove_hook,         plugin_clock.free_hook),
    ("LRU-2",         plugin_lru2.init_hook,          plugin_lru2.hit_hook,          plugin_lru2.miss_hook,          plugin_lru2.eviction_hook,          plugin_lru2.remove_hook,          plugin_lru2.free_hook),
    ("ARC",           plugin_arc.init_hook,           plugin_arc.hit_hook,           plugin_arc.miss_hook,           plugin_arc.eviction_hook,           plugin_arc.remove_hook,           plugin_arc.free_hook),
    ("2Q",            plugin_2q.init_hook,            plugin_2q.hit_hook,            plugin_2q.miss_hook,            plugin_2q.eviction_hook,            plugin_2q.remove_hook,            plugin_2q.free_hook),
    ("SLRU",          plugin_slru.init_hook,          plugin_slru.hit_hook,          plugin_slru.miss_hook,          plugin_slru.eviction_hook,          plugin_slru.remove_hook,          plugin_slru.free_hook),
    ("S3-FIFO",       plugin_s3fifo.init_hook,        plugin_s3fifo.hit_hook,        plugin_s3fifo.miss_hook,        plugin_s3fifo.eviction_hook,        plugin_s3fifo.remove_hook,        plugin_s3fifo.free_hook),
    ("S3FIFO+SIEVE",  plugin_s3fifo_sieve.init_hook,  plugin_s3fifo_sieve.hit_hook,  plugin_s3fifo_sieve.miss_hook,  plugin_s3fifo_sieve.eviction_hook,  plugin_s3fifo_sieve.remove_hook,  plugin_s3fifo_sieve.free_hook),
    ("W-TinyLFU",     plugin_tinylfu.init_hook,       plugin_tinylfu.hit_hook,       plugin_tinylfu.miss_hook,       plugin_tinylfu.eviction_hook,       plugin_tinylfu.remove_hook,       plugin_tinylfu.free_hook),
    ("LFU-DA",        plugin_lfuda.init_hook,         plugin_lfuda.hit_hook,         plugin_lfuda.miss_hook,         plugin_lfuda.eviction_hook,         plugin_lfuda.remove_hook,         plugin_lfuda.free_hook),
]

# ──────────────────────────────────────────────────────────────────────────────
# Workload generation
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"


def gen_zipf_trace(path: str, n_obj: int, n_req: int, alpha: float, seed: int = 42):
    if os.path.exists(path):
        return
    rng = np.random.default_rng(seed)
    np_tmp = np.power(np.arange(1, n_obj + 1), -alpha)
    dist_map = np.cumsum(np_tmp) / np.cumsum(np_tmp)[-1]
    r = rng.uniform(0, 1, n_req)
    reqs = np.searchsorted(dist_map, r) + 1
    s = struct.Struct("<IQIq")
    with open(path, "wb") as f:
        for i, obj in enumerate(reqs):
            f.write(s.pack(i, int(obj), 1, -2))


def prepare_workloads():
    gen_zipf_trace("/tmp/bench_zipf12.bin", 10_000, 500_000, 1.2)
    gen_zipf_trace("/tmp/bench_zipf10.bin", 10_000, 500_000, 1.0)
    gen_zipf_trace("/tmp/bench_zipf07.bin", 10_000, 500_000, 0.7)

    return [
        {
            "name":       "cloudPhysicsIO.vscsi (1MB)",
            "trace":      str(DATA_DIR / "cloudPhysicsIO.vscsi"),
            "trace_type": TraceType.VSCSI_TRACE,
            "cache_size": 1 * 1024 * 1024,
        },
        {
            "name":       "Zipf α=1.2 — 10k objs, 5% cache",
            "trace":      "/tmp/bench_zipf12.bin",
            "trace_type": TraceType.ORACLE_GENERAL_TRACE,
            "cache_size": 500,
        },
        {
            "name":       "Zipf α=1.0 — 10k objs, 5% cache",
            "trace":      "/tmp/bench_zipf10.bin",
            "trace_type": TraceType.ORACLE_GENERAL_TRACE,
            "cache_size": 500,
        },
        {
            "name":       "Zipf α=0.7 — 10k objs, 5% cache",
            "trace":      "/tmp/bench_zipf07.bin",
            "trace_type": TraceType.ORACLE_GENERAL_TRACE,
            "cache_size": 500,
        },
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

def run_all(workloads):
    n_wl = len(workloads)
    # results[algo_name][wl_idx] = req_miss_ratio
    results = {name: [None] * n_wl for name, *_ in ALGORITHMS}

    for wl_idx, wl in enumerate(workloads):
        print(f"\n── Workload: {wl['name']} ──")
        reader_orig = TraceReader(trace=wl["trace"], trace_type=wl["trace_type"])
        for (name, init_h, hit_h, miss_h, evict_h, remove_h, free_h) in ALGORITHMS:
            cache = PluginCache(
                cache_size=wl["cache_size"],
                cache_init_hook=init_h,
                cache_hit_hook=hit_h,
                cache_miss_hook=miss_h,
                cache_eviction_hook=evict_h,
                cache_remove_hook=remove_h,
                cache_free_hook=free_h,
                cache_name=name,
            )
            reader = reader_orig.clone()
            req_mr, byte_mr = cache.process_trace(reader)
            results[name][wl_idx] = req_mr
            print(f"  {name:<18}  req_miss={req_mr:.4f}  byte_miss={byte_mr:.4f}")

    return results


def print_summary(results, workloads):
    wl_names = [wl["name"] for wl in workloads]
    print("\n" + "=" * 90)
    print("SUMMARY — Request Miss Ratio (lower is better)")
    print("=" * 90)

    col_w = 16
    header = f"{'Algorithm':<18}" + "".join(f"{n[:col_w]:>{col_w}}" for n in wl_names)
    print(header)
    print("-" * len(header))

    algo_names = [name for name, *_ in ALGORITHMS]
    for name in algo_names:
        row = f"{name:<18}"
        for idx in range(len(workloads)):
            v = results[name][idx]
            row += f"{v:>{col_w}.4f}" if v is not None else f"{'N/A':>{col_w}}"
        print(row)

    # Rank by average miss ratio
    print("\n" + "=" * 90)
    print("RANKING by average request miss ratio across all workloads (lower = better)")
    print("=" * 90)
    avg = {}
    for name in algo_names:
        vals = [v for v in results[name] if v is not None]
        avg[name] = sum(vals) / len(vals) if vals else float("inf")

    ranked = sorted(algo_names, key=lambda n: avg[n])
    for rank, name in enumerate(ranked, 1):
        baseline_fifo = avg.get("FIFO", 1.0)
        improvement = (baseline_fifo - avg[name]) / baseline_fifo * 100
        print(f"  #{rank:2d}  {name:<18}  avg_miss={avg[name]:.4f}  "
              f"vs FIFO: {improvement:+.1f}%")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Preparing workloads...")
    workloads = prepare_workloads()
    print(f"Running {len(ALGORITHMS)} algorithms × {len(workloads)} workloads...\n")
    results = run_all(workloads)
    print_summary(results, workloads)
