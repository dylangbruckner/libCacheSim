"""
Tune S3-FIFO and S3FIFO+SIEVE small-queue ratio.

Standard S3-FIFO uses 10% small / 90% main.  This script sweeps the
split ratio (2% … 40%) on multiple workload types to find the best setting
across different access patterns.

Run:  python3 tune_s3fifo.py
"""

import struct, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from collections import deque, OrderedDict
from libcachesim import CommonCacheParams, Request, PluginCache, TraceReader, TraceType

DATA_DIR = Path(__file__).parent.parent / "data"

# ─────────────────────────────────────────────────────────────────────────────
# Parameterised S3-FIFO
# ─────────────────────────────────────────────────────────────────────────────

def make_s3fifo(small_ratio: float):
    """Return (init_hook, hit_hook, miss_hook, eviction_hook, remove_hook, free_hook)
    for S3-FIFO with the given small-queue ratio (e.g. 0.10 = 10%)."""

    class S3FIFOCache:
        def __init__(self, cache_size: int):
            self.cache_size = cache_size
            self.small_max  = max(1, int(cache_size * small_ratio))
            self.ghost_max  = cache_size

            self.small: deque = deque()
            self.main:  deque = deque()
            self.ghost: deque = deque()
            self.obj_info: dict = {}
            self.ghost_set: set = set()
            self.small_bytes: int = 0
            self.main_bytes:  int = 0
            self.ghost_bytes: int = 0

        def _add_to_ghost(self, obj_id, size):
            if obj_id in self.ghost_set:
                return
            self.ghost.append(obj_id)
            self.ghost_set.add(obj_id)
            self.ghost_bytes += size
            while self.ghost_bytes > self.ghost_max and self.ghost:
                old = self.ghost.popleft()
                self.ghost_set.discard(old)
                self.ghost_bytes = max(0, self.ghost_bytes - size)

        def on_hit(self, req):
            if req.obj_id in self.obj_info:
                sz, fr = self.obj_info[req.obj_id]
                self.obj_info[req.obj_id] = (sz, min(fr + 1, 3))

        def on_miss(self, req):
            oid, sz = req.obj_id, req.obj_size
            if sz > self.cache_size or oid in self.obj_info:
                return
            if oid in self.ghost_set:
                self.ghost_set.discard(oid)
                self.main.appendleft(oid)
                self.obj_info[oid] = (sz, 0)
                self.main_bytes += sz
            else:
                self.small.append(oid)
                self.obj_info[oid] = (sz, 0)
                self.small_bytes += sz

        def _evict_small_one(self):
            """Pop from small: freq=0 → evict (return id); freq>=1 → promote to main."""
            while self.small:
                oid = self.small.popleft()
                if oid not in self.obj_info: continue
                sz, fr = self.obj_info[oid]
                self.small_bytes -= sz
                if fr == 0:
                    del self.obj_info[oid]
                    self._add_to_ghost(oid, sz)
                    return oid
                self.obj_info[oid] = (sz, 0)
                self.main.appendleft(oid)
                self.main_bytes += sz
            return 0

        def _evict_main_one(self):
            """Sweep main: freq=0 → evict; freq>=1 → decrement, reinsert at head."""
            scanned, limit = 0, len(self.main)
            while self.main and scanned <= limit:
                oid = self.main.pop(); scanned += 1
                if oid not in self.obj_info: continue
                sz, fr = self.obj_info[oid]
                self.main_bytes -= sz
                if fr == 0:
                    del self.obj_info[oid]
                    return oid
                self.obj_info[oid] = (sz, fr - 1)
                self.main.appendleft(oid)
                self.main_bytes += sz
            return 0

        def evict(self, req):
            # Enforce the small/main ratio:
            # if main has grown over its target (because of promotions),
            # bleed from main first; otherwise drain from small.
            main_max = self.cache_size - self.small_max
            if self.main_bytes > main_max and self.main:
                v = self._evict_main_one()
                if v: return v
            v = self._evict_small_one()
            if v: return v
            return self._evict_main_one()

        def on_remove(self, obj_id):
            if obj_id not in self.obj_info: return
            sz, _ = self.obj_info.pop(obj_id)
            try: self.small.remove(obj_id); self.small_bytes -= sz
            except ValueError:
                try: self.main.remove(obj_id); self.main_bytes -= sz
                except ValueError: pass
            self.ghost_set.discard(obj_id)

    def init_hook(p): return S3FIFOCache(p.cache_size)
    def hit_hook(d, r): d.on_hit(r)
    def miss_hook(d, r): d.on_miss(r)
    def eviction_hook(d, r): return d.evict(r)
    def remove_hook(d, oid): d.on_remove(oid)
    def free_hook(d): d.small.clear(); d.main.clear(); d.obj_info.clear(); d.ghost_set.clear()

    return init_hook, hit_hook, miss_hook, eviction_hook, remove_hook, free_hook


# ─────────────────────────────────────────────────────────────────────────────
# Parameterised S3FIFO+SIEVE hybrid
# ─────────────────────────────────────────────────────────────────────────────

def make_s3fifo_sieve(small_ratio: float):
    """S3-FIFO with SIEVE main queue, parameterised small ratio."""

    class SieveNode:
        __slots__ = ("obj_id", "size", "accessed", "prev", "next")
        def __init__(self, oid, sz):
            self.obj_id = oid; self.size = sz; self.accessed = False
            self.prev = self.next = None

    class SieveList:
        def __init__(self):
            self._h = SieveNode(-1, 0); self._t = SieveNode(-2, 0)
            self._h.next = self._t; self._t.prev = self._h
            self._hand = self._t

        def insert(self, node):
            node.next = self._h.next; node.prev = self._h
            self._h.next.prev = node; self._h.next = node

        def remove(self, node):
            if self._hand is node:
                self._hand = node.next if node.next is not self._t else self._t
            node.prev.next = node.next; node.next.prev = node.prev
            node.prev = node.next = None

        def evict_one(self):
            if self._t.prev is self._h:
                return None  # empty list
            cur = self._hand
            if cur is self._t or cur is self._h:
                cur = self._t.prev
            # Two full sweeps maximum to guarantee termination
            start = cur
            passes = 0
            while True:
                if cur is self._h or cur is self._t:
                    cur = self._t.prev
                    if cur is self._h: return None
                    passes += 1
                    if passes > 2: break
                    continue
                if not cur.accessed:
                    v = cur
                    nxt = cur.next if cur.next is not self._t else self._t.prev
                    self._hand = nxt if nxt is not self._h else self._t.prev
                    self.remove(v)
                    return v
                cur.accessed = False
                cur = cur.prev if cur.prev is not self._h else self._t.prev
            # Fallback: evict tail
            if self._t.prev is not self._h:
                v = self._t.prev; self.remove(v); return v
            return None

    class S3FIFOSieveCache:
        def __init__(self, cache_size):
            self.cache_size = cache_size
            self.small_max  = max(1, int(cache_size * small_ratio))
            self.small: deque = deque()
            self.main = SieveList()
            self.small_map: dict = {}
            self.main_map:  dict = {}
            self.ghost_set: set = set()
            self.ghost_q:   deque = deque()
            self.ghost_max = max(1, cache_size)
            self.small_bytes = 0
            self.main_bytes  = 0

        def _add_to_ghost(self, oid):
            if oid in self.ghost_set: return
            self.ghost_q.append(oid); self.ghost_set.add(oid)
            while len(self.ghost_set) > self.ghost_max:
                self.ghost_set.discard(self.ghost_q.popleft())

        def on_hit(self, req):
            oid = req.obj_id
            if oid in self.small_map:
                sz, fr = self.small_map[oid]; self.small_map[oid] = (sz, min(fr+1, 3))
            elif oid in self.main_map:
                self.main_map[oid].accessed = True

        def on_miss(self, req):
            oid, sz = req.obj_id, req.obj_size
            if sz > self.cache_size or oid in self.small_map or oid in self.main_map: return
            if oid in self.ghost_set:
                self.ghost_set.discard(oid)
                node = SieveNode(oid, sz)
                self.main.insert(node); self.main_map[oid] = node
            else:
                self.small.append(oid); self.small_map[oid] = (sz, 0)
                self.small_bytes += sz

        def _evict_small_one(self):
            while self.small:
                oid = self.small.popleft()
                if oid not in self.small_map: continue
                sz, fr = self.small_map.pop(oid); self.small_bytes -= sz
                if fr == 0:
                    self._add_to_ghost(oid); return oid
                node = SieveNode(oid, sz)
                self.main.insert(node); self.main_map[oid] = node
                self.main_bytes += sz
            return 0

        def _evict_main_one(self):
            v = self.main.evict_one()
            if v:
                del self.main_map[v.obj_id]
                self.main_bytes -= v.size
                return v.obj_id
            return 0

        def evict(self, req):
            main_max = self.cache_size - self.small_max
            if self.main_bytes > main_max:
                v = self._evict_main_one()
                if v: return v
            v = self._evict_small_one()
            if v: return v
            return self._evict_main_one()

        def on_remove(self, oid):
            if oid in self.small_map:
                sz, _ = self.small_map.pop(oid); self.small_bytes -= sz
                try: self.small.remove(oid)
                except ValueError: pass
            elif oid in self.main_map:
                self.main.remove(self.main_map.pop(oid))
            self.ghost_set.discard(oid)

    def init_hook(p): return S3FIFOSieveCache(p.cache_size)
    def hit_hook(d, r): d.on_hit(r)
    def miss_hook(d, r): d.on_miss(r)
    def eviction_hook(d, r): return d.evict(r)
    def remove_hook(d, oid): d.on_remove(oid)
    def free_hook(d): d.small.clear(); d.small_map.clear(); d.main_map.clear(); d.ghost_set.clear()

    return init_hook, hit_hook, miss_hook, eviction_hook, remove_hook, free_hook


# ─────────────────────────────────────────────────────────────────────────────
# Workloads
# ─────────────────────────────────────────────────────────────────────────────

def gen_trace(path, n_obj, n_req, alpha, seed=42):
    if os.path.exists(path): return
    rng = np.random.default_rng(seed)
    np_tmp = np.power(np.arange(1, n_obj+1), -alpha)
    dist_map = np.cumsum(np_tmp) / np_tmp.sum()
    reqs = np.searchsorted(dist_map, rng.uniform(0, 1, n_req)) + 1
    s = struct.Struct("<IQIq")
    with open(path, "wb") as f:
        for i, obj in enumerate(reqs):
            f.write(s.pack(i, int(obj), 1, -2))

def gen_scan_trace(path, n_obj, n_req, hot_frac=0.1, seed=42):
    """Mix: 70% accesses to hot set (10% of objects), 30% sequential scan."""
    if os.path.exists(path): return
    rng = np.random.default_rng(seed)
    hot_n = max(1, int(n_obj * hot_frac))
    reqs = []
    for _ in range(n_req):
        if rng.random() < 0.7:
            reqs.append(int(rng.integers(1, hot_n + 1)))
        else:
            reqs.append(int(rng.integers(hot_n + 1, n_obj + 1)))
    s = struct.Struct("<IQIq")
    with open(path, "wb") as f:
        for i, obj in enumerate(reqs):
            f.write(s.pack(i, obj, 1, -2))

def gen_onehit_trace(path, n_obj, n_req, onehit_frac=0.8, seed=42):
    """Mix: one-hit-wonder objects (accessed once) + repeated hot set."""
    if os.path.exists(path): return
    rng = np.random.default_rng(seed)
    hot_n = max(1, int(n_obj * (1 - onehit_frac)))
    reqs = []
    for i in range(n_req):
        if rng.random() < onehit_frac:
            reqs.append(n_obj * 10 + i)   # unique id → never repeated
        else:
            reqs.append(int(rng.integers(1, hot_n + 1)))
    s = struct.Struct("<IQIq")
    with open(path, "wb") as f:
        for i, obj in enumerate(reqs):
            f.write(s.pack(i, obj % (2**32), 1, -2))

WORKLOADS = [
    {"name": "Zipf α=1.2  (high skew)",  "path": "/tmp/tune_zipf12.bin",    "gen": lambda p: gen_trace(p, 10_000, 300_000, 1.2),   "cache_sz": 500},
    {"name": "Zipf α=1.0  (std skew)",   "path": "/tmp/tune_zipf10.bin",    "gen": lambda p: gen_trace(p, 10_000, 300_000, 1.0),   "cache_sz": 500},
    {"name": "Zipf α=0.7  (low skew)",   "path": "/tmp/tune_zipf07.bin",    "gen": lambda p: gen_trace(p, 10_000, 300_000, 0.7),   "cache_sz": 500},
    {"name": "Scan+Hot    (scan-heavy)",  "path": "/tmp/tune_scan.bin",      "gen": lambda p: gen_scan_trace(p, 10_000, 300_000),   "cache_sz": 500},
    {"name": "OneHit 80%  (CDN-like)",    "path": "/tmp/tune_onehit.bin",    "gen": lambda p: gen_onehit_trace(p, 5_000, 300_000),  "cache_sz": 500},
    {"name": "cloudPhysicsIO",            "path": str(DATA_DIR/"cloudPhysicsIO.vscsi"), "gen": None, "cache_sz": 1*1024*1024,
     "trace_type": TraceType.VSCSI_TRACE},
]

# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

SMALL_RATIOS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.33, 0.40]

def run_sweep(workloads, small_ratios):
    # Prepare traces
    for wl in workloads:
        if wl.get("gen"):
            wl["gen"](wl["path"])
        if "trace_type" not in wl:
            wl["trace_type"] = TraceType.ORACLE_GENERAL_TRACE

    results_s3  = {}   # ratio → [miss_ratio per workload]
    results_hyb = {}   # ratio → [miss_ratio per workload]

    for ratio in small_ratios:
        results_s3[ratio]  = []
        results_hyb[ratio] = []
        hooks_s3  = make_s3fifo(ratio)
        hooks_hyb = make_s3fifo_sieve(ratio)
        for wl in workloads:
            reader = TraceReader(trace=wl["path"], trace_type=wl["trace_type"])
            cache = PluginCache(
                cache_size=wl["cache_sz"],
                cache_init_hook=hooks_s3[0], cache_hit_hook=hooks_s3[1],
                cache_miss_hook=hooks_s3[2], cache_eviction_hook=hooks_s3[3],
                cache_remove_hook=hooks_s3[4], cache_free_hook=hooks_s3[5],
                cache_name=f"s3fifo-{int(ratio*100)}",
            )
            mr, _ = cache.process_trace(reader)
            results_s3[ratio].append(mr)

            reader = TraceReader(trace=wl["path"], trace_type=wl["trace_type"])
            cache = PluginCache(
                cache_size=wl["cache_sz"],
                cache_init_hook=hooks_hyb[0], cache_hit_hook=hooks_hyb[1],
                cache_miss_hook=hooks_hyb[2], cache_eviction_hook=hooks_hyb[3],
                cache_remove_hook=hooks_hyb[4], cache_free_hook=hooks_hyb[5],
                cache_name=f"s3sieve-{int(ratio*100)}",
            )
            mr, _ = cache.process_trace(reader)
            results_hyb[ratio].append(mr)

    return results_s3, results_hyb


def print_results(results, algo_name, workloads, small_ratios):
    wl_names = [wl["name"] for wl in workloads]
    col = 14
    print(f"\n{'='*80}")
    print(f"{algo_name} — miss ratio by small-queue ratio (lower = better)")
    print(f"{'='*80}")
    header = f"{'Ratio':<8}" + "".join(f"{n[:col]:>{col}}" for n in wl_names) + f"{'AVG':>{col}}"
    print(header)
    print("-" * len(header))

    best_avg = float("inf")
    best_ratio = None
    for ratio in small_ratios:
        vals = results[ratio]
        avg  = sum(vals) / len(vals)
        if avg < best_avg:
            best_avg = avg; best_ratio = ratio
        marker = " ←"  # will add later
        row = f"{ratio*100:5.0f}%  " + "".join(f"{v:>{col}.4f}" for v in vals) + f"{avg:>{col}.4f}"
        print(row)

    print(f"\nBest ratio: {best_ratio*100:.0f}%  (avg miss = {best_avg:.4f})")


if __name__ == "__main__":
    print(f"Sweeping small-queue ratio {[f'{r*100:.0f}%' for r in SMALL_RATIOS]}")
    print(f"on {len(WORKLOADS)} workloads × 2 algorithms\n")

    results_s3, results_hyb = run_sweep(WORKLOADS, SMALL_RATIOS)

    print_results(results_s3,  "S3-FIFO",        WORKLOADS, SMALL_RATIOS)
    print_results(results_hyb, "S3FIFO+SIEVE",   WORKLOADS, SMALL_RATIOS)

    # Side-by-side best ratio comparison
    print(f"\n{'='*80}")
    print("COMPARISON: S3-FIFO vs S3FIFO+SIEVE — best config for each workload")
    print(f"{'='*80}")
    print(f"{'Workload':<30} {'S3-FIFO best':>16} {'ratio':>8} {'Hybrid best':>16} {'ratio':>8}")
    print("-" * 80)
    for i, wl in enumerate(WORKLOADS):
        s3_best   = min(SMALL_RATIOS, key=lambda r: results_s3[r][i])
        hyb_best  = min(SMALL_RATIOS, key=lambda r: results_hyb[r][i])
        s3_val    = results_s3[s3_best][i]
        hyb_val   = results_hyb[hyb_best][i]
        winner = "S3-FIFO" if s3_val <= hyb_val else "Hybrid"
        print(f"{wl['name'][:30]:<30} {s3_val:>16.4f} {s3_best*100:>6.0f}%  "
              f"{hyb_val:>16.4f} {hyb_best*100:>6.0f}%  ← {winner}")
