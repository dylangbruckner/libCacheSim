"""
Extended parameter sweep for S3-FIFO variants.

Tests:
  1. S3-FIFO (FIFO main):  ghost ratio 1x–20x, small ratio 6%–20%
  2. S3-FIFO-LRU (LRU main): same parameter grid
  3. MAS3 (metadata-aware LRU): iat_factor sweep

On course traces if available (/data/course_traces/trace_*.lcs.zst),
falls back to synthetic Zipf + cloudPhysicsIO traces otherwise.

Run:  python3 tune_extended.py
"""

import struct, os, sys
import numpy as np
from pathlib import Path
from collections import deque, OrderedDict

sys.path.insert(0, str(Path(__file__).parent))

from libcachesim import CommonCacheParams, Request, PluginCache, TraceReader, TraceType

DATA_DIR = Path(__file__).parent.parent / "data"
COURSE_TRACE_DIR = Path("/data/course_traces")

# ─────────────────────────────────────────────────────────────────────────────
# Detect available traces
# ─────────────────────────────────────────────────────────────────────────────

def find_course_traces():
    """Return list of (name, path, cache_size_small, cache_size_large, trace_type)."""
    traces = []
    if COURSE_TRACE_DIR.exists():
        for i in range(10):
            p = COURSE_TRACE_DIR / f"trace_{i}.lcs.zst"
            if p.exists():
                # Cache sizes match the grader's reported values
                sizes = {
                    0: (7027,  70273),
                    1: (1241,  12414),
                    2: (3762,  37627),
                    3: (728,   7282),
                    4: (4263,  42632),
                    5: (4915,  49156),
                    6: (7555,  75551),
                    7: (1646,  16460),
                    8: (3225,  32254),
                    9: (7164,  71647),
                }
                sm, lg = sizes[i]
                traces.append((f"trace_{i}_small", str(p), sm,  TraceType.LCS_TRACE))
                traces.append((f"trace_{i}_large", str(p), lg, TraceType.LCS_TRACE))
    return traces


def gen_zipf_trace(path, n_obj, n_req, alpha, seed=42):
    if os.path.exists(path):
        return
    rng = np.random.default_rng(seed)
    np_tmp = np.power(np.arange(1, n_obj + 1), -alpha)
    dist_map = np.cumsum(np_tmp) / np_tmp.sum()
    reqs = np.searchsorted(dist_map, rng.uniform(0, 1, n_req)) + 1
    s = struct.Struct("<IQIq")
    with open(path, "wb") as f:
        for i, obj in enumerate(reqs):
            f.write(s.pack(i, int(obj), 1, -2))


def prepare_synthetic_workloads():
    wls = []
    for alpha, sz in [(1.2, 500), (1.0, 500), (0.7, 500)]:
        path = f"/tmp/tune_ext_zipf{int(alpha*10)}.bin"
        gen_zipf_trace(path, 10_000, 500_000, alpha)
        wls.append((f"Zipf α={alpha}", path, sz, TraceType.ORACLE_GENERAL_TRACE))
    vscsi = str(DATA_DIR / "cloudPhysicsIO.vscsi")
    if Path(vscsi).exists():
        wls.append(("cloudPhysicsIO", vscsi, 1 * 1024 * 1024, TraceType.VSCSI_TRACE))
    return wls


# ─────────────────────────────────────────────────────────────────────────────
# Parameterised S3-FIFO (FIFO main)
# ─────────────────────────────────────────────────────────────────────────────

def make_s3fifo(small_ratio: float, ghost_ratio: float):
    class Cache:
        def __init__(self, cs):
            self.cache_size = cs
            self.small_max = max(1, int(cs * small_ratio))
            self.ghost_max_n = int(cs * ghost_ratio)
            self.small = deque(); self.small_map = {}; self.small_bytes = 0
            self.main = deque(); self.obj_info = {}; self.main_bytes = 0
            self.ghost_set = set(); self.ghost_q = deque()

        def _ghost(self, oid):
            if oid in self.ghost_set: return
            self.ghost_set.add(oid); self.ghost_q.append(oid)
            while len(self.ghost_set) > self.ghost_max_n:
                self.ghost_set.discard(self.ghost_q.popleft())

        def on_hit(self, req):
            oid = req.obj_id
            if oid in self.obj_info:
                sz, fr = self.obj_info[oid]; self.obj_info[oid] = (sz, min(fr+1, 3))

        def on_miss(self, req):
            oid, sz = req.obj_id, req.obj_size
            if sz > self.cache_size or oid in self.obj_info: return
            if oid in self.ghost_set:
                self.ghost_set.discard(oid)
                self.main.appendleft(oid); self.obj_info[oid] = (sz, 0); self.main_bytes += sz
            else:
                self.small.append(oid); self.obj_info[oid] = (sz, 0); self.small_bytes += sz

        def _evict_small(self):
            while self.small:
                oid = self.small.popleft()
                if oid not in self.obj_info: continue
                sz, fr = self.obj_info[oid]; self.small_bytes -= sz
                if fr == 0:
                    del self.obj_info[oid]; self._ghost(oid); return oid
                self.obj_info[oid] = (sz, 0); self.main.appendleft(oid); self.main_bytes += sz
            return 0

        def _evict_main(self):
            sc, lim = 0, len(self.main)
            while self.main and sc <= lim:
                oid = self.main.pop(); sc += 1
                if oid not in self.obj_info: continue
                sz, fr = self.obj_info[oid]; self.main_bytes -= sz
                if fr == 0:
                    del self.obj_info[oid]; return oid
                self.obj_info[oid] = (sz, fr-1); self.main.appendleft(oid); self.main_bytes += sz
            return 0

        def evict(self, req):
            mm = self.cache_size - self.small_max
            if self.main_bytes > mm and self.main:
                v = self._evict_main()
                if v: return v
            v = self._evict_small()
            if v: return v
            return self._evict_main()

        def on_remove(self, oid):
            if oid not in self.obj_info: return
            sz, _ = self.obj_info.pop(oid)
            try: self.small.remove(oid); self.small_bytes -= sz
            except ValueError:
                try: self.main.remove(oid); self.main_bytes -= sz
                except ValueError: pass
            self.ghost_set.discard(oid)

    def init(p): return Cache(p.cache_size)
    def hit(d, r): d.on_hit(r)
    def miss(d, r): d.on_miss(r)
    def evict(d, r): return d.evict(r)
    def remove(d, oid): d.on_remove(oid)
    def free(d): d.small.clear(); d.main.clear(); d.obj_info.clear(); d.ghost_set.clear()
    return init, hit, miss, evict, remove, free


# ─────────────────────────────────────────────────────────────────────────────
# Parameterised S3-FIFO-LRU (LRU main)
# ─────────────────────────────────────────────────────────────────────────────

def make_s3fifo_lru(small_ratio: float, ghost_ratio: float):
    class Cache:
        def __init__(self, cs):
            self.cache_size = cs
            self.small_max = max(1, int(cs * small_ratio))
            self.ghost_max_n = int(cs * ghost_ratio)
            self.small = deque(); self.small_map = {}; self.small_bytes = 0
            self.main = OrderedDict(); self.main_bytes = 0
            self.ghost_set = set(); self.ghost_q = deque()

        def _ghost(self, oid):
            if oid in self.ghost_set: return
            self.ghost_set.add(oid); self.ghost_q.append(oid)
            while len(self.ghost_set) > self.ghost_max_n:
                self.ghost_set.discard(self.ghost_q.popleft())

        def on_hit(self, req):
            oid = req.obj_id
            if oid in self.small_map:
                sz, fr = self.small_map[oid]; self.small_map[oid] = (sz, min(fr+1,3))
            elif oid in self.main:
                self.main.move_to_end(oid)

        def on_miss(self, req):
            oid, sz = req.obj_id, req.obj_size
            if sz > self.cache_size or oid in self.small_map or oid in self.main: return
            if oid in self.ghost_set:
                self.ghost_set.discard(oid)
                self.main[oid] = sz; self.main_bytes += sz; self.main.move_to_end(oid)
            else:
                self.small.append(oid); self.small_map[oid] = (sz, 0); self.small_bytes += sz

        def _evict_small(self):
            while self.small:
                oid = self.small.popleft()
                if oid not in self.small_map: continue
                sz, fr = self.small_map.pop(oid); self.small_bytes -= sz
                if fr == 0:
                    self._ghost(oid); return oid
                self.main[oid] = sz; self.main_bytes += sz; self.main.move_to_end(oid)
            return 0

        def _evict_main(self):
            if not self.main: return 0
            oid, sz = self.main.popitem(last=False); self.main_bytes -= sz; return oid

        def evict(self, req):
            mm = self.cache_size - self.small_max
            if self.main_bytes > mm and self.main:
                v = self._evict_main()
                if v: return v
            v = self._evict_small()
            if v: return v
            return self._evict_main()

        def on_remove(self, oid):
            if oid in self.small_map:
                sz, _ = self.small_map.pop(oid); self.small_bytes -= sz
                try: self.small.remove(oid)
                except ValueError: pass
            elif oid in self.main:
                sz = self.main.pop(oid); self.main_bytes -= sz
            self.ghost_set.discard(oid)

    def init(p): return Cache(p.cache_size)
    def hit(d, r): d.on_hit(r)
    def miss(d, r): d.on_miss(r)
    def evict(d, r): return d.evict(r)
    def remove(d, oid): d.on_remove(oid)
    def free(d): d.small.clear(); d.small_map.clear(); d.main.clear(); d.ghost_set.clear()
    return init, hit, miss, evict, remove, free


# ─────────────────────────────────────────────────────────────────────────────
# Parameterised MAS3 (metadata-aware S3-FIFO-LRU)
# ─────────────────────────────────────────────────────────────────────────────

def make_mas3(small_ratio: float, ghost_ratio: float, iat_factor: float):
    IAT_ALPHA = 0.3

    class Cache:
        def __init__(self, cs):
            self.cache_size = cs
            self.small_max = max(1, int(cs * small_ratio))
            self.ghost_max_n = int(cs * ghost_ratio)
            self.small = deque(); self.small_map = {}; self.small_bytes = 0
            self.main = OrderedDict(); self.main_bytes = 0
            self.ghost_set = set(); self.ghost_q = deque()
            self.metadata = {}  # obj_id → (count, last_time, iat_ewma)
            self.time = 0
            self._total_size = 0; self._total_objs = 0

        @property
        def _est_capacity(self):
            avg = self._total_size / self._total_objs if self._total_objs else 1.0
            return self.cache_size / max(1.0, avg)

        def _update_meta(self, oid, sz):
            self.time += 1
            if oid in self.metadata:
                c, lt, ew = self.metadata[oid]
                iat = self.time - lt
                new_ew = (1-IAT_ALPHA)*ew + IAT_ALPHA*iat if ew > 0 else float(iat)
                self.metadata[oid] = (c+1, self.time, new_ew)
            else:
                self.metadata[oid] = (1, self.time, 0.0)
                self._total_objs += 1; self._total_size += sz

        def _fast_track(self, oid):
            m = self.metadata.get(oid)
            if not m: return False
            c, _, ew = m
            if c < 2 or ew <= 0: return False
            return ew < iat_factor * self._est_capacity

        def _ghost(self, oid):
            if oid in self.ghost_set: return
            self.ghost_set.add(oid); self.ghost_q.append(oid)
            while len(self.ghost_set) > self.ghost_max_n:
                self.ghost_set.discard(self.ghost_q.popleft())

        def on_hit(self, req):
            oid, sz = req.obj_id, req.obj_size
            self._update_meta(oid, sz)
            if oid in self.small_map:
                s, f = self.small_map[oid]; self.small_map[oid] = (s, min(f+1, 3))
            elif oid in self.main:
                self.main.move_to_end(oid)

        def on_miss(self, req):
            oid, sz = req.obj_id, req.obj_size
            self._update_meta(oid, sz)
            if sz > self.cache_size or oid in self.small_map or oid in self.main: return
            in_ghost = oid in self.ghost_set
            fast = (not in_ghost) and self._fast_track(oid)
            if in_ghost or fast:
                self.ghost_set.discard(oid)
                self.main[oid] = sz; self.main_bytes += sz; self.main.move_to_end(oid)
            else:
                self.small.append(oid); self.small_map[oid] = (sz, 0); self.small_bytes += sz

        def _evict_small(self):
            while self.small:
                oid = self.small.popleft()
                if oid not in self.small_map: continue
                sz, fr = self.small_map.pop(oid); self.small_bytes -= sz
                if fr == 0:
                    self._ghost(oid); return oid
                self.main[oid] = sz; self.main_bytes += sz; self.main.move_to_end(oid)
            return 0

        def _evict_main(self):
            if not self.main: return 0
            oid, sz = self.main.popitem(last=False); self.main_bytes -= sz; return oid

        def evict(self, req):
            mm = self.cache_size - self.small_max
            if self.main_bytes > mm and self.main:
                v = self._evict_main()
                if v: return v
            v = self._evict_small()
            if v: return v
            return self._evict_main()

        def on_remove(self, oid):
            if oid in self.small_map:
                sz, _ = self.small_map.pop(oid); self.small_bytes -= sz
                try: self.small.remove(oid)
                except ValueError: pass
            elif oid in self.main:
                sz = self.main.pop(oid); self.main_bytes -= sz
            self.ghost_set.discard(oid)

    def init(p): return Cache(p.cache_size)
    def hit(d, r): d.on_hit(r)
    def miss(d, r): d.on_miss(r)
    def evict(d, r): return d.evict(r)
    def remove(d, oid): d.on_remove(oid)
    def free(d): d.small.clear(); d.small_map.clear(); d.main.clear(); d.ghost_set.clear(); d.metadata.clear()
    return init, hit, miss, evict, remove, free


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_variant(hooks, cache_size, trace_path, trace_type, name):
    init, hit, miss, evict, remove, free = hooks
    cache = PluginCache(
        cache_size=cache_size,
        cache_init_hook=init, cache_hit_hook=hit,
        cache_miss_hook=miss, cache_eviction_hook=evict,
        cache_remove_hook=remove, cache_free_hook=free,
        cache_name=name,
    )
    reader = TraceReader(trace=trace_path, trace_type=trace_type)
    mr, _ = cache.process_trace(reader)
    return mr


def sweep_ghost_small(workloads, small_ratios, ghost_ratios, algo="fifo"):
    """Sweep small_ratio × ghost_ratio for the given algo ('fifo' or 'lru')."""
    make = make_s3fifo if algo == "fifo" else make_s3fifo_lru
    results = {}
    for sr in small_ratios:
        for gr in ghost_ratios:
            key = (sr, gr)
            results[key] = []
            hooks = make(sr, gr)
            for name, path, cs, tt in workloads:
                mr = run_variant(hooks, cs, path, tt, f"s3{algo}-{int(sr*100)}pct-{int(gr)}x")
                results[key].append(mr)
    return results


def sweep_mas3(workloads, small_ratios, ghost_ratios, iat_factors):
    results = {}
    for sr in small_ratios:
        for gr in ghost_ratios:
            for iatf in iat_factors:
                key = (sr, gr, iatf)
                results[key] = []
                hooks = make_mas3(sr, gr, iatf)
                for name, path, cs, tt in workloads:
                    mr = run_variant(hooks, cs, path, tt, f"mas3-{int(sr*100)}-{int(gr)}x-iat{iatf}")
                    results[key].append(mr)
    return results


def print_sweep(results, param_labels, wl_names, title):
    print(f"\n{'='*90}")
    print(f"{title} — avg miss ratio across workloads (lower = better)")
    print(f"{'='*90}")
    col = 12
    hdr = f"{'Params':<28}" + "".join(f"{n[:col]:>{col}}" for n in wl_names) + f"{'AVG':>{col}}"
    print(hdr); print("-" * len(hdr))

    rows = []
    for key, vals in results.items():
        avg = sum(vals) / len(vals) if vals else 1.0
        rows.append((avg, key, vals))
    rows.sort()

    for avg, key, vals in rows:
        label = param_labels(key)
        row = f"{label:<28}" + "".join(f"{v:>{col}.4f}" for v in vals) + f"{avg:>{col}.4f}"
        print(row)

    best_avg, best_key, _ = rows[0]
    print(f"\nBest: {param_labels(best_key)}  avg={best_avg:.4f}")
    return best_key, best_avg


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── workload setup ────────────────────────────────────────────────────
    course = find_course_traces()
    if course:
        print(f"Using {len(course)} course trace configurations.")
        workloads = course
    else:
        print("Course traces not found; using synthetic workloads.")
        workloads = prepare_synthetic_workloads()
        # Convert to (name, path, cache_size, trace_type) tuples
        workloads = [(n, p, cs, tt) for n, p, cs, tt in workloads]

    wl_names = [w[0] for w in workloads]

    # ── Phase 1: ghost × small ratio sweep (FIFO main) ───────────────────
    SMALL_RATIOS  = [0.06, 0.08, 0.10, 0.11, 0.12, 0.15]
    GHOST_RATIOS  = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0]

    print(f"\nPhase 1: S3-FIFO (FIFO main) — {len(SMALL_RATIOS)} small × {len(GHOST_RATIOS)} ghost ratios")
    res_fifo = sweep_ghost_small(workloads, SMALL_RATIOS, GHOST_RATIOS, algo="fifo")
    best_fifo, _ = print_sweep(
        res_fifo,
        lambda k: f"small={int(k[0]*100)}% ghost={int(k[1])}x",
        wl_names, "S3-FIFO (FIFO main)"
    )

    # ── Phase 2: ghost × small ratio sweep (LRU main) ────────────────────
    print(f"\nPhase 2: S3-FIFO-LRU (LRU main) — same grid")
    res_lru = sweep_ghost_small(workloads, SMALL_RATIOS, GHOST_RATIOS, algo="lru")
    best_lru, _ = print_sweep(
        res_lru,
        lambda k: f"small={int(k[0]*100)}% ghost={int(k[1])}x",
        wl_names, "S3-FIFO-LRU (LRU main)"
    )

    # ── Phase 3: MAS3 IAT-factor sweep (around best params) ──────────────
    best_sr_fifo, best_gr_fifo = best_fifo
    best_sr_lru,  best_gr_lru  = best_lru
    # Use the LRU best as a starting point for MAS3
    MAS3_SMALL  = sorted(set([best_sr_lru, 0.10, 0.11]))
    MAS3_GHOST  = sorted(set([best_gr_lru, 4.0, 6.0]))
    MAS3_IAT    = [0.5, 1.0, 2.0, 4.0, 8.0, float('inf')]

    print(f"\nPhase 3: MAS3 — IAT factor sweep (small={MAS3_SMALL}, ghost={MAS3_GHOST})")
    res_mas3 = sweep_mas3(workloads, MAS3_SMALL, MAS3_GHOST, MAS3_IAT)
    best_mas3, _ = print_sweep(
        res_mas3,
        lambda k: f"small={int(k[0]*100)}% ghost={int(k[1])}x iat={k[2]}",
        wl_names, "MAS3"
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    sr_f, gr_f = best_fifo
    sr_l, gr_l = best_lru
    sr_m, gr_m, iatf_m = best_mas3
    print(f"  Best S3-FIFO (FIFO main): small={int(sr_f*100)}%  ghost={int(gr_f)}x")
    print(f"  Best S3-FIFO-LRU:         small={int(sr_l*100)}%  ghost={int(gr_l)}x")
    print(f"  Best MAS3:                small={int(sr_m*100)}%  ghost={int(gr_m)}x  iat_factor={iatf_m}")
    print()
    print("Update the relevant plugin file with these parameters and submit!")
