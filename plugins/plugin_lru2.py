"""
LRU-2 (LRU-K with K=2) — O'Neil, O'Neil & Weikum, SIGMOD '93.

LRU-K evicts the object whose K-th most recent access is oldest.  With K=2,
each object tracks its two most recent access times; the object with the
oldest *second* access time is the eviction victim.

Objects accessed only once ("one-hit wonders") have no second access time
and are treated as infinitely old — they are evicted before any two-time
object, in FIFO order among themselves.

Why K=2 is special: K=1 is plain LRU.  K=2 gives significantly better
performance on many workloads by protecting recently-used objects from a
single accidental re-access without long-term frequency bias.
"""

from collections import deque
from libcachesim import CommonCacheParams, Request


class LRU2Cache:
    """
    Two-tier structure:
    - correlated_ref (one-access objects): simple FIFO — evicted first.
    - hist_list (two-or-more-access objects): keyed by their 2nd-access time,
      oldest 2nd-access time is the victim.  Implemented as an OrderedDict
      ordered by insertion time (= promotion time = 2nd access time).
    """

    def __init__(self, cache_size: int):
        self.cache_size = cache_size

        # Objects seen once: FIFO queue (obj_id) + metadata dict
        self.once_q:    deque = deque()
        self.once_map:  dict  = {}   # obj_id → size

        # Objects seen ≥2 times: ordered by 2nd-access time (oldest first)
        from collections import OrderedDict
        self.hist:      "OrderedDict" = __import__("collections").OrderedDict()  # obj_id → size
        self.hist_map:  dict = {}    # obj_id → size (redundant but fast check)

        self.once_bytes: int = 0
        self.hist_bytes: int = 0

    # ------------------------------------------------------------------ #

    def on_hit(self, req: Request):
        obj_id = req.obj_id
        if obj_id in self.once_map:
            # First repeat access → promote from once to hist
            size = self.once_map.pop(obj_id)
            self.once_bytes -= size
            try:
                self.once_q.remove(obj_id)
            except ValueError:
                pass
            self.hist[obj_id] = size
            self.hist.move_to_end(obj_id)  # 2nd access time = now
            self.hist_map[obj_id] = size
            self.hist_bytes += size
        elif obj_id in self.hist_map:
            # Subsequent accesses → refresh 2nd-access time
            self.hist.move_to_end(obj_id)

    def on_miss(self, req: Request):
        obj_id = req.obj_id
        size   = req.obj_size
        if size > self.cache_size or obj_id in self.once_map or obj_id in self.hist_map:
            return
        self.once_q.append(obj_id)
        self.once_map[obj_id] = size
        self.once_bytes += size

    def evict(self, req: Request) -> int:
        # Prefer evicting from once_q (no 2nd access) before hist
        if self.once_q:
            obj_id = self.once_q.popleft()
            size = self.once_map.pop(obj_id, 0)
            self.once_bytes -= size
            return obj_id

        if self.hist:
            obj_id, size = self.hist.popitem(last=False)   # oldest 2nd-access time
            self.hist_map.pop(obj_id, None)
            self.hist_bytes -= size
            return obj_id

        return 0

    def on_remove(self, obj_id: int):
        if obj_id in self.once_map:
            self.once_bytes -= self.once_map.pop(obj_id)
            try:
                self.once_q.remove(obj_id)
            except ValueError:
                pass
        elif obj_id in self.hist_map:
            self.hist_bytes -= self.hist_map.pop(obj_id)
            self.hist.pop(obj_id, None)


# ---------------------------------------------------------------------------
# Hook functions
# ---------------------------------------------------------------------------

def init_hook(common_cache_params: CommonCacheParams) -> LRU2Cache:
    return LRU2Cache(common_cache_params.cache_size)


def hit_hook(data: LRU2Cache, req: Request):
    data.on_hit(req)


def miss_hook(data: LRU2Cache, req: Request):
    data.on_miss(req)


def eviction_hook(data: LRU2Cache, req: Request) -> int:
    return data.evict(req)


def remove_hook(data: LRU2Cache, obj_id: int):
    data.on_remove(obj_id)


def free_hook(data: LRU2Cache):
    data.once_q.clear()
    data.once_map.clear()
    data.hist.clear()
    data.hist_map.clear()


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import struct, os
    import numpy as np
    from pathlib import Path
    from libcachesim import PluginCache, TraceReader, TraceType

    DATA_DIR = Path(__file__).parent.parent / "data"

    def gen_zipf_trace(path, n_obj, n_req, alpha):
        np_tmp = np.power(np.arange(1, n_obj + 1), -alpha)
        dist_map = np.cumsum(np_tmp) / np.cumsum(np_tmp)[-1]
        r = np.random.uniform(0, 1, n_req)
        reqs = np.searchsorted(dist_map, r) + 1
        s = struct.Struct("<IQIq")
        with open(path, "wb") as f:
            for i, obj in enumerate(reqs):
                f.write(s.pack(i, int(obj), 1, -2))

    workloads = [
        {
            "name": "cloudPhysicsIO (1 MB cache)",
            "trace": str(DATA_DIR / "cloudPhysicsIO.vscsi"),
            "trace_type": TraceType.VSCSI_TRACE,
            "cache_size": 1 * 1024 * 1024,
        },
    ]

    for label, alpha, cache_sz in [("zipf_1.0", 1.0, 500), ("zipf_0.7", 0.7, 500), ("zipf_1.2", 1.2, 500)]:
        path = f"/tmp/lru2_{label}.bin"
        if not os.path.exists(path):
            gen_zipf_trace(path, 10_000, 500_000, alpha)
        workloads.append({
            "name": f"{label} ({cache_sz}/10k objs = {cache_sz/10000:.1%} cache ratio)",
            "trace": path,
            "trace_type": TraceType.ORACLE_GENERAL_TRACE,
            "cache_size": cache_sz,
        })

    print(f"{'Workload':<55} {'Req Miss':>10} {'Byte Miss':>10}")
    print("-" * 77)
    for wl in workloads:
        cache = PluginCache(
            cache_size=wl["cache_size"],
            cache_init_hook=init_hook,
            cache_hit_hook=hit_hook,
            cache_miss_hook=miss_hook,
            cache_eviction_hook=eviction_hook,
            cache_remove_hook=remove_hook,
            cache_free_hook=free_hook,
            cache_name="lru-2",
        )
        reader = TraceReader(trace=wl["trace"], trace_type=wl["trace_type"])
        req_mr, byte_mr = cache.process_trace(reader)
        print(f"{wl['name']:<55} {req_mr:>10.4f} {byte_mr:>10.4f}")
