"""
ARC (Adaptive Replacement Cache) — Nimrod Megiddo & Dharmendra Modha, FAST '03.

Faithful implementation of the original paper's Algorithm 1, with byte-aware
ghost-list trimming for variable-size objects.

Four lists:
  T1 — recently seen exactly once (LRU; MRU = most recent)
  T2 — recently seen more than once (LRU; MRU = most recent)
  B1 — ghost directory for recently evicted T1 objects (stores obj size)
  B2 — ghost directory for recently evicted T2 objects (stores obj size)

Adaptive parameter p (object count): target number of objects to keep in T1.
  B1 ghost hit → p += max(1, |B2| / |B1|)   (favour recency / T1)
  B2 ghost hit → p -= max(1, |B1| / |B2|)   (favour frequency / T2)
  p is capped to [0, len(T1)+len(T2)] at all times.

REPLACE(in_b2):
  Evict LRU(T1) → B1  if  len(T1) > p
                           or (len(T1)==p and inserting a B2-ghost object).
  Evict LRU(T2) → B2  otherwise.

Ghost lists are trimmed by byte capacity (≤ cache_size bytes each) so they
cannot grow unboundedly on variable-size workloads while still providing a
wide observation window proportional to the cache.
"""

from collections import OrderedDict
from libcachesim import CommonCacheParams, Request


class ARCCache:
    def __init__(self, cache_size: int):
        self.c: int = cache_size          # total cache capacity (bytes)

        # Live cache: obj_id → size (bytes), ordered LRU → MRU
        self.t1: OrderedDict = OrderedDict()
        self.t2: OrderedDict = OrderedDict()

        # Ghost directories: obj_id → size (bytes)
        self.b1: OrderedDict = OrderedDict()
        self.b2: OrderedDict = OrderedDict()

        # Byte counters (for ghost trimming and bookkeeping)
        self.t1b: int = 0
        self.t2b: int = 0
        self.b1b: int = 0
        self.b2b: int = 0

        # p: target NUMBER OF OBJECTS in T1 (paper uses object counts).
        # Starts at 0 (all frequency); adapts toward recency via ghost hits.
        self.p: int = 0

    # ------------------------------------------------------------------ #
    # REPLACE
    # ------------------------------------------------------------------ #

    def _replace(self, in_b2: bool) -> int:
        t1n = len(self.t1)
        # Evict from T1 when it exceeds target p, or when tied and inserting B2 ghost
        if self.t1 and (t1n > self.p or (in_b2 and t1n == self.p)):
            obj_id, sz = self.t1.popitem(last=False)
            self.t1b -= sz
            self.b1[obj_id] = sz
            self.b1b += sz
            self._trim_b1()
            return obj_id

        if self.t2:
            obj_id, sz = self.t2.popitem(last=False)
            self.t2b -= sz
            self.b2[obj_id] = sz
            self.b2b += sz
            self._trim_b2()
            return obj_id

        # Fallback (shouldn't happen in steady state)
        if self.t1:
            obj_id, sz = self.t1.popitem(last=False)
            self.t1b -= sz
            self.b1[obj_id] = sz
            self.b1b += sz
            self._trim_b1()
            return obj_id

        return 0

    def _trim_b1(self):
        while self.b1b > self.c and self.b1:
            _, sz = self.b1.popitem(last=False)
            self.b1b -= sz

    def _trim_b2(self):
        while self.b2b > self.c and self.b2:
            _, sz = self.b2.popitem(last=False)
            self.b2b -= sz

    # ------------------------------------------------------------------ #
    # Hook implementations
    # ------------------------------------------------------------------ #

    def on_hit(self, req: Request):
        obj_id = req.obj_id
        if obj_id in self.t1:
            # Promote from T1 → T2 (seen ≥ 2 times now)
            sz = self.t1.pop(obj_id)
            self.t1b -= sz
            self.t2[obj_id] = sz
            self.t2.move_to_end(obj_id)
            self.t2b += sz
        elif obj_id in self.t2:
            # Refresh LRU position in T2
            self.t2.move_to_end(obj_id)

    def on_miss(self, req: Request):
        obj_id = req.obj_id
        sz     = req.obj_size
        if sz > self.c:
            return

        live_n = len(self.t1) + len(self.t2)
        b1n    = len(self.b1)
        b2n    = len(self.b2)

        if obj_id in self.b1:
            # B1 ghost hit: workload trending toward recency → increase p
            delta = max(1, b2n // max(b1n, 1))
            self.p = min(self.p + delta, live_n)
            ghost_sz = self.b1.pop(obj_id)
            self.b1b -= ghost_sz
            # Returning objects go to T2 (seen at least twice)
            self.t2[obj_id] = sz
            self.t2.move_to_end(obj_id)
            self.t2b += sz

        elif obj_id in self.b2:
            # B2 ghost hit: workload trending toward frequency → decrease p
            delta = max(1, b1n // max(b2n, 1))
            self.p = max(self.p - delta, 0)
            ghost_sz = self.b2.pop(obj_id)
            self.b2b -= ghost_sz
            # Returning objects go to T2
            self.t2[obj_id] = sz
            self.t2.move_to_end(obj_id)
            self.t2b += sz

        else:
            # Brand-new object → T1 (seen for the first time)
            self.t1[obj_id] = sz
            self.t1.move_to_end(obj_id)
            self.t1b += sz

    def evict(self, req: Request) -> int:
        in_b2 = req.obj_id in self.b2
        return self._replace(in_b2)

    def on_remove(self, obj_id: int):
        if obj_id in self.t1:
            self.t1b -= self.t1.pop(obj_id)
        elif obj_id in self.t2:
            self.t2b -= self.t2.pop(obj_id)
        if obj_id in self.b1:
            self.b1b -= self.b1.pop(obj_id)
        if obj_id in self.b2:
            self.b2b -= self.b2.pop(obj_id)


# ---------------------------------------------------------------------------
# Hook functions
# ---------------------------------------------------------------------------

def init_hook(common_cache_params: CommonCacheParams) -> ARCCache:
    return ARCCache(common_cache_params.cache_size)


def hit_hook(data: ARCCache, req: Request):
    data.on_hit(req)


def miss_hook(data: ARCCache, req: Request):
    data.on_miss(req)


def eviction_hook(data: ARCCache, req: Request) -> int:
    return data.evict(req)


def remove_hook(data: ARCCache, obj_id: int):
    data.on_remove(obj_id)


def free_hook(data: ARCCache):
    data.t1.clear()
    data.t2.clear()
    data.b1.clear()
    data.b2.clear()


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
        path = f"/tmp/arc_{label}.bin"
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
            cache_name="arc",
        )
        reader = TraceReader(trace=wl["trace"], trace_type=wl["trace_type"])
        req_mr, byte_mr = cache.process_trace(reader)
        print(f"{wl['name']:<55} {req_mr:>10.4f} {byte_mr:>10.4f}")
