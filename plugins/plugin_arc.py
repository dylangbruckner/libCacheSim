"""
ARC (Adaptive Replacement Cache) — Nimrod Megiddo & Dharmendra Modha, FAST '03.

ARC self-tunes between recency (LRU) and frequency (LFU) by maintaining two
LRU lists—T1 (seen once) and T2 (seen more than once)—and two ghost lists
B1 and B2 that track recently evicted objects.  A target parameter p adapts
to observed workload patterns; hitting a B1 ghost promotes p (more space to
T1) while hitting a B2 ghost demotes p (more space to T2).

ARC is particularly effective on storage I/O workloads, where temporal
locality and frequency patterns both matter.
"""

from collections import OrderedDict
from libcachesim import CommonCacheParams, Request


class ARCCache:
    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        # Four LRU lists (MRU order: move_to_end = most recent)
        self.t1: OrderedDict = OrderedDict()   # recently seen once (obj_id -> size)
        self.t2: OrderedDict = OrderedDict()   # recently seen >=2 (obj_id -> size)
        self.b1: OrderedDict = OrderedDict()   # ghost of T1 (obj_id -> 1)
        self.b2: OrderedDict = OrderedDict()   # ghost of T2 (obj_id -> 1)

        # p: target fraction for T1 objects, range [0.0, 1.0]
        # Represents what fraction of the total cache should be T1.
        self.p: float = 0.5
        self.ghost_max: int = max(8, cache_size // 16)   # max ghost entries

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _replace(self, in_b2: bool) -> int:
        """Evict one object from T1 or T2 based on the target p."""
        t1_n = len(self.t1)
        t2_n = len(self.t2)
        total = t1_n + t2_n
        if total == 0:
            return 0
        t1_ratio = t1_n / total

        if t1_n > 0 and (t1_ratio > self.p or (in_b2 and t1_ratio == self.p)):
            # Evict LRU of T1 → B1
            obj_id, _ = next(iter(self.t1.items()))
            del self.t1[obj_id]
            self.b1[obj_id] = 1
            self._trim_ghost(self.b1)
            return obj_id
        elif self.t2:
            # Evict LRU of T2 → B2
            obj_id, _ = next(iter(self.t2.items()))
            del self.t2[obj_id]
            self.b2[obj_id] = 1
            self._trim_ghost(self.b2)
            return obj_id
        elif self.t1:
            obj_id, _ = next(iter(self.t1.items()))
            del self.t1[obj_id]
            self.b1[obj_id] = 1
            self._trim_ghost(self.b1)
            return obj_id
        return 0

    def _trim_ghost(self, ghost: OrderedDict):
        while len(ghost) > self.ghost_max:
            ghost.popitem(last=False)

    # ------------------------------------------------------------------ #
    # Hook implementations
    # ------------------------------------------------------------------ #
    def on_hit(self, req: Request):
        obj_id = req.obj_id
        if obj_id in self.t1:
            size = self.t1.pop(obj_id)
            self.t2[obj_id] = size
            self.t2.move_to_end(obj_id)
        elif obj_id in self.t2:
            self.t2.move_to_end(obj_id)

    def on_miss(self, req: Request):
        obj_id = req.obj_id
        size = req.obj_size
        if size > self.cache_size:
            return

        if obj_id in self.b1:
            # Ghost hit in B1 → increase p (favor recency)
            b1_n = len(self.b1)
            b2_n = len(self.b2)
            delta = max(1.0 / max(b1_n, 1), b2_n / max(b1_n, 1)) / (b1_n + b2_n + 1)
            self.p = min(self.p + delta, 1.0)
            del self.b1[obj_id]
            self.t2[obj_id] = size
            self.t2.move_to_end(obj_id)
        elif obj_id in self.b2:
            # Ghost hit in B2 → decrease p (favor frequency)
            b1_n = len(self.b1)
            b2_n = len(self.b2)
            delta = max(1.0 / max(b2_n, 1), b1_n / max(b2_n, 1)) / (b1_n + b2_n + 1)
            self.p = max(self.p - delta, 0.0)
            del self.b2[obj_id]
            self.t2[obj_id] = size
            self.t2.move_to_end(obj_id)
        else:
            # Brand-new object → T1
            self.t1[obj_id] = size
            self.t1.move_to_end(obj_id)

    def evict(self, req: Request) -> int:
        in_b2 = req.obj_id in self.b2
        return self._replace(in_b2)

    def on_remove(self, obj_id: int):
        self.t1.pop(obj_id, None)
        self.t2.pop(obj_id, None)
        self.b1.pop(obj_id, None)
        self.b2.pop(obj_id, None)


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
