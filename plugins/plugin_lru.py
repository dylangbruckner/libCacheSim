"""
LRU (Least Recently Used) cache eviction algorithm.

Classic O(1) LRU using Python's OrderedDict. On every hit, the object is moved
to the most-recently-used end. On eviction, the least-recently-used object is
removed. The simple, strong baseline that many algorithms are compared against.
"""

from collections import OrderedDict
from libcachesim import CommonCacheParams, Request


class LRUCache:
    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.cache: OrderedDict = OrderedDict()  # obj_id -> obj_size

    def on_hit(self, req: Request):
        # Move to MRU end
        self.cache.move_to_end(req.obj_id)

    def on_miss(self, req: Request):
        if req.obj_size <= self.cache_size:
            self.cache[req.obj_id] = req.obj_size

    def evict(self, req: Request) -> int:
        if not self.cache:
            return 0
        # Evict LRU end (first item = oldest)
        obj_id, _ = self.cache.popitem(last=False)
        return obj_id

    def on_remove(self, obj_id: int):
        self.cache.pop(obj_id, None)


def init_hook(common_cache_params: CommonCacheParams) -> LRUCache:
    return LRUCache(common_cache_params.cache_size)


def hit_hook(data: LRUCache, req: Request):
    data.on_hit(req)


def miss_hook(data: LRUCache, req: Request):
    data.on_miss(req)


def eviction_hook(data: LRUCache, req: Request) -> int:
    return data.evict(req)


def remove_hook(data: LRUCache, obj_id: int):
    data.on_remove(obj_id)


def free_hook(data: LRUCache):
    data.cache.clear()


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def _make_plugin(name="lru"):
    from libcachesim import PluginCache
    return lambda size: PluginCache(
        cache_size=size,
        cache_init_hook=init_hook,
        cache_hit_hook=hit_hook,
        cache_miss_hook=miss_hook,
        cache_eviction_hook=eviction_hook,
        cache_remove_hook=remove_hook,
        cache_free_hook=free_hook,
        cache_name=name,
    )


if __name__ == "__main__":
    import struct, os, sys
    import numpy as np
    from pathlib import Path
    from libcachesim import PluginCache, TraceReader, TraceType

    DATA_DIR = Path(__file__).parent.parent / "data"

    def gen_zipf_trace(path, n_obj, n_req, alpha, obj_size=1):
        """Write an oracleGeneral binary trace with Zipf-distributed requests."""
        np_tmp = np.power(np.arange(1, n_obj + 1), -alpha)
        dist_map = np.cumsum(np_tmp) / np.cumsum(np_tmp)[-1]
        r = np.random.uniform(0, 1, n_req)
        reqs = np.searchsorted(dist_map, r) + 1
        s = struct.Struct("<IQIq")
        with open(path, "wb") as f:
            for i, obj in enumerate(reqs):
                f.write(s.pack(i, int(obj), obj_size, -2))

    workloads = [
        {
            "name": "cloudPhysicsIO (1 MB cache)",
            "trace": str(DATA_DIR / "cloudPhysicsIO.vscsi"),
            "trace_type": TraceType.VSCSI_TRACE,
            "cache_size": 1 * 1024 * 1024,
        },
    ]

    # Generate synthetic workloads
    synthetic = [
        ("zipf_1.0", 10_000, 500_000, 1.0, 500),
        ("zipf_0.7", 10_000, 500_000, 0.7, 500),
        ("zipf_1.2", 10_000, 500_000, 1.2, 500),
    ]
    for label, n_obj, n_req, alpha, cache_sz in synthetic:
        path = f"/tmp/lru_{label}.bin"
        if not os.path.exists(path):
            gen_zipf_trace(path, n_obj, n_req, alpha)
        workloads.append({
            "name": f"{label} ({cache_sz}/{n_obj} = {cache_sz/n_obj:.1%} cache ratio)",
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
            cache_name="lru",
        )
        reader = TraceReader(trace=wl["trace"], trace_type=wl["trace_type"])
        req_mr, byte_mr = cache.process_trace(reader)
        print(f"{wl['name']:<55} {req_mr:>10.4f} {byte_mr:>10.4f}")
