"""
CLOCK cache eviction algorithm.

CLOCK approximates LRU using a circular list with a reference (accessed) bit
per object. A "hand" sweeps the list; objects with ref=1 get their bit cleared
(given a second chance), objects with ref=0 are evicted. This is the classic
algorithm behind many OS page-replacement policies and hardware TLBs.
"""

from collections import deque
from libcachesim import CommonCacheParams, Request


class CLOCKCache:
    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        # Clock ring: deque of obj_ids in insertion order
        self.ring: deque = deque()
        # ref_bits[obj_id] = 0 or 1
        self.ref_bits: dict = {}

    def on_hit(self, req: Request):
        if req.obj_id in self.ref_bits:
            self.ref_bits[req.obj_id] = 1

    def on_miss(self, req: Request):
        if req.obj_size <= self.cache_size:
            self.ring.append(req.obj_id)
            self.ref_bits[req.obj_id] = 0

    def evict(self, req: Request) -> int:
        # Sweep the ring until we find a ref=0 object
        attempts = 0
        limit = len(self.ring) * 2  # safety cap to avoid infinite loop
        while self.ring and attempts < limit:
            obj_id = self.ring[0]
            if obj_id not in self.ref_bits:
                # Already removed externally
                self.ring.popleft()
                attempts += 1
                continue
            if self.ref_bits[obj_id] == 0:
                self.ring.popleft()
                del self.ref_bits[obj_id]
                return obj_id
            else:
                # Give a second chance: clear bit and rotate
                self.ref_bits[obj_id] = 0
                self.ring.rotate(-1)
                attempts += 1
        # Fallback: evict whatever is at the front
        if self.ring:
            obj_id = self.ring.popleft()
            self.ref_bits.pop(obj_id, None)
            return obj_id
        return 0

    def on_remove(self, obj_id: int):
        # Remove from ref_bits; the ring entry will be lazily dropped in evict()
        self.ref_bits.pop(obj_id, None)


def init_hook(common_cache_params: CommonCacheParams) -> CLOCKCache:
    return CLOCKCache(common_cache_params.cache_size)


def hit_hook(data: CLOCKCache, req: Request):
    data.on_hit(req)


def miss_hook(data: CLOCKCache, req: Request):
    data.on_miss(req)


def eviction_hook(data: CLOCKCache, req: Request) -> int:
    return data.evict(req)


def remove_hook(data: CLOCKCache, obj_id: int):
    data.on_remove(obj_id)


def free_hook(data: CLOCKCache):
    data.ring.clear()
    data.ref_bits.clear()


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
        path = f"/tmp/clock_{label}.bin"
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
            cache_name="clock",
        )
        reader = TraceReader(trace=wl["trace"], trace_type=wl["trace_type"])
        req_mr, byte_mr = cache.process_trace(reader)
        print(f"{wl['name']:<55} {req_mr:>10.4f} {byte_mr:>10.4f}")
