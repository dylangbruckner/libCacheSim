"""
SLRU (Segmented LRU) — Karedla et al., "Caching Strategies to Improve Disk
System Performance", IEEE Computer 1994.  Also the eviction backbone of the
Caffeine and Guava Java caches.

The cache is split into two LRU segments:
  • Probationary (25% of capacity): objects accessed only once live here.
    Eviction from a full cache always targets the LRU end of this segment.
  • Protected     (75% of capacity): objects that were hit while in the
    probationary segment are promoted here.  Objects evicted from protected
    are demoted back to the MRU end of probationary (not discarded outright).

Compared to plain LRU, SLRU filters out one-hit wonders more aggressively
and gives long-lived "hot" objects a protected buffer that is harder to scan
out.  It is the direct ancestor of W-TinyLFU's main cache.
"""

from collections import OrderedDict
from libcachesim import CommonCacheParams, Request


class SLRUCache:
    def __init__(self, cache_size: int):
        self.cache_size   = cache_size
        self.prob_max     = max(1, cache_size // 4)       # 25% probationary
        self.prot_max     = cache_size - self.prob_max    # 75% protected

        # Both segments: obj_id → size, MRU end = most recently used (move_to_end)
        self.prob: OrderedDict = OrderedDict()
        self.prot: OrderedDict = OrderedDict()

        self.prob_bytes: int = 0
        self.prot_bytes: int = 0

    # ------------------------------------------------------------------ #

    def _promote(self, obj_id: int, size: int):
        """Move object from probationary to protected."""
        self.prob_bytes -= self.prob.pop(obj_id, 0)
        self.prot[obj_id] = size
        self.prot.move_to_end(obj_id)
        self.prot_bytes += size
        # If protected overflows, demote the LRU of protected to probationary
        while self.prot_bytes > self.prot_max and self.prot:
            demoted_id, demoted_sz = self.prot.popitem(last=False)
            self.prot_bytes -= demoted_sz
            self.prob[demoted_id] = demoted_sz
            self.prob.move_to_end(demoted_id)  # MRU of probationary
            self.prob_bytes += demoted_sz

    def on_hit(self, req: Request):
        obj_id = req.obj_id
        if obj_id in self.prob:
            size = self.prob[obj_id]
            self._promote(obj_id, size)
        elif obj_id in self.prot:
            self.prot.move_to_end(obj_id)

    def on_miss(self, req: Request):
        obj_id = req.obj_id
        size   = req.obj_size
        if size > self.cache_size or obj_id in self.prob or obj_id in self.prot:
            return
        # New objects enter probationary
        self.prob[obj_id] = size
        self.prob.move_to_end(obj_id)
        self.prob_bytes += size

    def evict(self, req: Request) -> int:
        # Evict LRU of probationary
        if self.prob:
            obj_id, size = self.prob.popitem(last=False)
            self.prob_bytes -= size
            return obj_id
        # If probationary is empty, evict from protected
        if self.prot:
            obj_id, size = self.prot.popitem(last=False)
            self.prot_bytes -= size
            return obj_id
        return 0

    def on_remove(self, obj_id: int):
        if obj_id in self.prob:
            self.prob_bytes -= self.prob.pop(obj_id)
        elif obj_id in self.prot:
            self.prot_bytes -= self.prot.pop(obj_id)


# ---------------------------------------------------------------------------
# Hook functions
# ---------------------------------------------------------------------------

def init_hook(common_cache_params: CommonCacheParams) -> SLRUCache:
    return SLRUCache(common_cache_params.cache_size)


def hit_hook(data: SLRUCache, req: Request):
    data.on_hit(req)


def miss_hook(data: SLRUCache, req: Request):
    data.on_miss(req)


def eviction_hook(data: SLRUCache, req: Request) -> int:
    return data.evict(req)


def remove_hook(data: SLRUCache, obj_id: int):
    data.on_remove(obj_id)


def free_hook(data: SLRUCache):
    data.prob.clear()
    data.prot.clear()


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
        path = f"/tmp/slru_{label}.bin"
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
            cache_name="slru",
        )
        reader = TraceReader(trace=wl["trace"], trace_type=wl["trace_type"])
        req_mr, byte_mr = cache.process_trace(reader)
        print(f"{wl['name']:<55} {req_mr:>10.4f} {byte_mr:>10.4f}")
