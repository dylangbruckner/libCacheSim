"""
LFU-DA (Least Frequently Used with Dynamic Aging) — Arlitt et al., 2000.

Pure LFU has a well-known "cache pollution" problem: objects accessed many
times in the past retain high frequency even after they become unpopular,
blocking new hot objects.

LFU-DA solves this by adding a global "age" counter L that starts at 0.
When an object is *first inserted* its frequency is set to L+1 instead of 1.
Whenever an object is hit, its frequency is incremented by 1.
When the victim with the minimum frequency F_min is evicted, L is bumped to
F_min — effectively erasing the evicted object's age from the counter.

This ensures that newly inserted objects start with a count competitive with
recently-active (but aging) objects, preventing stale-high-frequency objects
from monopolizing the cache.

Implementation uses a min-heap for O(log n) minimum extraction and a dict
for O(1) hit updates.
"""

import heapq
from libcachesim import CommonCacheParams, Request


class LFUDACache:
    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.L: int = 0                    # global age / inflation counter

        # obj_id → (freq, size)
        self.freq_map: dict = {}

        # Min-heap: (freq, insertion_counter, obj_id)
        # insertion_counter breaks ties deterministically (lower = older)
        self.heap: list = []
        self.counter: int = 0              # monotonically increasing tie-breaker

        # "Lazy deletion" set: entries in heap that have stale (wrong) freq
        self.stale: set = set()            # obj_ids whose heap entry is stale

    # ------------------------------------------------------------------ #

    def _push(self, obj_id: int, freq: int):
        self.counter += 1
        heapq.heappush(self.heap, (freq, self.counter, obj_id))

    def _find_min_valid(self):
        """Pop stale entries from heap; return (freq, obj_id) of true min."""
        while self.heap:
            freq, _, obj_id = self.heap[0]
            if obj_id not in self.freq_map:
                heapq.heappop(self.heap)
                continue
            actual_freq, _ = self.freq_map[obj_id]
            if actual_freq != freq:
                heapq.heappop(self.heap)
                # Re-push with correct freq
                self._push(obj_id, actual_freq)
                continue
            return freq, obj_id
        return None, None

    # ------------------------------------------------------------------ #

    def on_hit(self, req: Request):
        obj_id = req.obj_id
        if obj_id in self.freq_map:
            freq, size = self.freq_map[obj_id]
            new_freq = freq + 1
            self.freq_map[obj_id] = (new_freq, size)
            # Lazy: old heap entry will be seen as stale and re-pushed on next evict

    def on_miss(self, req: Request):
        obj_id = req.obj_id
        size   = req.obj_size
        if size > self.cache_size or obj_id in self.freq_map:
            return
        init_freq = self.L + 1
        self.freq_map[obj_id] = (init_freq, size)
        self._push(obj_id, init_freq)

    def evict(self, req: Request) -> int:
        freq_min, victim = self._find_min_valid()
        if victim is None:
            return 0
        heapq.heappop(self.heap)
        del self.freq_map[victim]
        # Advance global age to evicted object's frequency
        self.L = freq_min
        return victim

    def on_remove(self, obj_id: int):
        self.freq_map.pop(obj_id, None)
        # Lazy: heap entry will be cleaned up on next eviction call


# ---------------------------------------------------------------------------
# Hook functions
# ---------------------------------------------------------------------------

def init_hook(common_cache_params: CommonCacheParams) -> LFUDACache:
    return LFUDACache(common_cache_params.cache_size)


def hit_hook(data: LFUDACache, req: Request):
    data.on_hit(req)


def miss_hook(data: LFUDACache, req: Request):
    data.on_miss(req)


def eviction_hook(data: LFUDACache, req: Request) -> int:
    return data.evict(req)


def remove_hook(data: LFUDACache, obj_id: int):
    data.on_remove(obj_id)


def free_hook(data: LFUDACache):
    data.freq_map.clear()
    data.heap.clear()


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
        path = f"/tmp/lfuda_{label}.bin"
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
            cache_name="lfu-da",
        )
        reader = TraceReader(trace=wl["trace"], trace_type=wl["trace_type"])
        req_mr, byte_mr = cache.process_trace(reader)
        print(f"{wl['name']:<55} {req_mr:>10.4f} {byte_mr:>10.4f}")
