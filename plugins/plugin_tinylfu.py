"""
W-TinyLFU (Window Tiny Least Frequently Used) — Ben Manes et al., TOS 2015.
Simplified implementation suitable for the plugin framework.

W-TinyLFU combines:
  1. A small Window LRU (1% of capacity) to absorb recent bursts.
  2. A main cache with an SLRU structure:
       – Protected segment  (~80% of main ≈ 79% total)
       – Probationary segment (~20% of main ≈ 20% total)
  3. A Count-Min Sketch (CMS) for approximate frequency estimation.

Admission policy (TinyLFU gate):
  When a Window cache object is evicted into the main cache, it competes
  with the victim of the main cache's probationary LRU:
    • If freq(window_candidate) ≥ freq(main_victim) → admit candidate,
      evict main_victim.
    • Otherwise → evict the window candidate instead.

This elegantly handles both temporal locality (the window LRU) and frequency
locality (the CMS filter) without the overhead of exact frequency tracking.
"""

from collections import OrderedDict
import math
from libcachesim import CommonCacheParams, Request


# ---------------------------------------------------------------------------
# Count-Min Sketch — approximate frequency counter
# ---------------------------------------------------------------------------

class CountMinSketch:
    """
    Width × depth CMS with reset-on-halving (aging) to decay stale counts.
    """

    def __init__(self, capacity: int, depth: int = 4):
        width = max(8, 1 << math.ceil(math.log2(capacity * 10)))
        self.width  = width
        self.depth  = depth
        self.table  = [[0] * width for _ in range(depth)]
        self.seeds  = [i * 2654435761 & 0xFFFFFFFF for i in range(1, depth + 1)]
        self.count  = 0
        self.reset_threshold = capacity * 10  # reset every N insertions

    def _hash(self, key: int, seed: int) -> int:
        h = key ^ seed
        h = ((h >> 16) ^ h) * 0x45D9F3B
        h = ((h >> 16) ^ h) * 0x45D9F3B
        h = (h >> 16) ^ h
        return h % self.width

    def add(self, key: int):
        self.count += 1
        for d in range(self.depth):
            idx = self._hash(key, self.seeds[d])
            if self.table[d][idx] < 15:  # cap at 15 (4-bit saturation)
                self.table[d][idx] += 1
        if self.count >= self.reset_threshold:
            self._reset()

    def estimate(self, key: int) -> int:
        return min(self.table[d][self._hash(key, self.seeds[d])]
                   for d in range(self.depth))

    def _reset(self):
        """Halve all counters (aging)."""
        for d in range(self.depth):
            self.table[d] = [v >> 1 for v in self.table[d]]
        self.count >>= 1


# ---------------------------------------------------------------------------
# W-TinyLFU Cache
# ---------------------------------------------------------------------------

class WTinyLFUCache:
    def __init__(self, cache_size: int):
        self.cache_size  = cache_size
        self.win_max     = max(1, cache_size // 100)      # 1%  window
        self.prob_max    = max(1, (cache_size - self.win_max) // 5)  # 20% of main
        self.prot_max    = cache_size - self.win_max - self.prob_max  # 79% of main

        # Window LRU
        self.window: OrderedDict = OrderedDict()   # obj_id → size
        self.win_bytes: int = 0

        # Main SLRU: probationary + protected
        self.prob: OrderedDict = OrderedDict()
        self.prot: OrderedDict = OrderedDict()
        self.prob_bytes: int = 0
        self.prot_bytes: int = 0

        # Approx frequency estimator — cap width at ~100k to control memory
        n_capacity = min(max(8, cache_size), 100_000)
        self.sketch = CountMinSketch(n_capacity)

    # ------------------------------------------------------------------ #

    def _prot_insert(self, obj_id: int, size: int):
        self.prot[obj_id] = size
        self.prot.move_to_end(obj_id)
        self.prot_bytes += size
        # Overflow protection → demote LRU of prot to prob
        while self.prot_bytes > self.prot_max and self.prot:
            evicted_id, evicted_sz = self.prot.popitem(last=False)
            self.prot_bytes -= evicted_sz
            self.prob[evicted_id] = evicted_sz
            self.prob.move_to_end(evicted_id)
            self.prob_bytes += evicted_sz

    def on_hit(self, req: Request):
        obj_id = req.obj_id
        self.sketch.add(obj_id)
        if obj_id in self.window:
            self.window.move_to_end(obj_id)
        elif obj_id in self.prob:
            size = self.prob.pop(obj_id)
            self.prob_bytes -= size
            self._prot_insert(obj_id, size)
        elif obj_id in self.prot:
            self.prot.move_to_end(obj_id)

    def on_miss(self, req: Request):
        obj_id = req.obj_id
        size   = req.obj_size
        if size > self.cache_size:
            return
        self.sketch.add(obj_id)
        # Insert into window
        self.window[obj_id] = size
        self.window.move_to_end(obj_id)
        self.win_bytes += size

    def evict(self, req: Request) -> int:
        # If window overflows → candidate enters TinyLFU gate vs main victim
        if self.win_bytes > self.win_max and self.window:
            cand_id, cand_sz = self.window.popitem(last=False)
            self.win_bytes -= cand_sz

            if self.prob or self.prot:
                # Main cache victim = LRU of probationary (or protected if empty)
                if self.prob:
                    victim_id, victim_sz = next(iter(self.prob.items()))
                else:
                    victim_id, victim_sz = next(iter(self.prot.items()))

                # Admission gate
                if self.sketch.estimate(cand_id) >= self.sketch.estimate(victim_id):
                    # Admit candidate into main, evict victim
                    if victim_id in self.prob:
                        self.prob.pop(victim_id)
                        self.prob_bytes -= victim_sz
                    else:
                        self.prot.pop(victim_id)
                        self.prot_bytes -= victim_sz
                    # Insert candidate to probationary
                    self.prob[cand_id] = cand_sz
                    self.prob.move_to_end(cand_id)
                    self.prob_bytes += cand_sz
                    return victim_id
                else:
                    # Reject candidate (evict it from window, not from main)
                    return cand_id
            else:
                # Main cache empty — just evict window candidate
                return cand_id

        # Window not overflowing — evict from probationary or protected
        if self.prob:
            obj_id, size = self.prob.popitem(last=False)
            self.prob_bytes -= size
            return obj_id
        if self.prot:
            obj_id, size = self.prot.popitem(last=False)
            self.prot_bytes -= size
            return obj_id
        if self.window:
            obj_id, size = self.window.popitem(last=False)
            self.win_bytes -= size
            return obj_id
        return 0

    def on_remove(self, obj_id: int):
        if obj_id in self.window:
            self.win_bytes -= self.window.pop(obj_id)
        elif obj_id in self.prob:
            self.prob_bytes -= self.prob.pop(obj_id)
        elif obj_id in self.prot:
            self.prot_bytes -= self.prot.pop(obj_id)


# ---------------------------------------------------------------------------
# Hook functions
# ---------------------------------------------------------------------------

def init_hook(common_cache_params: CommonCacheParams) -> WTinyLFUCache:
    return WTinyLFUCache(common_cache_params.cache_size)


def hit_hook(data: WTinyLFUCache, req: Request):
    data.on_hit(req)


def miss_hook(data: WTinyLFUCache, req: Request):
    data.on_miss(req)


def eviction_hook(data: WTinyLFUCache, req: Request) -> int:
    return data.evict(req)


def remove_hook(data: WTinyLFUCache, obj_id: int):
    data.on_remove(obj_id)


def free_hook(data: WTinyLFUCache):
    data.window.clear()
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
        path = f"/tmp/tinylfu_{label}.bin"
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
            cache_name="w-tinylfu",
        )
        reader = TraceReader(trace=wl["trace"], trace_type=wl["trace_type"])
        req_mr, byte_mr = cache.process_trace(reader)
        print(f"{wl['name']:<55} {req_mr:>10.4f} {byte_mr:>10.4f}")
