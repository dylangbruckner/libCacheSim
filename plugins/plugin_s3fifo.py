"""
S3-FIFO (Small, Medium/Main, Ghost FIFO) — Yang et al., SOSP '23.

S3-FIFO achieves low miss ratios and excellent throughput by combining three
simple FIFO queues:

  • Small (S): 10% of cache.  New objects enter here.
  • Main  (M): 90% of cache.  Objects with freq≥1 when evicted from S are
               promoted here; the object's freq is reset to 0.
  • Ghost (G): a fixed-capacity ghost set tracking recently evicted small-queue
               objects.  If a ghosted object is requested again, it is inserted
               directly into M instead of S.

Eviction logic:
  1. Scan Small from oldest to newest:
     – freq=0 → evict (return as victim), optionally add to ghost.
     – freq≥1 → promote to M (insert at head of M), reset freq=0, continue.
  2. If Small is exhausted, scan Main:
     – freq=0 → evict.
     – freq≥1 → decrement freq, reinsert at head of M.

Unlike LRU, S3-FIFO never rewrites object metadata on hits—just flips a
single bit—making it cache-friendly in hardware and very fast in practice.
"""

from collections import deque
from libcachesim import CommonCacheParams, Request


class S3FIFOCache:
    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.small_max  = max(1, cache_size // 9)    # 10% for small queue
        self.ghost_max  = cache_size * 4                   # ghost capacity (bytes)

        self.small: deque = deque()   # obj_ids, oldest at left
        self.main:  deque = deque()   # obj_ids, oldest at right (appendleft = newest)
        self.ghost: deque = deque()   # ghost FIFO for recently evicted small objects

        # obj_id → (size, freq)
        self.obj_info: dict = {}
        self.ghost_set: set = set()

        self.small_bytes: int = 0
        self.main_bytes:  int = 0
        self.ghost_bytes: int = 0

    # ------------------------------------------------------------------ #

    def _add_to_ghost(self, obj_id: int, size: int):
        if obj_id in self.ghost_set:
            return
        self.ghost.append(obj_id)
        self.ghost_set.add(obj_id)
        self.ghost_bytes += size
        # Trim ghost
        while self.ghost_bytes > self.ghost_max and self.ghost:
            old = self.ghost.popleft()
            if old in self.ghost_set:
                self.ghost_set.discard(old)
                # We don't track ghost sizes individually; just approximate
                self.ghost_bytes = max(0, self.ghost_bytes - size)

    def on_hit(self, req: Request):
        obj_id = req.obj_id
        if obj_id in self.obj_info:
            sz, freq = self.obj_info[obj_id]
            self.obj_info[obj_id] = (sz, min(freq + 1, 7))  # cap at 7; small resets to 0 on promotion

    def on_miss(self, req: Request):
        obj_id = req.obj_id
        size   = req.obj_size
        if size > self.cache_size:
            return
        if obj_id in self.obj_info:
            return  # already cached (shouldn't happen, but guard)

        if obj_id in self.ghost_set:
            # Ghost hit → insert directly into main
            self.ghost_set.discard(obj_id)
            self.main.appendleft(obj_id)
            self.obj_info[obj_id] = (size, 0)
            self.main_bytes += size
        else:
            # Fresh miss → insert into small
            self.small.append(obj_id)
            self.obj_info[obj_id] = (size, 0)
            self.small_bytes += size

    def _evict_small_one(self) -> int:
        """Pop from small: freq=0 → evict; freq≥1 → promote to main."""
        while self.small:
            obj_id = self.small.popleft()
            if obj_id not in self.obj_info:
                continue
            sz, freq = self.obj_info[obj_id]
            self.small_bytes -= sz
            if freq == 0:
                del self.obj_info[obj_id]
                self._add_to_ghost(obj_id, sz)
                return obj_id
            self.obj_info[obj_id] = (sz, 0)
            self.main.appendleft(obj_id)
            self.main_bytes += sz
        return 0

    def _evict_main_one(self) -> int:
        """Sweep main: freq=0 → evict; freq≥1 → decrement and reinsert at head."""
        scanned, limit = 0, len(self.main)
        while self.main and scanned <= limit:
            obj_id = self.main.pop()
            scanned += 1
            if obj_id not in self.obj_info:
                continue
            sz, freq = self.obj_info[obj_id]
            self.main_bytes -= sz
            if freq == 0:
                del self.obj_info[obj_id]
                return obj_id
            self.obj_info[obj_id] = (sz, freq - 1)
            self.main.appendleft(obj_id)
            self.main_bytes += sz
        return 0

    def evict(self, req: Request) -> int:
        # Enforce small/main split: if main has grown beyond its 90% target
        # (due to promotions), drain main first; otherwise drain small.
        main_max = self.cache_size - self.small_max
        if self.main_bytes > main_max and self.main:
            v = self._evict_main_one()
            if v:
                return v
        v = self._evict_small_one()
        if v:
            return v
        return self._evict_main_one()

    def on_remove(self, obj_id: int):
        if obj_id not in self.obj_info:
            return
        sz, _ = self.obj_info.pop(obj_id)
        # Remove from whichever queue it logically lives in.
        # Deque removal is O(n); acceptable for a plugin.
        try:
            self.small.remove(obj_id)
            self.small_bytes -= sz
        except ValueError:
            try:
                self.main.remove(obj_id)
                self.main_bytes -= sz
            except ValueError:
                pass
        self.ghost_set.discard(obj_id)


# ---------------------------------------------------------------------------
# Hook functions
# ---------------------------------------------------------------------------

def init_hook(common_cache_params: CommonCacheParams) -> S3FIFOCache:
    return S3FIFOCache(common_cache_params.cache_size)


def hit_hook(data: S3FIFOCache, req: Request):
    data.on_hit(req)


def miss_hook(data: S3FIFOCache, req: Request):
    data.on_miss(req)


def eviction_hook(data: S3FIFOCache, req: Request) -> int:
    return data.evict(req)


def remove_hook(data: S3FIFOCache, obj_id: int):
    data.on_remove(obj_id)


def free_hook(data: S3FIFOCache):
    data.small.clear()
    data.main.clear()
    data.ghost.clear()
    data.obj_info.clear()
    data.ghost_set.clear()


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
        path = f"/tmp/s3fifo_{label}.bin"
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
            cache_name="s3fifo",
        )
        reader = TraceReader(trace=wl["trace"], trace_type=wl["trace_type"])
        req_mr, byte_mr = cache.process_trace(reader)
        print(f"{wl['name']:<55} {req_mr:>10.4f} {byte_mr:>10.4f}")
