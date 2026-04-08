"""
2Q (Two Queue) cache eviction algorithm — Johnson & Shasha, VLDB '94.

2Q splits the cache into:
  • A1in  (FIFO, ~25% of capacity) — objects seen for the first time
  • Am    (LRU,  ~75% of capacity) — objects seen more than once
  • A1out (ghost FIFO)             — recently evicted from A1in

On first access an object enters A1in.  If it is accessed again while still
in A1in, or if it returns via the ghost A1out, it is promoted to Am (LRU).
Hits in Am simply refresh the LRU order.  When eviction is needed, if A1in
has grown too large the LRU of A1in is evicted (moved to A1out ghost);
otherwise the LRU of Am is evicted.

This cleanly separates one-hit wonders (A1in/A1out) from frequently accessed
objects (Am) without the complexity of full frequency counting.
"""

from collections import deque, OrderedDict
from libcachesim import CommonCacheParams, Request


class TwoQCache:
    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.kin = max(1, cache_size // 4)    # target byte capacity for A1in (25%)
        self.km  = cache_size - self.kin       # target byte capacity for Am  (75%)

        self.a1in:  deque         = deque()        # FIFO queue: obj_ids (newest at right)
        self.am:    OrderedDict   = OrderedDict()  # LRU map: obj_id -> size
        self.a1out: deque         = deque()        # ghost FIFO (obj_ids only)
        self.a1out_set: set       = set()          # fast membership test for a1out

        self.a1in_map:  dict = {}  # obj_id -> size (for a1in objects)
        self.a1in_bytes: int  = 0
        self.am_bytes:   int  = 0
        self.a1out_max:  int  = max(1, cache_size // 2)  # ghost capacity (obj count)

    def on_hit(self, req: Request):
        obj_id = req.obj_id
        if obj_id in self.a1in_map:
            # Still in A1in on re-access — promote to Am
            size = self.a1in_map.pop(obj_id)
            self.a1in_bytes -= size
            try:
                self.a1in.remove(obj_id)
            except ValueError:
                pass
            self.am[obj_id] = size
            self.am.move_to_end(obj_id)
            self.am_bytes += size
        elif obj_id in self.am:
            self.am.move_to_end(obj_id)

    def on_miss(self, req: Request):
        obj_id  = req.obj_id
        size    = req.obj_size
        if size > self.cache_size:
            return

        if obj_id in self.a1out_set:
            # Ghost hit → place in Am (it was recently in A1in)
            self.a1out_set.discard(obj_id)
            self.am[obj_id] = size
            self.am.move_to_end(obj_id)
            self.am_bytes += size
        else:
            # Brand new object → A1in
            self.a1in.append(obj_id)
            self.a1in_map[obj_id] = size
            self.a1in_bytes += size

    def evict(self, req: Request) -> int:
        # Evict from A1in if it is over target, else from Am
        if self.a1in_bytes > self.kin and self.a1in:
            obj_id = self.a1in.popleft()
            size = self.a1in_map.pop(obj_id, 0)
            self.a1in_bytes -= size
            # Add to ghost
            self.a1out.append(obj_id)
            self.a1out_set.add(obj_id)
            # Trim ghost
            while len(self.a1out) > self.a1out_max:
                old = self.a1out.popleft()
                self.a1out_set.discard(old)
            return obj_id

        if self.am:
            obj_id, size = self.am.popitem(last=False)
            self.am_bytes -= size
            return obj_id

        # Fallback: evict from A1in even if under target
        if self.a1in:
            obj_id = self.a1in.popleft()
            size = self.a1in_map.pop(obj_id, 0)
            self.a1in_bytes -= size
            self.a1out.append(obj_id)
            self.a1out_set.add(obj_id)
            while len(self.a1out) > self.a1out_max:
                old = self.a1out.popleft()
                self.a1out_set.discard(old)
            return obj_id

        return 0

    def on_remove(self, obj_id: int):
        if obj_id in self.a1in_map:
            size = self.a1in_map.pop(obj_id)
            self.a1in_bytes -= size
            try:
                self.a1in.remove(obj_id)
            except ValueError:
                pass
        elif obj_id in self.am:
            size = self.am.pop(obj_id)
            self.am_bytes -= size
        self.a1out_set.discard(obj_id)


# ---------------------------------------------------------------------------
# Hook functions
# ---------------------------------------------------------------------------

def init_hook(common_cache_params: CommonCacheParams) -> TwoQCache:
    return TwoQCache(common_cache_params.cache_size)


def hit_hook(data: TwoQCache, req: Request):
    data.on_hit(req)


def miss_hook(data: TwoQCache, req: Request):
    data.on_miss(req)


def eviction_hook(data: TwoQCache, req: Request) -> int:
    return data.evict(req)


def remove_hook(data: TwoQCache, obj_id: int):
    data.on_remove(obj_id)


def free_hook(data: TwoQCache):
    data.a1in.clear()
    data.a1in_map.clear()
    data.am.clear()
    data.a1out.clear()
    data.a1out_set.clear()


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
        path = f"/tmp/2q_{label}.bin"
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
            cache_name="2q",
        )
        reader = TraceReader(trace=wl["trace"], trace_type=wl["trace_type"])
        req_mr, byte_mr = cache.process_trace(reader)
        print(f"{wl['name']:<55} {req_mr:>10.4f} {byte_mr:>10.4f}")
