"""
MAS3 — Metadata-Aware S3-FIFO.

Builds on S3-FIFO by exploiting two additional ideas:

1. Unbounded metadata dictionary
   Every object ever requested is tracked: (count, last_time, iat_ewma).
   This is possible because the assignment allows relatively unbounded
   metadata storage.  Even objects long since evicted retain their history.

2. IAT-based fast-track admission
   When an object is re-requested and is NOT in the ghost set (e.g. it was
   previously in main and got evicted, or its ghost entry aged out), its
   history can still classify it as "frequently recurring":
     • count ≥ 2  (seen at least twice)
     • iat_ewma < iat_factor * estimated_cache_capacity
   Such objects are admitted directly to main, bypassing the small-queue
   one-hit-wonder filter.

3. LRU main queue
   Instead of S3-FIFO's FIFO + second-chance scan, the main queue is a
   plain LRU (OrderedDict).  Every hit moves the object to the MRU end;
   eviction always removes from the LRU end.  This keeps medium-frequency
   objects alive longer than FIFO + second-chance in many real workloads.

Architecture:
  • small  (10%): FIFO, second-chance (freq=0 → evict; freq≥1 → promote).
  • main   (90%): LRU via OrderedDict.
  • ghost       : bounded FIFO set of recently evicted small objects.
  • metadata    : unbounded dict {obj_id: (count, last_time, iat_ewma)}.

Tunable at top of class:
  SMALL_RATIO   = 0.10   # fraction of cache for small queue
  GHOST_RATIO   = 4.0    # ghost size relative to cache (in object-count units)
  IAT_FACTOR    = 2.0    # objects with iat < IAT_FACTOR * capacity → fast-track
"""

from collections import deque, OrderedDict
from libcachesim import CommonCacheParams, Request

# ── tunables (hardcoded for submission) ──────────────────────────────────────
SMALL_RATIO = 0.10
GHOST_RATIO = 4.0    # ghost measured in "number of objects" ≈ cache_size * ratio
IAT_FACTOR  = 2.0    # lower → stricter fast-track (fewer items bypass small queue)
IAT_ALPHA   = 0.3    # EWMA weight for newest IAT sample
# ─────────────────────────────────────────────────────────────────────────────


class MAS3Cache:
    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.small_max  = max(1, int(cache_size * SMALL_RATIO))

        # Small queue: FIFO with second-chance
        self.small: deque = deque()
        self.small_map: dict = {}   # obj_id → (size, freq)
        self.small_bytes: int = 0

        # Main queue: LRU (last=MRU, first=LRU)
        self.main: OrderedDict = OrderedDict()  # obj_id → size
        self.main_bytes: int   = 0

        # Ghost: bounded by object count
        self.ghost_max_n: int = int(cache_size * GHOST_RATIO)
        self.ghost_set: set   = set()
        self.ghost_q: deque   = deque()

        # Unbounded metadata for ALL seen objects
        # obj_id → [count, last_time, iat_ewma]
        self.metadata: dict = {}
        self.time: int      = 0

        # Running estimates for avg object size (to compute cache capacity)
        self._total_size: int  = 0
        self._total_objs: int  = 0

    # ── internal helpers ──────────────────────────────────────────────────

    @property
    def _avg_obj_size(self) -> float:
        return self._total_size / self._total_objs if self._total_objs else 1.0

    @property
    def _est_capacity(self) -> float:
        """Estimated number of objects the cache can hold."""
        return self.cache_size / max(1.0, self._avg_obj_size)

    def _update_metadata(self, obj_id: int, size: int):
        """Update access metadata for obj_id (called on every hit or miss)."""
        self.time += 1
        if obj_id in self.metadata:
            count, last_time, iat_ewma = self.metadata[obj_id]
            iat = self.time - last_time
            new_ewma = (1 - IAT_ALPHA) * iat_ewma + IAT_ALPHA * iat if iat_ewma > 0 else float(iat)
            self.metadata[obj_id] = (count + 1, self.time, new_ewma)
        else:
            self.metadata[obj_id] = (1, self.time, 0.0)
            self._total_objs += 1
            self._total_size += size

    def _should_fast_track(self, obj_id: int) -> bool:
        """
        True if historical access pattern justifies bypassing the small queue.
        Requires: seen ≥2 times AND avg IAT < IAT_FACTOR * estimated_capacity.
        """
        meta = self.metadata.get(obj_id)
        if not meta:
            return False
        count, _last, iat_ewma = meta
        if count < 2 or iat_ewma <= 0:
            return False
        return iat_ewma < IAT_FACTOR * self._est_capacity

    def _add_to_ghost(self, obj_id: int):
        if obj_id in self.ghost_set:
            return
        self.ghost_set.add(obj_id)
        self.ghost_q.append(obj_id)
        while len(self.ghost_set) > self.ghost_max_n:
            self.ghost_set.discard(self.ghost_q.popleft())

    # ── hook implementations ──────────────────────────────────────────────

    def on_hit(self, req: Request):
        obj_id = req.obj_id
        self._update_metadata(obj_id, req.obj_size)

        if obj_id in self.small_map:
            sz, freq = self.small_map[obj_id]
            self.small_map[obj_id] = (sz, min(freq + 1, 3))
        elif obj_id in self.main:
            self.main.move_to_end(obj_id)  # LRU: promote to MRU

    def on_miss(self, req: Request):
        obj_id = req.obj_id
        size   = req.obj_size
        self._update_metadata(obj_id, size)

        if size > self.cache_size:
            return
        if obj_id in self.small_map or obj_id in self.main:
            return  # already cached (shouldn't happen, guard)

        in_ghost     = obj_id in self.ghost_set
        fast_track   = (not in_ghost) and self._should_fast_track(obj_id)

        if in_ghost or fast_track:
            # Promote directly to main (bypass small-queue filter)
            self.ghost_set.discard(obj_id)
            self.main[obj_id] = size
            self.main_bytes  += size
            self.main.move_to_end(obj_id)
        else:
            # Normal path: enter small queue (one-hit-wonder filter)
            self.small.append(obj_id)
            self.small_map[obj_id] = (size, 0)
            self.small_bytes      += size

    def _evict_small_one(self) -> int:
        while self.small:
            obj_id = self.small.popleft()
            if obj_id not in self.small_map:
                continue
            sz, freq = self.small_map.pop(obj_id)
            self.small_bytes -= sz
            if freq == 0:
                self._add_to_ghost(obj_id)
                return obj_id
            # freq ≥ 1 → promote to main (LRU, MRU end)
            self.main[obj_id] = sz
            self.main_bytes  += sz
            self.main.move_to_end(obj_id)
        return 0

    def _evict_main_one(self) -> int:
        if not self.main:
            return 0
        obj_id, sz = self.main.popitem(last=False)  # LRU end
        self.main_bytes -= sz
        return obj_id

    def evict(self, req: Request) -> int:
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
        if obj_id in self.small_map:
            sz, _ = self.small_map.pop(obj_id)
            self.small_bytes -= sz
            try:
                self.small.remove(obj_id)
            except ValueError:
                pass
        elif obj_id in self.main:
            sz = self.main.pop(obj_id)
            self.main_bytes -= sz
        self.ghost_set.discard(obj_id)


# ---------------------------------------------------------------------------
# Hook functions
# ---------------------------------------------------------------------------

def init_hook(common_cache_params: CommonCacheParams) -> MAS3Cache:
    return MAS3Cache(common_cache_params.cache_size)


def hit_hook(data: MAS3Cache, req: Request):
    data.on_hit(req)


def miss_hook(data: MAS3Cache, req: Request):
    data.on_miss(req)


def eviction_hook(data: MAS3Cache, req: Request) -> int:
    return data.evict(req)


def remove_hook(data: MAS3Cache, obj_id: int):
    data.on_remove(obj_id)


def free_hook(data: MAS3Cache):
    data.small.clear()
    data.small_map.clear()
    data.main.clear()
    data.ghost_set.clear()
    data.ghost_q.clear()
    data.metadata.clear()


# ---------------------------------------------------------------------------
# Self-test benchmark
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import struct, os
    import numpy as np
    from pathlib import Path
    from libcachesim import PluginCache, TraceReader, TraceType

    DATA_DIR = Path(__file__).parent.parent / "data"

    def gen_zipf_trace(path, n_obj, n_req, alpha, seed=42):
        if os.path.exists(path):
            return
        rng = np.random.default_rng(seed)
        np_tmp = np.power(np.arange(1, n_obj + 1), -alpha)
        dist_map = np.cumsum(np_tmp) / np_tmp.sum()
        reqs = np.searchsorted(dist_map, rng.uniform(0, 1, n_req)) + 1
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
    for label, alpha, sz in [("zipf_1.2", 1.2, 500), ("zipf_1.0", 1.0, 500), ("zipf_0.7", 0.7, 500)]:
        path = f"/tmp/mas3_{label}.bin"
        gen_zipf_trace(path, 10_000, 500_000, alpha)
        workloads.append({
            "name": f"{label} ({sz}/10k objs)",
            "trace": path,
            "trace_type": TraceType.ORACLE_GENERAL_TRACE,
            "cache_size": sz,
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
            cache_name="mas3",
        )
        reader = TraceReader(trace=wl["trace"], trace_type=wl["trace_type"])
        req_mr, byte_mr = cache.process_trace(reader)
        print(f"{wl['name']:<55} {req_mr:>10.4f} {byte_mr:>10.4f}")
