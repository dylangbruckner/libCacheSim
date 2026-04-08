"""
MAS3-FIFO — Metadata-Aware S3-FIFO with FIFO+second-chance main queue.

Same as MAS3 but keeps the original S3-FIFO second-chance FIFO for the main
queue (which outperforms LRU on highly-skewed Zipf workloads).  The only
addition over vanilla S3-FIFO is:

1. Unbounded metadata dict tracking (count, last_time, iat_ewma) for ALL
   ever-seen objects — even those long evicted.

2. IAT-based fast-track admission: a re-requested object that is NOT in the
   ghost set but has historically short inter-arrival time is inserted directly
   into main, bypassing the small-queue one-hit-wonder filter.
   This benefits objects that were in main, got evicted (and thus are NOT in
   ghost), and are requested again with short IAT.

3. Large ghost cache (default 10× cache_size in object-count units).

Tunable constants at the top of the file.
"""

from collections import deque
from libcachesim import CommonCacheParams, Request

# ── tunables ─────────────────────────────────────────────────────────────────
SMALL_RATIO  = 0.11   # fraction of cache for small queue
GHOST_RATIO  = 10.0   # ghost entries relative to cache_size (object-count proxy)
#                       Set high because real workloads benefit from large ghost.
#                       On Zipf this hurts slightly vs 1x, but course traces
#                       have shown consistent improvement up to 4-10x.
IAT_FACTOR   = 5.0    # IAT threshold = IAT_FACTOR * estimated_cache_capacity
#                       Objects with historical avg IAT < factor*capacity bypass
#                       the small queue filter (go directly to main).
#                       iat_factor ≤ 5.0 is essentially transparent on Zipf
#                       but can help real traces with recurring access patterns.
IAT_ALPHA    = 0.3    # EWMA weight for newest IAT sample
# ─────────────────────────────────────────────────────────────────────────────


class MAS3FIFOCache:
    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.small_max  = max(1, int(cache_size * SMALL_RATIO))

        # Small queue: FIFO with second-chance
        self.small: deque = deque()
        self.small_map: dict = {}   # obj_id → (size, freq)
        self.small_bytes: int = 0

        # Main queue: FIFO with second-chance (like original S3-FIFO)
        self.main: deque = deque()   # appendleft = MRU end, pop = LRU end
        self.obj_info: dict = {}     # obj_id → (size, freq)  for main queue
        self.main_bytes: int = 0

        # Ghost: bounded by object count
        self.ghost_max_n: int = int(cache_size * GHOST_RATIO)
        self.ghost_set: set   = set()
        self.ghost_q: deque   = deque()

        # Unbounded metadata for ALL seen objects
        # obj_id → (count, last_time, iat_ewma)
        self.metadata: dict = {}
        self.time: int      = 0

        # Running estimates for adaptive threshold
        self._total_size: int  = 0
        self._total_objs: int  = 0

    # ── helpers ───────────────────────────────────────────────────────────

    @property
    def _est_capacity(self) -> float:
        avg = self._total_size / self._total_objs if self._total_objs else 1.0
        return self.cache_size / max(1.0, avg)

    def _update_metadata(self, obj_id: int, size: int):
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
        # Update freq counter in whichever queue holds this object
        if obj_id in self.small_map:
            sz, freq = self.small_map[obj_id]
            self.small_map[obj_id] = (sz, min(freq + 1, 3))
        elif obj_id in self.obj_info:
            sz, freq = self.obj_info[obj_id]
            self.obj_info[obj_id] = (sz, min(freq + 1, 3))

    def on_miss(self, req: Request):
        obj_id = req.obj_id
        size   = req.obj_size
        self._update_metadata(obj_id, size)

        if size > self.cache_size:
            return
        if obj_id in self.small_map or obj_id in self.obj_info:
            return

        in_ghost   = obj_id in self.ghost_set
        fast_track = (not in_ghost) and self._should_fast_track(obj_id)

        if in_ghost or fast_track:
            self.ghost_set.discard(obj_id)
            self.main.appendleft(obj_id)
            self.obj_info[obj_id] = (size, 0)
            self.main_bytes += size
        else:
            self.small.append(obj_id)
            self.small_map[obj_id] = (size, 0)
            self.small_bytes += size

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
            # Promote to main (MRU end), reset freq
            self.main.appendleft(obj_id)
            self.obj_info[obj_id] = (sz, 0)
            self.main_bytes += sz
        return 0

    def _evict_main_one(self) -> int:
        scanned, limit = 0, len(self.main)
        while self.main and scanned <= limit:
            obj_id = self.main.pop()   # LRU end
            scanned += 1
            if obj_id not in self.obj_info:
                continue
            sz, freq = self.obj_info[obj_id]
            self.main_bytes -= sz
            if freq == 0:
                del self.obj_info[obj_id]
                return obj_id
            # Second chance: decrement freq, reinsert at MRU end
            self.obj_info[obj_id] = (sz, freq - 1)
            self.main.appendleft(obj_id)
            self.main_bytes += sz
        return 0

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
        elif obj_id in self.obj_info:
            sz, _ = self.obj_info.pop(obj_id)
            self.main_bytes -= sz
            try:
                self.main.remove(obj_id)
            except ValueError:
                pass
        self.ghost_set.discard(obj_id)


# ---------------------------------------------------------------------------
# Hook functions
# ---------------------------------------------------------------------------

def init_hook(common_cache_params: CommonCacheParams) -> MAS3FIFOCache:
    return MAS3FIFOCache(common_cache_params.cache_size)


def hit_hook(data: MAS3FIFOCache, req: Request):
    data.on_hit(req)


def miss_hook(data: MAS3FIFOCache, req: Request):
    data.on_miss(req)


def eviction_hook(data: MAS3FIFOCache, req: Request) -> int:
    return data.evict(req)


def remove_hook(data: MAS3FIFOCache, obj_id: int):
    data.on_remove(obj_id)


def free_hook(data: MAS3FIFOCache):
    data.small.clear()
    data.small_map.clear()
    data.main.clear()
    data.obj_info.clear()
    data.ghost_set.clear()
    data.ghost_q.clear()
    data.metadata.clear()
