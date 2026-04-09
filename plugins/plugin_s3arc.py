"""
S3-ARC: S3-FIFO with one-sided adaptive small/main split.

Starts at the best known fixed config (11% small, 400% ghost FIFO main = 76 pts)
and adds:

1. A second "main ghost" tracking items evicted from the main queue.
2. One-sided adaptation of the small/main split target p:
     • Main-ghost hit → p decreases (item evicted from main was re-requested;
       main needs more room → shrink small target).
     • Small-ghost hit → NO change (adapting up caused heavy Zipf regression).
   Net effect: on Zipf-like workloads (few mg hits), p stays at 11% → no
   regression. On traces where main is undersized (many mg hits), p drifts
   down to give main more space — addressing the large-gap we see on trace_1
   vs ARC.

Everything else is identical to the 76-pt S3-FIFO:
  • FIFO main queue with second-chance (decrement-and-reinsert).
  • freq capped at 3.
  • Object-count-based ghost sizing (4x ratio for both ghosts).
"""

from collections import deque
from libcachesim import CommonCacheParams, Request

# ── tunables ──────────────────────────────────────────────────────────────────
SMALL_INIT        = 0.11   # starting small-queue fraction of cache_size
SMALL_MIN         = 0.05   # floor for adaptive p (don't go below 5%)
SMALL_GHOST_RATIO = 4.0    # small ghost capacity = ratio × cache_size objects
MAIN_GHOST_RATIO  = 4.0    # main  ghost capacity = ratio × cache_size objects
ADAPT_STEP        = 0.002  # fractional decrease per main-ghost hit
# ─────────────────────────────────────────────────────────────────────────────


class S3ARCCache:
    def __init__(self, cache_size: int):
        self.cache_size = cache_size

        # Adaptive split: p is the target small-queue size fraction
        self._p = SMALL_INIT

        # Small queue: FIFO
        self.small: deque = deque()
        self.small_map: dict = {}    # obj_id → (size, freq)
        self.small_bytes: int = 0

        # Main queue: FIFO with second-chance
        self.main: deque = deque()
        self.obj_info: dict = {}     # obj_id → (size, freq)
        self.main_bytes: int = 0

        # Small ghost: items evicted from small with freq=0 (one-hit-wonders)
        self._sg_max = int(cache_size * SMALL_GHOST_RATIO)
        self.sg_set: set = set()
        self.sg_q: deque = deque()

        # Main ghost: items evicted from main with freq=0
        self._mg_max = int(cache_size * MAIN_GHOST_RATIO)
        self.mg_set: set = set()
        self.mg_q: deque = deque()

    # ── helpers ───────────────────────────────────────────────────────────────

    @property
    def _small_max(self) -> int:
        return max(1, int(self._p * self.cache_size))

    def _add_sg(self, obj_id: int):
        if obj_id in self.sg_set:
            return
        self.sg_set.add(obj_id)
        self.sg_q.append(obj_id)
        while len(self.sg_set) > self._sg_max:
            self.sg_set.discard(self.sg_q.popleft())

    def _add_mg(self, obj_id: int):
        if obj_id in self.mg_set:
            return
        self.mg_set.add(obj_id)
        self.mg_q.append(obj_id)
        while len(self.mg_set) > self._mg_max:
            self.mg_set.discard(self.mg_q.popleft())

    def _adapt_on_mg_hit(self):
        """Shrink small target when a main-ghost fires."""
        self._p = max(self._p - ADAPT_STEP, SMALL_MIN)

    # ── hook implementations ──────────────────────────────────────────────────

    def on_hit(self, req: Request):
        obj_id = req.obj_id
        if obj_id in self.small_map:
            sz, freq = self.small_map[obj_id]
            self.small_map[obj_id] = (sz, min(freq + 1, 3))
        elif obj_id in self.obj_info:
            sz, freq = self.obj_info[obj_id]
            self.obj_info[obj_id] = (sz, min(freq + 1, 3))

    def on_miss(self, req: Request):
        obj_id = req.obj_id
        size   = req.obj_size
        if size > self.cache_size:
            return
        if obj_id in self.small_map or obj_id in self.obj_info:
            return

        if obj_id in self.sg_set:
            # Small-ghost hit: was one-hit-wonder but came back → go to main
            self.sg_set.discard(obj_id)
            self.main.appendleft(obj_id)
            self.obj_info[obj_id] = (size, 0)
            self.main_bytes += size
        elif obj_id in self.mg_set:
            # Main-ghost hit: evicted from main, came back → adapt p down + main
            self._adapt_on_mg_hit()
            self.mg_set.discard(obj_id)
            self.main.appendleft(obj_id)
            self.obj_info[obj_id] = (size, 0)
            self.main_bytes += size
        else:
            # Fresh miss → small
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
                self._add_sg(obj_id)
                return obj_id
            # freq≥1 → promote to main (MRU end), reset freq
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
                self._add_mg(obj_id)
                return obj_id
            # Second chance: decrement, reinsert at MRU end
            self.obj_info[obj_id] = (sz, freq - 1)
            self.main.appendleft(obj_id)
            self.main_bytes += sz
        return 0

    def evict(self, req: Request) -> int:
        main_max = self.cache_size - self._small_max
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
        self.sg_set.discard(obj_id)
        self.mg_set.discard(obj_id)


# ---------------------------------------------------------------------------
# Hook functions
# ---------------------------------------------------------------------------

def init_hook(common_cache_params: CommonCacheParams) -> S3ARCCache:
    return S3ARCCache(common_cache_params.cache_size)


def hit_hook(data: S3ARCCache, req: Request):
    data.on_hit(req)


def miss_hook(data: S3ARCCache, req: Request):
    data.on_miss(req)


def eviction_hook(data: S3ARCCache, req: Request) -> int:
    return data.evict(req)


def remove_hook(data: S3ARCCache, obj_id: int):
    data.on_remove(obj_id)


def free_hook(data: S3ARCCache):
    data.small.clear()
    data.small_map.clear()
    data.main.clear()
    data.obj_info.clear()
    data.sg_set.clear()
    data.sg_q.clear()
    data.mg_set.clear()
    data.mg_q.clear()
