"""
S3-FIFO with LRU main queue.

Identical to S3-FIFO except the main queue uses LRU (via OrderedDict) instead
of FIFO with a second-chance scan.  On every hit to a main-queue object the
entry is moved to the MRU end; eviction always removes from the LRU end.

This keeps frequently re-accessed objects warm even when the main queue is
large, which helps workloads where the same objects are accessed at medium
intervals (not caught by the small-queue second-chance but still worth keeping).
"""

from collections import deque, OrderedDict
from libcachesim import CommonCacheParams, Request


class S3FIFOLRUCache:
    def __init__(self, cache_size: int,
                 small_ratio: float = 0.10,
                 ghost_ratio: float = 4.0):
        self.cache_size = cache_size
        self.small_max  = max(1, int(cache_size * small_ratio))

        # Main queue: LRU via OrderedDict (last = MRU, first = LRU)
        self.main: OrderedDict = OrderedDict()   # obj_id → size
        self.main_bytes: int   = 0

        # Small queue: FIFO with second-chance
        self.small: deque = deque()
        self.small_map: dict = {}   # obj_id → (size, freq)
        self.small_bytes: int = 0

        # Ghost: track recently evicted small objects (obj count bound)
        self.ghost_max_n: int = int(cache_size * ghost_ratio)
        self.ghost_set: set   = set()
        self.ghost_q: deque   = deque()

    # ------------------------------------------------------------------ #

    def _add_to_ghost(self, obj_id: int):
        if obj_id in self.ghost_set:
            return
        self.ghost_set.add(obj_id)
        self.ghost_q.append(obj_id)
        while len(self.ghost_set) > self.ghost_max_n:
            self.ghost_set.discard(self.ghost_q.popleft())

    def on_hit(self, req: Request):
        obj_id = req.obj_id
        if obj_id in self.small_map:
            sz, freq = self.small_map[obj_id]
            self.small_map[obj_id] = (sz, min(freq + 1, 3))
        elif obj_id in self.main:
            self.main.move_to_end(obj_id)  # promote to MRU

    def on_miss(self, req: Request):
        obj_id = req.obj_id
        size   = req.obj_size
        if size > self.cache_size:
            return
        if obj_id in self.small_map or obj_id in self.main:
            return

        if obj_id in self.ghost_set:
            self.ghost_set.discard(obj_id)
            self.main[obj_id] = size
            self.main_bytes += size
            self.main.move_to_end(obj_id)
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
            # Promote to main (MRU end)
            self.main[obj_id] = sz
            self.main_bytes += sz
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

def init_hook(common_cache_params: CommonCacheParams) -> S3FIFOLRUCache:
    return S3FIFOLRUCache(common_cache_params.cache_size)


def hit_hook(data: S3FIFOLRUCache, req: Request):
    data.on_hit(req)


def miss_hook(data: S3FIFOLRUCache, req: Request):
    data.on_miss(req)


def eviction_hook(data: S3FIFOLRUCache, req: Request) -> int:
    return data.evict(req)


def remove_hook(data: S3FIFOLRUCache, obj_id: int):
    data.on_remove(obj_id)


def free_hook(data: S3FIFOLRUCache):
    data.small.clear()
    data.small_map.clear()
    data.main.clear()
    data.ghost_set.clear()
    data.ghost_q.clear()
