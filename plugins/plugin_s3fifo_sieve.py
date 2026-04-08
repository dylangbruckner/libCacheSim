"""
S3-FIFO + SIEVE Hybrid — original combination.

Combines the two-queue insight of S3-FIFO (small buffer for new objects,
main buffer for promoted objects) with SIEVE's lazy promotion / hand-based
eviction strategy for the main queue.

Architecture:
  • Small queue (10% of cache):  FIFO.  New objects enter here.
    – freq=0 on eviction → discard (add to ghost set).
    – freq≥1 on eviction → promote to main.
  • Main queue  (90% of cache):  SIEVE.  A "hand" pointer sweeps from tail
    to head; objects with accessed=True get their bit cleared (second chance);
    objects with accessed=False are evicted.
  • Ghost set: recently evicted small-queue objects.  If a ghost object is
    requested again, it is inserted directly into the main queue instead of
    the small queue.

Why this matters: S3-FIFO's main queue uses plain FIFO reinsertion (a
coarse second-chance policy).  Replacing it with SIEVE's fine-grained hand
sweep provides better hit-ratio on workloads with mixed access frequencies.
"""

from collections import deque
from libcachesim import CommonCacheParams, Request


class SieveNode:
    __slots__ = ("obj_id", "size", "accessed", "prev", "next")

    def __init__(self, obj_id: int, size: int):
        self.obj_id   = obj_id
        self.size     = size
        self.accessed = False
        self.prev     = None
        self.next     = None


class SieveList:
    """Doubly-linked list supporting O(1) insert-at-head, remove, and hand sweep."""

    def __init__(self):
        # Sentinel nodes
        self._head = SieveNode(-1, 0)  # MRU end
        self._tail = SieveNode(-2, 0)  # LRU end
        self._head.next = self._tail
        self._tail.prev = self._head
        self._hand: SieveNode = self._tail   # hand starts at tail (oldest)
        self._size_bytes = 0

    def insert(self, node: SieveNode):
        """Insert at head (MRU end)."""
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node
        self._size_bytes += node.size

    def remove(self, node: SieveNode):
        """Remove an arbitrary node."""
        if self._hand is node:
            self._hand = node.next if node.next is not self._tail else self._tail
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None
        self._size_bytes -= node.size

    def evict_one(self) -> "SieveNode | None":
        """Sweep hand; evict first unaccessed node."""
        # Start from hand (or tail sentinel's prev if hand is sentinel)
        cur = self._hand
        if cur is self._tail or cur is self._head:
            cur = self._tail.prev

        visited = 0
        total = 0
        # Count items
        tmp = self._head.next
        while tmp is not self._tail:
            total += 1
            tmp = tmp.next
        if total == 0:
            return None

        while visited <= total:
            if cur is self._head or cur is self._tail:
                cur = self._tail.prev
                if cur is self._head:
                    return None
                visited += 1
                continue
            if not cur.accessed:
                victim = cur
                self._hand = cur.next if cur.next is not self._tail else self._tail.prev
                self.remove(victim)
                return victim
            cur.accessed = False
            self._hand = cur
            cur = cur.prev if cur.prev is not self._head else self._tail.prev
            visited += 1

        # Fallback: evict tail
        if self._tail.prev is not self._head:
            victim = self._tail.prev
            self.remove(victim)
            return victim
        return None

    def __len__(self):
        tmp, n = self._head.next, 0
        while tmp is not self._tail:
            n += 1
            tmp = tmp.next
        return n


class S3FIFOSieveCache:
    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.small_max  = max(1, cache_size // 10)

        self.small: deque = deque()   # FIFO small queue (obj_ids)
        self.main:  SieveList = SieveList()  # SIEVE main queue

        self.small_map: dict = {}     # obj_id → (node=None, size, freq)  for small
        self.main_map:  dict = {}     # obj_id → SieveNode                for main

        self.ghost_set: set  = set()
        self.ghost_q:   deque = deque()
        self.ghost_max  = max(1, cache_size)

        self.small_bytes: int = 0

    # ------------------------------------------------------------------ #

    def _add_to_ghost(self, obj_id: int):
        if obj_id in self.ghost_set:
            return
        self.ghost_q.append(obj_id)
        self.ghost_set.add(obj_id)
        while len(self.ghost_set) > self.ghost_max:
            old = self.ghost_q.popleft()
            self.ghost_set.discard(old)

    def on_hit(self, req: Request):
        obj_id = req.obj_id
        if obj_id in self.small_map:
            sz, freq = self.small_map[obj_id]
            self.small_map[obj_id] = (sz, min(freq + 1, 3))
        elif obj_id in self.main_map:
            self.main_map[obj_id].accessed = True

    def on_miss(self, req: Request):
        obj_id = req.obj_id
        size   = req.obj_size
        if size > self.cache_size or obj_id in self.small_map or obj_id in self.main_map:
            return

        if obj_id in self.ghost_set:
            # Ghost hit → directly to main (SIEVE list)
            self.ghost_set.discard(obj_id)
            node = SieveNode(obj_id, size)
            self.main.insert(node)
            self.main_map[obj_id] = node
        else:
            # Fresh miss → small queue (FIFO)
            self.small.append(obj_id)
            self.small_map[obj_id] = (size, 0)
            self.small_bytes += size

    def evict(self, req: Request) -> int:
        # Phase 1: evict from small (promote freq≥1 to main SIEVE)
        scanned = 0
        limit = len(self.small)
        while self.small and scanned <= limit:
            obj_id = self.small.popleft()
            scanned += 1
            if obj_id not in self.small_map:
                continue
            sz, freq = self.small_map.pop(obj_id)
            self.small_bytes -= sz
            if freq == 0:
                self._add_to_ghost(obj_id)
                return obj_id
            else:
                # Promote to main SIEVE
                node = SieveNode(obj_id, sz)
                self.main.insert(node)
                self.main_map[obj_id] = node

        # Phase 2: evict from main using SIEVE hand sweep
        victim = self.main.evict_one()
        if victim:
            del self.main_map[victim.obj_id]
            return victim.obj_id

        return 0

    def on_remove(self, obj_id: int):
        if obj_id in self.small_map:
            sz, _ = self.small_map.pop(obj_id)
            self.small_bytes -= sz
            try:
                self.small.remove(obj_id)
            except ValueError:
                pass
        elif obj_id in self.main_map:
            node = self.main_map.pop(obj_id)
            self.main.remove(node)
        self.ghost_set.discard(obj_id)


# ---------------------------------------------------------------------------
# Hook functions
# ---------------------------------------------------------------------------

def init_hook(common_cache_params: CommonCacheParams) -> S3FIFOSieveCache:
    return S3FIFOSieveCache(common_cache_params.cache_size)


def hit_hook(data: S3FIFOSieveCache, req: Request):
    data.on_hit(req)


def miss_hook(data: S3FIFOSieveCache, req: Request):
    data.on_miss(req)


def eviction_hook(data: S3FIFOSieveCache, req: Request) -> int:
    return data.evict(req)


def remove_hook(data: S3FIFOSieveCache, obj_id: int):
    data.on_remove(obj_id)


def free_hook(data: S3FIFOSieveCache):
    data.small.clear()
    data.small_map.clear()
    data.main_map.clear()
    data.ghost_set.clear()
    data.ghost_q.clear()


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
        path = f"/tmp/s3fifo_sieve_{label}.bin"
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
            cache_name="s3fifo-sieve",
        )
        reader = TraceReader(trace=wl["trace"], trace_type=wl["trace_type"])
        req_mr, byte_mr = cache.process_trace(reader)
        print(f"{wl['name']:<55} {req_mr:>10.4f} {byte_mr:>10.4f}")
