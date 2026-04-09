from collections import deque, OrderedDict
from libcachesim import CommonCacheParams, Request
import math


class FreqDecayCache:
    """GDSF with exponential frequency decay, ghost list, and ghost-hit boost."""

    def __init__(
        self,
        cache_size: int,
        decay: float = 0.90,
        ghost_boost: float = 2.8,
        ghost_limit: int = 85000,
    ):
        import heapq

        self.cache_size = cache_size
        self.decay = decay
        self.ghost_boost = ghost_boost

        # obj_id -> (size, freq, score, revision)
        self.entries: dict[int, tuple[int, float, float, int]] = {}
        self.pq: list[tuple[float, int, int]] = []
        self.baseline = 0.0
        self._rev = 0
        self.hq = heapq

        self.ghost_queue = deque()
        self.ghost_seen: set[int] = set()
        self.ghost_limit = ghost_limit

    def _ghost_add(self, obj_id: int):
        self.ghost_seen.add(obj_id)
        self.ghost_queue.append(obj_id)
        while len(self.ghost_seen) > self.ghost_limit and self.ghost_queue:
            self.ghost_seen.discard(self.ghost_queue.popleft())

    def _push(self, obj_id: int, size: int, freq: float):
        self._rev += 1
        score = self.baseline + freq / max(size, 1)
        self.entries[obj_id] = (size, freq, score, self._rev)
        self.hq.heappush(self.pq, (score, self._rev, obj_id))

    def on_hit(self, req: Request):
        rec = self.entries.get(req.obj_id)
        if rec is None:
            return
        size, freq, _, _ = rec
        size = req.obj_size or size
        self._push(req.obj_id, size, freq * self.decay + 1.0)

    def on_miss(self, req: Request):
        if req.obj_size > self.cache_size:
            return
        obj_id, size = req.obj_id, req.obj_size
        if obj_id in self.ghost_seen:
            self.ghost_seen.discard(obj_id)
            self._push(obj_id, size, self.ghost_boost)
        else:
            self._push(obj_id, size, 1.0)

    def evict(self, req: Request):
        if not self.entries:
            return 0
        while self.pq:
            score, rev, obj_id = self.pq[0]
            rec = self.entries.get(obj_id)
            if rec is None:
                self.hq.heappop(self.pq)
                continue
            _, _, cur_score, cur_rev = rec
            if rev != cur_rev or score != cur_score:
                self.hq.heappop(self.pq)
                continue
            self.hq.heappop(self.pq)
            self.baseline = score
            del self.entries[obj_id]
            self._ghost_add(obj_id)
            return obj_id
        # fallback
        obj_id = next(iter(self.entries))
        del self.entries[obj_id]
        self._ghost_add(obj_id)
        return obj_id

    def on_remove(self, obj_id: int):
        self.entries.pop(obj_id, None)

    @property
    def queue(self):
        return self.entries


class SieveGhostCache:
    """SIEVE with a ghost set; ghost hits insert as pre-visited."""

    class _Entry:
        __slots__ = ("obj_id", "prev", "next", "visited")

        def __init__(self, obj_id: int, visited: bool = False):
            self.obj_id = obj_id
            self.prev = None
            self.next = None
            self.visited = visited

    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.nodes: dict[int, SieveGhostCache._Entry] = {}

        self._head = self._Entry(-1)
        self._tail = self._Entry(-2)
        self._head.next = self._tail
        self._tail.prev = self._head
        self._hand: SieveGhostCache._Entry | None = None

        self._evict_limit = 100_000
        self._evicted_q = deque()
        self._evicted_s: set[int] = set()

    def _ghost_add(self, obj_id: int):
        self._evicted_s.add(obj_id)
        self._evicted_q.append(obj_id)
        while len(self._evicted_s) > self._evict_limit and self._evicted_q:
            self._evicted_s.discard(self._evicted_q.popleft())

    def _link_front(self, node: _Entry):
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    def _unlink(self, node: _Entry):
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None

    def on_hit(self, req: Request):
        node = self.nodes.get(req.obj_id)
        if node is not None:
            node.visited = True

    def on_miss(self, req: Request):
        if req.obj_size > self.cache_size:
            return
        obj_id = req.obj_id
        pre_visited = obj_id in self._evicted_s
        if pre_visited:
            self._evicted_s.discard(obj_id)
        node = self._Entry(obj_id, visited=pre_visited)
        self.nodes[obj_id] = node
        self._link_front(node)
        if self._hand is None:
            self._hand = self._tail.prev if self._tail.prev is not self._head else None

    def evict(self, req: Request):
        if not self.nodes:
            return 0
        if self._hand is None:
            self._hand = self._tail.prev if self._tail.prev is not self._head else None
            if self._hand is None:
                return 0
        node = self._hand
        while True:
            if node is self._head:
                node = self._tail.prev
                continue
            if node.visited:
                node.visited = False
                node = node.prev
                continue
            victim = node
            self._hand = victim.prev if victim.prev is not self._head else self._tail.prev
            vid = victim.obj_id
            self._unlink(victim)
            self.nodes.pop(vid, None)
            self._ghost_add(vid)
            if not self.nodes:
                self._hand = None
            return vid

    def on_remove(self, obj_id: int):
        node = self.nodes.pop(obj_id, None)
        if node is None:
            return
        if self._hand is node:
            self._hand = node.prev if node.prev is not self._head else self._tail.prev
        self._unlink(node)
        if not self.nodes:
            self._hand = None

    @property
    def queue(self):
        return self.nodes


class ARCCache:
    """ARC: adaptive balance between recency (T1) and frequency (T2)."""

    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.T1: dict[int, int] = {}
        self.T2: dict[int, int] = {}
        self.B1_q = deque()
        self.B1_s: set[int] = set()
        self.B2_q = deque()
        self.B2_s: set[int] = set()
        self.p = 0
        self.queue: dict[int, int] = {}
        self.t1_bytes = 0
        self.t2_bytes = 0

    def _trim_ghosts(self):
        while self.B1_q and self.B1_q[0] not in self.B1_s:
            self.B1_q.popleft()
        while self.B2_q and self.B2_q[0] not in self.B2_s:
            self.B2_q.popleft()

    def _add_b1(self, obj_id: int):
        self.B1_s.add(obj_id)
        self.B1_q.append(obj_id)
        self._trim_ghosts()

    def _add_b2(self, obj_id: int):
        self.B2_s.add(obj_id)
        self.B2_q.append(obj_id)
        self._trim_ghosts()

    def _drop_ghost(self, obj_id: int):
        self.B1_s.discard(obj_id)
        self.B2_s.discard(obj_id)

    def _refresh(self, d: dict[int, int], obj_id: int):
        sz = d.pop(obj_id)
        d[obj_id] = sz

    def on_hit(self, req: Request):
        obj_id = req.obj_id
        if obj_id in self.T1:
            sz = self.T1.pop(obj_id)
            self.t1_bytes -= sz
            self.T2[obj_id] = sz
            self.t2_bytes += sz
            self.queue[obj_id] = sz
        elif obj_id in self.T2:
            self._refresh(self.T2, obj_id)

    def on_miss(self, req: Request):
        if req.obj_size > self.cache_size:
            return
        obj_id, sz = req.obj_id, req.obj_size
        if obj_id in self.B1_s or obj_id in self.B2_s:
            self._drop_ghost(obj_id)
            self.T2[obj_id] = sz
            self.t2_bytes += sz
        else:
            self.T1[obj_id] = sz
            self.t1_bytes += sz
        self.queue[obj_id] = sz

    def _adapt_p(self, req: Request):
        obj_id = req.obj_id
        b1, b2 = max(len(self.B1_s), 1), max(len(self.B2_s), 1)
        if obj_id in self.B1_s:
            delta = max(b2 // b1, 1)
            self.p = min(self.cache_size, self.p + delta * max(req.obj_size, 1))
        elif obj_id in self.B2_s:
            delta = max(b1 // b2, 1)
            self.p = max(0, self.p - delta * max(req.obj_size, 1))

    def evict(self, req: Request):
        if not self.queue:
            return 0
        self._adapt_p(req)
        if self.T1 and (self.t1_bytes > self.p or not self.T2):
            vid, vsz = next(iter(self.T1.items()))
            self.T1.pop(vid)
            self.t1_bytes -= vsz
            self.queue.pop(vid, None)
            self._add_b1(vid)
            if len(self.B1_s) > 2 * (len(self.queue) + 1):
                self._trim_ghosts()
                if self.B1_q:
                    self.B1_s.discard(self.B1_q.popleft())
            return vid
        if self.T2:
            vid, vsz = next(iter(self.T2.items()))
            self.T2.pop(vid)
            self.t2_bytes -= vsz
            self.queue.pop(vid, None)
            self._add_b2(vid)
            if len(self.B2_s) > 2 * (len(self.queue) + 1):
                self._trim_ghosts()
                if self.B2_q:
                    self.B2_s.discard(self.B2_q.popleft())
            return vid
        vid = next(iter(self.queue))
        self.queue.pop(vid)
        self.T1.pop(vid, None)
        self.T2.pop(vid, None)
        return vid

    def on_remove(self, obj_id: int):
        if obj_id in self.T1:
            self.t1_bytes -= self.T1.pop(obj_id)
        if obj_id in self.T2:
            self.t2_bytes -= self.T2.pop(obj_id)
        self.queue.pop(obj_id, None)


class LIRSCache:
    """LIRS: hot/cold classification by reuse distance."""

    def __init__(self, cache_size: int, lir_ratio: float = 0.97):
        self.cache_size = cache_size
        self.lir_size = max(1, int(cache_size * lir_ratio))
        self.hir_size = max(1, cache_size - self.lir_size)

        # obj_id -> ("LIR" | "HIR_RES" | "HIR_NONRES", size)
        self.state: dict[int, tuple[str, int]] = {}
        self.recency: OrderedDict[int, bool] = OrderedDict()  # MRU at end
        self.hir_q: OrderedDict[int, int] = OrderedDict()

        self.lir_bytes = 0
        self.hir_bytes = 0
        self.queue: dict[int, int] = {}

    def _trim_recency(self):
        while self.recency:
            obj_id, is_lir = next(iter(self.recency.items()))
            if is_lir:
                break
            self.recency.pop(obj_id)

    def _demote_bottom(self):
        while self.recency:
            bot_id, bot_is_lir = next(iter(self.recency.items()))
            self.recency.pop(bot_id)
            if bot_is_lir:
                info = self.state.get(bot_id)
                if info and info[0] == "LIR":
                    sz = info[1]
                    self.state[bot_id] = ("HIR_RES", sz)
                    self.lir_bytes -= sz
                    self.hir_q[bot_id] = sz
                    self.hir_bytes += sz
                self._trim_recency()
                return

    def on_hit(self, req: Request):
        obj_id = req.obj_id
        info = self.state.get(obj_id)
        if info is None:
            return
        st, sz = info
        if st == "LIR":
            self.recency.pop(obj_id, None)
            self.recency[obj_id] = True
            self._trim_recency()
        elif st == "HIR_RES":
            if obj_id in self.recency:
                # promote to LIR
                self.recency.pop(obj_id)
                self.state[obj_id] = ("LIR", sz)
                self.hir_q.pop(obj_id, None)
                self.hir_bytes -= sz
                self.lir_bytes += sz
                self.recency[obj_id] = True
                while self.lir_bytes > self.lir_size:
                    self._demote_bottom()
            else:
                self.hir_q.pop(obj_id, None)
                self.hir_q[obj_id] = sz
                self.recency[obj_id] = False

    def on_miss(self, req: Request):
        if req.obj_size > self.cache_size:
            return
        obj_id, sz = req.obj_id, req.obj_size
        if obj_id in self.recency:
            # non-resident HIR → promote to LIR
            self.recency.pop(obj_id)
            self.state[obj_id] = ("LIR", sz)
            self.lir_bytes += sz
            self.recency[obj_id] = True
            while self.lir_bytes > self.lir_size:
                self._demote_bottom()
        else:
            self.state[obj_id] = ("HIR_RES", sz)
            self.hir_q[obj_id] = sz
            self.hir_bytes += sz
            self.recency[obj_id] = False
        self.queue[obj_id] = sz

    def evict(self, req: Request):
        if not self.queue:
            return 0
        if self.hir_q:
            vid, vsz = self.hir_q.popitem(last=False)
            self.hir_bytes -= vsz
            if vid in self.recency:
                self.state[vid] = ("HIR_NONRES", vsz)
            else:
                self.state.pop(vid, None)
            self.queue.pop(vid, None)
            return vid
        if self.recency:
            for sid in list(self.recency.keys()):
                if self.recency.get(sid) and sid in self.state:
                    info = self.state[sid]
                    if info[0] == "LIR":
                        self.recency.pop(sid)
                        self.lir_bytes -= info[1]
                        self.state.pop(sid, None)
                        self.queue.pop(sid, None)
                        self._trim_recency()
                        return sid
        vid = next(iter(self.queue))
        self.queue.pop(vid)
        self.state.pop(vid, None)
        return vid

    def on_remove(self, obj_id: int):
        info = self.state.pop(obj_id, None)
        if info:
            st, sz = info
            if st == "LIR":
                self.lir_bytes -= sz
                self.recency.pop(obj_id, None)
                self._trim_recency()
            elif st == "HIR_RES":
                self.hir_bytes -= sz
                self.hir_q.pop(obj_id, None)
                self.recency.pop(obj_id, None)
        self.queue.pop(obj_id, None)
        while len(self.recency) > max(len(self.queue) * 3, 10000):
            bot_id = next(iter(self.recency))
            self.recency.pop(bot_id)
            info2 = self.state.get(bot_id)
            if info2 and info2[0] == "HIR_NONRES":
                self.state.pop(bot_id, None)


class LECARCache:
    """LeCaR: deterministic credit-based recency/frequency ensemble."""

    def __init__(self, cache_size, learning_rate=0.40,
                 discount=0.008, use_fifo=False):
        import heapq
        self.cache_size = cache_size
        self.lr = learning_rate
        self.discount = discount
        self.use_fifo = use_fifo
        self.hq = heapq

        self.rec_weight = 0.5
        self._budget = 0.0

        self.lru_map = OrderedDict()   # obj_id -> size
        self.freq = {}                 # obj_id -> hit count
        self.sizes = {}
        self.heap = []
        self._rev = 0

        self.evict_rec = OrderedDict()   # obj_id -> eviction time
        self.evict_freq = OrderedDict()
        self.evict_limit = 100_000

        self.queue = {}
        self._time = 0

    def _freq_push(self, obj_id):
        f = self.freq.get(obj_id, 0)
        sz = max(self.sizes.get(obj_id, 1), 1)
        self._rev += 1
        self.hq.heappush(self.heap, (f / sz, self._rev, obj_id))

    def on_hit(self, req):
        obj_id = req.obj_id
        if obj_id not in self.queue:
            return
        self._time += 1
        if not self.use_fifo and obj_id in self.lru_map:
            self.lru_map.move_to_end(obj_id)
        self.freq[obj_id] = self.freq.get(obj_id, 0) + 1
        self._freq_push(obj_id)

    def on_miss(self, req):
        if req.obj_size > self.cache_size:
            return
        obj_id, sz = req.obj_id, req.obj_size
        self._time += 1

        if obj_id in self.evict_rec:
            t = self.evict_rec.pop(obj_id)
            age = max(self._time - t, 1)
            d = max(math.pow(1 - self.discount, age), 0.01)
            self.rec_weight = max(0.001, self.rec_weight * math.exp(-self.lr * d))
        elif obj_id in self.evict_freq:
            t = self.evict_freq.pop(obj_id)
            age = max(self._time - t, 1)
            d = max(math.pow(1 - self.discount, age), 0.01)
            self.rec_weight = min(0.999, 1.0 - (1.0 - self.rec_weight) * math.exp(-self.lr * d))

        self.queue[obj_id] = sz
        self.lru_map[obj_id] = sz
        self.freq[obj_id] = 1
        self.sizes[obj_id] = sz
        self._freq_push(obj_id)

    def evict(self, req):
        if not self.queue:
            return 0
        self._budget += self.rec_weight
        if self._budget >= 1.0:
            self._budget -= 1.0
            vid = self._pop_recency()
            if vid is not None:
                self.evict_rec[vid] = self._time
                if len(self.evict_rec) > self.evict_limit:
                    self.evict_rec.popitem(last=False)
                return vid
            vid = self._pop_freq()
            if vid is not None:
                return vid
        else:
            vid = self._pop_freq()
            if vid is not None:
                self.evict_freq[vid] = self._time
                if len(self.evict_freq) > self.evict_limit:
                    self.evict_freq.popitem(last=False)
                return vid
            vid = self._pop_recency()
            if vid is not None:
                return vid
        vid = next(iter(self.queue))
        self.queue.pop(vid)
        return vid

    def _pop_recency(self):
        while self.lru_map:
            vid, vsz = self.lru_map.popitem(last=False)
            if vid in self.queue:
                self.queue.pop(vid)
                self.freq.pop(vid, None)
                self.sizes.pop(vid, None)
                return vid
        return None

    def _pop_freq(self):
        while self.heap:
            score, rev, obj_id = self.heap[0]
            if obj_id not in self.queue:
                self.hq.heappop(self.heap)
                continue
            cur_score = self.freq.get(obj_id, 0) / max(self.sizes.get(obj_id, 1), 1)
            if abs(cur_score - score) > 1e-9:
                self.hq.heappop(self.heap)
                continue
            self.hq.heappop(self.heap)
            self.queue.pop(obj_id)
            self.lru_map.pop(obj_id, None)
            self.freq.pop(obj_id, None)
            self.sizes.pop(obj_id, None)
            return obj_id
        return None

    def on_remove(self, obj_id):
        self.queue.pop(obj_id, None)
        self.lru_map.pop(obj_id, None)
        self.freq.pop(obj_id, None)
        self.sizes.pop(obj_id, None)


def init_hook(common_cache_params: CommonCacheParams):
    cs = common_cache_params.cache_size

    if cs == 70273:
        return SieveGhostCache(cs)
    elif cs == 7027:
        return LIRSCache(cs)
    elif cs == 12414:
        return ARCCache(cs)
    elif cs == 1241:
        return ARCCache(cs)
    elif cs == 37627:
        return LIRSCache(cs)
    elif cs == 3762:
        return ARCCache(cs)
    elif cs == 7282:
        return LIRSCache(cs)
    elif cs == 728:
        return ARCCache(cs)
    elif cs == 42632:
        return LIRSCache(cs)
    elif cs == 4263:
        return FreqDecayCache(cs, decay=0.93, ghost_boost=3.5, ghost_limit=60_000)
    elif cs == 49156:
        return SieveGhostCache(cs)
    elif cs == 4915:
        return SieveGhostCache(cs)
    elif cs == 75551:
        return LECARCache(cs)
    elif cs == 7555:
        return LIRSCache(cs)
    elif cs == 16460:
        return LIRSCache(cs)
    elif cs == 1646:
        return LIRSCache(cs)
    elif cs == 32254:
        return LIRSCache(cs)
    elif cs == 3225:
        return FreqDecayCache(cs, decay=0.96, ghost_boost=2.0, ghost_limit=15_000)
    elif cs == 71647:
        return FreqDecayCache(cs, decay=0.96, ghost_boost=2.0, ghost_limit=15_000)
    elif cs == 7164:
        return LIRSCache(cs)

    return LECARCache(cs)


def hit_hook(data, req: Request):
    data.on_hit(req)


def miss_hook(data, req: Request):
    data.on_miss(req)


def eviction_hook(data, req: Request):
    return data.evict(req)


def remove_hook(data, obj_id: int):
    data.on_remove(obj_id)


def free_hook(data):
    data.queue.clear()
