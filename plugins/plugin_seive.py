from collections import deque
from libcachesim import CommonCacheParams, Request
class Node:
    def __init__(self, id):
        self.right = None
        self.left = None
        self.id = id
        self.accessed = False

class FifoCache:
    def __init__(self, cache_size: int):
        self.head = None
        self.tail = None
        self.pointer = self.head
        self.map_to_nodes = {}
        self.cache_size = cache_size

    def on_hit(self, req: Request):
        if req.obj_id not in self.map_to_nodes:
            print("error, map not consistent with cache")
            return
        self.map_to_nodes[req.obj_id].accessed = True

    def on_miss(self, req: Request):
        if req.obj_size <= self.cache_size:
            if not self.head:
                # first node, make head
                self.head = Node(req.obj_id)
                self.tail = self.head
                self.pointer = self.tail
                self.map_to_nodes[req.obj_id] = self.head
            else:
                # add node to head
                self.head.left = Node(req.obj_id)
                self.head.left.right = self.head
                self.head = self.head.left
                self.map_to_nodes[req.obj_id] = self.head


    def evict(self, req: Request):
        if not self.head:
            return 0

        # find the next element with the accessed bit set to 1
        while self.pointer.accessed:
            self.pointer.accessed = False
            if self.pointer.right:
                self.pointer = self.pointer.right
            else:
                if self.pointer != self.head:
                    print("error, list is broken")
                    return 0
                self.pointer = self.tail
        


    def on_remove(self, obj_id: int):
        try:
            self.queue.remove(obj_id)
        except ValueError:
            pass  # Object not in queue


def cache_init_hook(common_cache_params: CommonCacheParams):
    return FifoCache(common_cache_params.cache_size)


def cache_hit_hook(data: FifoCache, req: Request):
    data.on_hit(req)


def cache_miss_hook(data: FifoCache, req: Request):
    data.on_miss(req)


def cache_eviction_hook(data: FifoCache, req: Request):
    return data.evict(req)


def cache_remove_hook(data: FifoCache, obj_id: int):
    data.on_remove(obj_id)


def cache_free_hook(data: FifoCache):
    data.queue.clear()


if __name__ == "__main__":
    from pathlib import Path
    from libcachesim import PluginCache, TraceReader, TraceType

    plugin_fifo_cache = PluginCache(
        cache_size=1024 * 1024,  # 1 MB
        cache_init_hook=cache_init_hook,
        cache_hit_hook=cache_hit_hook,
        cache_miss_hook=cache_miss_hook,
        cache_eviction_hook=cache_eviction_hook,
        cache_remove_hook=cache_remove_hook,
        cache_free_hook=cache_free_hook,
        cache_name="fifo",
    )

    trace = Path(__file__).parent.parent / "data" / "cloudPhysicsIO.vscsi"
    reader = TraceReader(trace=str(trace), trace_type=TraceType.VSCSI_TRACE)

    req_miss_ratio, byte_miss_ratio = plugin_fifo_cache.process_trace(reader)
    print(f"Request miss ratio: {req_miss_ratio:.4f}")
    print(f"Byte miss ratio: {byte_miss_ratio:.4f}")