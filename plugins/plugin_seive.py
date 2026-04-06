from collections import deque
from libcachesim import CommonCacheParams, Request
class Node:
    def __init__(self, id):
        self.right = None
        self.left = None
        self.id = id
        self.accessed = False

class Seive:
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
        if req.obj_id in self.map_to_nodes:
            print("error, trying to insert node already in cache")
            return
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
            if self.pointer.left:
                self.pointer = self.pointer.left
            else:
                if self.pointer != self.head:
                    print("error, list is broken2")
                    return 0
                self.pointer = self.tail
        
        # evict the element
        if self.pointer.right:
            self.pointer.right.left = self.pointer.left
        else:
            if self.pointer != self.tail:
                print("error, list is broken 1")
                return 0
            self.tail = self.pointer.left
        if self.pointer.left:
            self.pointer.left.right = self.pointer.right
        else:
            if self.pointer != self.head:
                print("error, list is broken 3")
                return 0
            self.head = self.pointer.right
        
        if self.pointer.id not in self.map_to_nodes:
            print("error, map is broken")
            return 0
        del self.map_to_nodes[self.pointer.id]
        id = self.pointer.id

        if self.pointer.left:
            self.pointer = self.pointer.left
        else:
            self.pointer = self.tail

        return id


    def on_remove(self, obj_id: int):
        if obj_id in self.map_to_nodes:
            node = self.map_to_nodes[obj_id]
            if node.right:
                node.right.left = .node.left
            else:
                if node != self.tail:
                    print("error, list is broken 1")
                    return 0
                self.tail = node.left
            if node.left:
                node.left.right = node.right
            else:
                if node != self.head:
                    print("error, list is broken 3")
                    return 0
                self.head = node.right
            if node is self.pointer:
                self.pointer = node.left if node.left is not None else self.tail
            del self.map_to_nodes[obj_id]
        else:
            print("error, obj was not in cache")
            pass


def cache_init_hook(common_cache_params: CommonCacheParams):
    return Seive(common_cache_params.cache_size)


def cache_hit_hook(data: Seive, req: Request):
    data.on_hit(req)


def cache_miss_hook(data: Seive, req: Request):
    data.on_miss(req)


def cache_eviction_hook(data: Seive, req: Request):
    return data.evict(req)


def cache_remove_hook(data: Seive, obj_id: int):
    data.on_remove(obj_id)


def cache_free_hook(data: Seive):
    data.queue.clear()


if __name__ == "__main__":
    from pathlib import Path
    from libcachesim import PluginCache, TraceReader, TraceType

    plugin_seive_cache = PluginCache(
        cache_size=1024 * 1024,  # 1 MB
        cache_init_hook=cache_init_hook,
        cache_hit_hook=cache_hit_hook,
        cache_miss_hook=cache_miss_hook,
        cache_eviction_hook=cache_eviction_hook,
        cache_remove_hook=cache_remove_hook,
        cache_free_hook=cache_free_hook,
        cache_name="seive",
    )

    trace = Path(__file__).parent.parent / "data" / "cloudPhysicsIO.vscsi"
    reader = TraceReader(trace=str(trace), trace_type=TraceType.VSCSI_TRACE)

    req_miss_ratio, byte_miss_ratio = plugin_seivecache.process_trace(reader)
    print(f"Request miss ratio: {req_miss_ratio:.4f}")
    print(f"Byte miss ratio: {byte_miss_ratio:.4f}")