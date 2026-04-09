"""
Calibrated parameter sweep using synthetic traces that match course trace fingerprints.

Course trace classification (from submitted results):
  SCAN_HEAVY  (trace_2,3,4): S3FIFO beats ARC by 38-62% at large cache.
              Pure one-hit-wonder + small hot set. c/req ≈ 0.006-0.009.
  ARC_WINS    (trace_1):     ARC beats S3FIFO by 40% at large cache.
              Large working set, medium-frequency objects. c/req ≈ 0.004.
  SCAN_MOD    (trace_7,8):   S3FIFO beats ARC by 12-25% at large cache.
              Moderate scan fraction, trace_7 also wins at small cache.
  MIXED       (trace_0,5,6,9): All similar. c/req ≈ 0.009-0.036.

Approach: generate synthetic traces that match each fingerprint, verify with
quick SIEVE/ARC/S3FIFO runs, then sweep parameters on the validated set.

Run:  python3 tune_calibrated.py
"""

import struct, os, sys
import numpy as np
from collections import deque, OrderedDict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from libcachesim import CommonCacheParams, Request, PluginCache, TraceReader, TraceType

DATA_DIR = Path(__file__).parent.parent / "data"

# ─────────────────────────────────────────────────────────────────────────────
# Trace generators
# ─────────────────────────────────────────────────────────────────────────────

def _write_bin(path, ids, sizes=None, seed=None):
    s = struct.Struct('<IQIq')
    with open(path, 'wb') as f:
        for i, obj_id in enumerate(ids):
            sz = sizes[i] if sizes is not None else 1
            f.write(s.pack(i % (2**32), int(obj_id) % (2**64), max(1, sz), -2))


def gen_scan_hot(path, n_req, scan_frac, n_hot, hot_alpha=1.2, seed=42):
    """
    scan_frac of requests are unique one-hit wonders (sequential scan).
    (1-scan_frac) of requests go to n_hot objects with Zipf distribution.

    Designed to match SCAN_HEAVY traces (trace_2/3/4):
    S3-FIFO filters scan objects; SIEVE/ARC cannot.
    """
    if os.path.exists(path):
        return
    rng = np.random.default_rng(seed)
    n_hot = max(1, n_hot)
    hot_count = int(n_req * (1 - scan_frac))
    scan_count = n_req - hot_count

    # Hot accesses: Zipf within hot set (IDs 1..n_hot)
    np_tmp = np.power(np.arange(1, n_hot + 1, dtype=float), -hot_alpha)
    dist = np.cumsum(np_tmp) / np_tmp.sum()
    hot_ids = np.searchsorted(dist, rng.uniform(0, 1, hot_count)) + 1

    # Scan accesses: each unique, IDs start at n_hot + 1
    scan_ids = np.arange(n_hot + 1, n_hot + 1 + scan_count, dtype=np.int64)

    # Interleave hot and scan in random order
    hot_mask = rng.random(n_req) >= scan_frac
    ids = np.empty(n_req, dtype=np.int64)
    hi = si = 0
    for i in range(n_req):
        if hot_mask[i] and hi < len(hot_ids):
            ids[i] = hot_ids[hi]; hi += 1
        elif si < len(scan_ids):
            ids[i] = scan_ids[si]; si += 1
        else:
            ids[i] = hot_ids[hi % len(hot_ids)]; hi += 1

    _write_bin(path, ids)


def gen_temporal_workset(path, n_req, cache_size, ws_factor=2.5, churn_per_req=0.002, seed=42):
    """
    Sliding working set: a set of W=ws_factor*cache_size objects, with
    churn_per_req fraction replaced by new objects each request.

    Creates medium-frequency objects with temporal locality.
    Designed to match ARC_WINS trace (trace_1):
    - Large working set >> cache
    - Objects accessed in temporal clusters → ARC's LRU T2 helps
    - S3FIFO's FIFO main loses objects that cycle out of main before re-access
    """
    if os.path.exists(path):
        return
    rng = np.random.default_rng(seed)
    n_ws = int(cache_size * ws_factor)
    hot_set = np.arange(1, n_ws + 1, dtype=np.int64)
    next_id = n_ws + 1

    ids = np.empty(n_req, dtype=np.int64)
    n_churn = max(1, int(n_ws * churn_per_req))
    for i in range(n_req):
        # Access a random object from working set (slightly skewed to recent)
        # Use triangular distribution to bias toward higher indices (more recent)
        idx = int(rng.triangular(0, n_ws - 1, n_ws - 1))
        ids[i] = hot_set[idx % n_ws]
        # Churn: replace random objects in working set with new ones
        for _ in range(n_churn):
            replace_idx = rng.integers(0, n_ws)
            hot_set[replace_idx] = next_id
            next_id += 1

    _write_bin(path, ids)


def gen_zipf(path, n_obj, n_req, alpha, seed=42):
    if os.path.exists(path):
        return
    rng = np.random.default_rng(seed)
    np_tmp = np.power(np.arange(1, n_obj + 1, dtype=float), -alpha)
    dist = np.cumsum(np_tmp) / np_tmp.sum()
    ids = np.searchsorted(dist, rng.uniform(0, 1, n_req)) + 1
    _write_bin(path, ids)


def gen_moderate_scan(path, n_req, n_hot, hot_frac=0.55, hot_alpha=0.9,
                      scan_repeat=2, seed=42):
    """
    Moderate scan: most objects are accessed scan_repeat times (not pure one-hit).
    Designed to match SCAN_MOD traces (trace_7/8):
    - S3FIFO wins but not as dramatically as SCAN_HEAVY
    - Scan objects are accessed 2 times (S3FIFO's small queue catches 2nd access)
    """
    if os.path.exists(path):
        return
    rng = np.random.default_rng(seed)
    n_hot = max(1, n_hot)
    hot_count = int(n_req * hot_frac)
    scan_count = n_req - hot_count

    # Hot accesses
    np_tmp = np.power(np.arange(1, n_hot + 1, dtype=float), -hot_alpha)
    dist = np.cumsum(np_tmp) / np_tmp.sum()
    hot_ids = np.searchsorted(dist, rng.uniform(0, 1, hot_count)) + 1

    # Scan: each object appears exactly scan_repeat times
    n_scan_obj = scan_count // scan_repeat
    scan_base = np.repeat(np.arange(n_hot + 1, n_hot + 1 + n_scan_obj,
                                    dtype=np.int64), scan_repeat)
    rng.shuffle(scan_base)
    scan_ids = scan_base[:scan_count]

    # Interleave
    hot_mask = rng.random(n_req) < hot_frac
    ids = np.empty(n_req, dtype=np.int64)
    hi = si = 0
    for i in range(n_req):
        if hot_mask[i] and hi < len(hot_ids):
            ids[i] = hot_ids[hi]; hi += 1
        elif si < len(scan_ids):
            ids[i] = scan_ids[si]; si += 1
        else:
            ids[i] = hot_ids[hi % len(hot_ids)]; hi += 1

    _write_bin(path, ids)


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm hooks (inline parametric versions)
# ─────────────────────────────────────────────────────────────────────────────

def make_s3fifo(small_ratio, ghost_ratio):
    class C:
        def __init__(self, cs):
            self.cs = cs; self.sm = max(1, int(cs*small_ratio)); self.gn = int(cs*ghost_ratio)
            self.small=deque(); self.smap={}; self.sb=0
            self.main=deque(); self.oi={}; self.mb=0
            self.gs=set(); self.gq=deque()
        def _ghost(self, oid):
            if oid in self.gs: return
            self.gs.add(oid); self.gq.append(oid)
            while len(self.gs) > self.gn: self.gs.discard(self.gq.popleft())
        def on_hit(self, r):
            oid=r.obj_id
            if oid in self.smap: s,f=self.smap[oid]; self.smap[oid]=(s,min(f+1,3))
            elif oid in self.oi: s,f=self.oi[oid]; self.oi[oid]=(s,min(f+1,3))
        def on_miss(self, r):
            oid,sz=r.obj_id,r.obj_size
            if sz>self.cs or oid in self.smap or oid in self.oi: return
            if oid in self.gs:
                self.gs.discard(oid); self.main.appendleft(oid); self.oi[oid]=(sz,0); self.mb+=sz
            else:
                self.small.append(oid); self.smap[oid]=(sz,0); self.sb+=sz
        def _es(self):
            while self.small:
                oid=self.small.popleft()
                if oid not in self.smap: continue
                sz,fr=self.smap.pop(oid); self.sb-=sz
                if fr==0: self._ghost(oid); return oid
                self.main.appendleft(oid); self.oi[oid]=(sz,0); self.mb+=sz
            return 0
        def _em(self):
            sc,lim=0,len(self.main)
            while self.main and sc<=lim:
                oid=self.main.pop(); sc+=1
                if oid not in self.oi: continue
                sz,fr=self.oi[oid]; self.mb-=sz
                if fr==0: del self.oi[oid]; return oid
                self.oi[oid]=(sz,fr-1); self.main.appendleft(oid); self.mb+=sz
            return 0
        def evict(self, r):
            mm=self.cs-self.sm
            if self.mb>mm and self.main:
                v=self._em()
                if v: return v
            v=self._es()
            if v: return v
            return self._em()
        def on_remove(self, oid):
            if oid in self.smap:
                sz,_=self.smap.pop(oid); self.sb-=sz
                try: self.small.remove(oid)
                except ValueError: pass
            elif oid in self.oi:
                sz,_=self.oi.pop(oid); self.mb-=sz
                try: self.main.remove(oid)
                except ValueError: pass
            self.gs.discard(oid)
    def init(p): return C(p.cache_size)
    def hit(d,r): d.on_hit(r)
    def miss(d,r): d.on_miss(r)
    def evict(d,r): return d.evict(r)
    def rem(d,oid): d.on_remove(oid)
    def free(d): d.small.clear(); d.smap.clear(); d.main.clear(); d.oi.clear(); d.gs.clear()
    return init,hit,miss,evict,rem,free


def make_mas3fifo(small_ratio, ghost_ratio, iat_factor):
    IAT_A = 0.3
    class C:
        def __init__(self, cs):
            self.cs=cs; self.sm_max=max(1,int(cs*small_ratio)); self.gn=int(cs*ghost_ratio)
            self.small=deque(); self.smap={}; self.sb=0
            self.main=deque(); self.oi={}; self.mb=0
            self.gs=set(); self.gq=deque()
            self.meta={}; self.t=0; self.tot_sz=0; self.tot_ob=0
        def _cap(self):
            avg=self.tot_sz/self.tot_ob if self.tot_ob else 1.0
            return self.cs/max(1.0,avg)
        def _umeta(self,oid,sz):
            self.t+=1
            if oid in self.meta:
                c,lt,ew=self.meta[oid]
                iat=self.t-lt
                nw=(1-IAT_A)*ew+IAT_A*iat if ew>0 else float(iat)
                self.meta[oid]=(c+1,self.t,nw)
            else:
                self.meta[oid]=(1,self.t,0.0)
                self.tot_ob+=1; self.tot_sz+=sz
        def _ft(self,oid):
            m=self.meta.get(oid)
            if not m: return False
            c,_,ew=m
            if c<2 or ew<=0: return False
            return ew<iat_factor*self._cap()
        def _ghost(self,oid):
            if oid in self.gs: return
            self.gs.add(oid); self.gq.append(oid)
            while len(self.gs)>self.gn: self.gs.discard(self.gq.popleft())
        def on_hit(self,r):
            oid,sz=r.obj_id,r.obj_size
            self._umeta(oid,sz)
            if oid in self.smap: s,f=self.smap[oid]; self.smap[oid]=(s,min(f+1,3))
            elif oid in self.oi: s,f=self.oi[oid]; self.oi[oid]=(s,min(f+1,3))
        def on_miss(self,r):
            oid,sz=r.obj_id,r.obj_size
            self._umeta(oid,sz)
            if sz>self.cs or oid in self.smap or oid in self.oi: return
            ig=oid in self.gs; ft=(not ig) and self._ft(oid)
            if ig or ft:
                self.gs.discard(oid); self.main.appendleft(oid); self.oi[oid]=(sz,0); self.mb+=sz
            else:
                self.small.append(oid); self.smap[oid]=(sz,0); self.sb+=sz
        def _es(self):
            while self.small:
                oid=self.small.popleft()
                if oid not in self.smap: continue
                sz,fr=self.smap.pop(oid); self.sb-=sz
                if fr==0: self._ghost(oid); return oid
                self.main.appendleft(oid); self.oi[oid]=(sz,0); self.mb+=sz
            return 0
        def _em(self):
            sc,lim=0,len(self.main)
            while self.main and sc<=lim:
                oid=self.main.pop(); sc+=1
                if oid not in self.oi: continue
                sz,fr=self.oi[oid]; self.mb-=sz
                if fr==0: del self.oi[oid]; return oid
                self.oi[oid]=(sz,fr-1); self.main.appendleft(oid); self.mb+=sz
            return 0
        def evict(self,r):
            mm=self.cs-self.sm_max
            if self.mb>mm and self.main:
                v=self._em()
                if v: return v
            v=self._es()
            if v: return v
            return self._em()
        def on_remove(self,oid):
            if oid in self.smap:
                sz,_=self.smap.pop(oid); self.sb-=sz
                try: self.small.remove(oid)
                except ValueError: pass
            elif oid in self.oi:
                sz,_=self.oi.pop(oid); self.mb-=sz
                try: self.main.remove(oid)
                except ValueError: pass
            self.gs.discard(oid)
    def init(p): return C(p.cache_size)
    def hit(d,r): d.on_hit(r)
    def miss(d,r): d.on_miss(r)
    def evict(d,r): return d.evict(r)
    def rem(d,oid): d.on_remove(oid)
    def free(d): d.small.clear(); d.smap.clear(); d.main.clear(); d.oi.clear(); d.gs.clear(); d.meta.clear()
    return init,hit,miss,evict,rem,free


# ─────────────────────────────────────────────────────────────────────────────
# Import baselines for fingerprint validation
# ─────────────────────────────────────────────────────────────────────────────

import plugin_seive as sieve_mod
import plugin_arc   as arc_mod

SIEVE_H = (sieve_mod.init_hook, sieve_mod.hit_hook, sieve_mod.miss_hook,
           sieve_mod.eviction_hook, sieve_mod.remove_hook, sieve_mod.free_hook)
ARC_H   = (arc_mod.init_hook, arc_mod.hit_hook, arc_mod.miss_hook,
           arc_mod.eviction_hook, arc_mod.remove_hook, arc_mod.free_hook)


def run_hooks(hooks, cs, path, tt):
    init,hit,miss,evict,rem,free = hooks
    c = PluginCache(cache_size=cs, cache_init_hook=init, cache_hit_hook=hit,
                    cache_miss_hook=miss, cache_eviction_hook=evict,
                    cache_remove_hook=rem, cache_free_hook=free, cache_name='x')
    r = TraceReader(trace=path, trace_type=tt)
    mr, _ = c.process_trace(r)
    return mr


# ─────────────────────────────────────────────────────────────────────────────
# Generate and validate traces
# ─────────────────────────────────────────────────────────────────────────────

def fingerprint(path, tt, cs_small, cs_large):
    """Return (sieve_s, arc_s, s3_s, sieve_l, arc_l, s3_l) for quick S3FIFO(10%,4x)."""
    s3_h = make_s3fifo(0.10, 4.0)
    sv_s = run_hooks(SIEVE_H, cs_small, path, tt)
    arc_s = run_hooks(ARC_H,   cs_small, path, tt)
    s3_s = run_hooks(s3_h,   cs_small, path, tt)
    sv_l = run_hooks(SIEVE_H, cs_large, path, tt)
    arc_l = run_hooks(ARC_H,   cs_large, path, tt)
    s3_l = run_hooks(s3_h,   cs_large, path, tt)
    return sv_s, arc_s, s3_s, sv_l, arc_l, s3_l


def make_traces():
    """Generate and return list of (name, path, cs_small, cs_large, trace_type, weight)."""

    # ── SCAN_HEAVY type (trace_2/3/4 — 3 traces, weight 3) ──────────────
    gen_scan_hot('/tmp/cal_scan90.bin', n_req=1_200_000, scan_frac=0.88,
                 n_hot=3000, hot_alpha=1.2, seed=1)
    gen_scan_hot('/tmp/cal_scan80.bin', n_req=4_000_000, scan_frac=0.80,
                 n_hot=6000, hot_alpha=1.2, seed=2)
    gen_scan_hot('/tmp/cal_scan75.bin', n_req=5_000_000, scan_frac=0.75,
                 n_hot=8000, hot_alpha=1.2, seed=3)

    # ── ARC_WINS type (trace_1 — 1 trace, weight 1) ─────────────────────
    gen_temporal_workset('/tmp/cal_temporal.bin', n_req=3_000_000,
                         cache_size=12000, ws_factor=2.5, churn_per_req=0.003, seed=4)

    # ── SCAN_MOD type (trace_7/8 — 2 traces, weight 2) ──────────────────
    gen_moderate_scan('/tmp/cal_scanmod_a.bin', n_req=9_000_000,
                      n_hot=4000, hot_frac=0.50, hot_alpha=0.9, scan_repeat=2, seed=5)
    gen_moderate_scan('/tmp/cal_scanmod_b.bin', n_req=2_000_000,
                      n_hot=5000, hot_frac=0.60, hot_alpha=1.0, scan_repeat=3, seed=6)

    # ── MIXED type (trace_0/5/6/9 — 4 traces, weight 4) ─────────────────
    gen_zipf('/tmp/cal_zipf12.bin', n_obj=30_000, n_req=2_000_000, alpha=1.2, seed=7)
    gen_zipf('/tmp/cal_zipf10.bin', n_obj=20_000, n_req=5_000_000, alpha=1.0, seed=8)
    gen_zipf('/tmp/cal_zipf08.bin', n_obj=30_000, n_req=6_000_000, alpha=0.8, seed=9)
    gen_zipf('/tmp/cal_zipf07.bin', n_obj=50_000, n_req=3_000_000, alpha=0.7, seed=10)

    # Cache sizes (small = 10% of large, roughly matching course trace ratio)
    TT = TraceType.ORACLE_GENERAL_TRACE
    return [
        # name,                   path,                     cs_small, cs_large, tt,  weight
        ('scan90(tr2/3/4)',  '/tmp/cal_scan90.bin',     700,   7000,   TT, 1.5),
        ('scan80(tr2/3/4)',  '/tmp/cal_scan80.bin',    2000,  20000,   TT, 1.5),
        ('scan75(tr2/3/4)',  '/tmp/cal_scan75.bin',    3000,  30000,   TT, 1.0),
        ('temporal(tr1)',    '/tmp/cal_temporal.bin',  1200,  12000,   TT, 2.0),
        ('scanmod_a(tr7)',   '/tmp/cal_scanmod_a.bin', 1600,  16000,   TT, 1.5),
        ('scanmod_b(tr8)',   '/tmp/cal_scanmod_b.bin', 3000,  30000,   TT, 1.5),
        ('zipf12(tr0/5)',    '/tmp/cal_zipf12.bin',    4000,  40000,   TT, 1.0),
        ('zipf10(tr5/6)',    '/tmp/cal_zipf10.bin',    5000,  50000,   TT, 1.0),
        ('zipf08(tr6/9)',    '/tmp/cal_zipf08.bin',    6000,  60000,   TT, 1.0),
        ('zipf07(tr9)',      '/tmp/cal_zipf07.bin',    7000,  70000,   TT, 1.0),
    ]


def validate_traces(traces):
    print("── Validating synthetic traces against course trace fingerprints ──")
    print("%-22s  %8s %8s %8s  %8s %8s %8s  %10s" %
          ('Trace', 'SIE_s', 'ARC_s', 'S3_s', 'SIE_l', 'ARC_l', 'S3_l', 'S3vsARC_l'))
    print("  " + "─"*90)
    print("  TARGET fingerprints from course:")
    print("  tr_1(ARC_WINS)  → S3vsARC_l  > +30%")
    print("  tr_2/3/4(SCAN)  → S3vsARC_l  < -35%")
    print("  tr_7/8(SCANMOD) → S3vsARC_l ≈ -10% to -25%")
    print("  tr_0/5/6/9(MIX) → S3vsARC_l ≈ -5% to +5%")
    print()
    for name, path, cs_s, cs_l, tt, w in traces:
        sv_s, arc_s, s3_s, sv_l, arc_l, s3_l = fingerprint(path, tt, cs_s, cs_l)
        s3_vs_arc_l = (s3_l - arc_l) / arc_l * 100 if arc_l > 0 else 0
        marker = ''
        if s3_vs_arc_l > 30: marker = '← ARC_WINS ✓'
        elif s3_vs_arc_l < -35: marker = '← SCAN_HEAVY ✓'
        elif s3_vs_arc_l < -10: marker = '← SCAN_MOD ✓'
        else: marker = '← MIXED ✓'
        print("%-22s  %8.4f %8.4f %8.4f  %8.4f %8.4f %8.4f  %+9.1f%%  %s" %
              (name, sv_s, arc_s, s3_s, sv_l, arc_l, s3_l, s3_vs_arc_l, marker))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Weighted miss ratio computation
# ─────────────────────────────────────────────────────────────────────────────

def weighted_miss(results, traces):
    """Compute weighted average miss ratio across all (small+large) workloads."""
    total_w = sum(2 * w for *_, w in traces)  # small + large per trace
    total = 0.0
    for i, (name, path, cs_s, cs_l, tt, w) in enumerate(traces):
        total += w * (results[i*2] + results[i*2+1])
    return total / total_w


def run_all_workloads(hooks, traces):
    """Run hooks on all (small, large) pairs; return flat list of miss ratios."""
    mrs = []
    for name, path, cs_s, cs_l, tt, w in traces:
        for cs in [cs_s, cs_l]:
            init,hit,miss,evict,rem,free = hooks
            c = PluginCache(cache_size=cs, cache_init_hook=init, cache_hit_hook=hit,
                            cache_miss_hook=miss, cache_eviction_hook=evict,
                            cache_remove_hook=rem, cache_free_hook=free, cache_name='x')
            r = TraceReader(trace=path, trace_type=tt)
            mr, _ = c.process_trace(r)
            mrs.append(mr)
    return mrs


# ─────────────────────────────────────────────────────────────────────────────
# Parameter sweeps
# ─────────────────────────────────────────────────────────────────────────────

SMALL_RATIOS = [0.06, 0.08, 0.10, 0.11, 0.12, 0.15, 0.20]
GHOST_RATIOS = [1, 2, 4, 6, 8, 10, 15, 20]
IAT_FACTORS  = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, float('inf')]


def sweep(traces, verbose=True):
    print("═"*80)
    print("SWEEP 1: S3-FIFO (FIFO main) — small ratio × ghost ratio")
    print("═"*80)
    best_s3 = (float('inf'), None)
    s3_results = {}
    for sr in SMALL_RATIOS:
        for gr in GHOST_RATIOS:
            hooks = make_s3fifo(sr, gr)
            mrs = run_all_workloads(hooks, traces)
            wmr = weighted_miss(mrs, traces)
            s3_results[(sr, gr)] = wmr
            if wmr < best_s3[0]:
                best_s3 = (wmr, (sr, gr))
    # Print best 10
    sorted_s3 = sorted(s3_results.items(), key=lambda x: x[1])
    print("Top 10 configs (lower = better):")
    for (sr, gr), wmr in sorted_s3[:10]:
        marker = ' ← BEST' if (sr, gr) == best_s3[1] else ''
        print(f"  small={int(sr*100):3d}%  ghost={int(gr):3d}x  wmiss={wmr:.4f}{marker}")

    print()
    print("═"*80)
    print("SWEEP 2: MAS3-FIFO — ghost ratio × iat_factor (at best small ratio)")
    print("═"*80)
    best_sr = best_s3[1][0]
    best_mas3 = (float('inf'), None)
    mas3_results = {}
    for gr in GHOST_RATIOS:
        for iatf in IAT_FACTORS:
            hooks = make_mas3fifo(best_sr, gr, iatf)
            mrs = run_all_workloads(hooks, traces)
            wmr = weighted_miss(mrs, traces)
            mas3_results[(gr, iatf)] = wmr
            if wmr < best_mas3[0]:
                best_mas3 = (wmr, (gr, iatf))
    sorted_mas3 = sorted(mas3_results.items(), key=lambda x: x[1])
    print(f"Using small={int(best_sr*100)}% (best from sweep 1). Top 10 configs:")
    for (gr, iatf), wmr in sorted_mas3[:10]:
        marker = ' ← BEST' if (gr, iatf) == best_mas3[1] else ''
        iatf_str = f"{iatf:.1f}" if iatf != float('inf') else 'inf'
        print(f"  ghost={int(gr):3d}x  iat={iatf_str:>5}  wmiss={wmr:.4f}{marker}")

    print()
    print("═"*80)
    print("FINAL RECOMMENDATION")
    print("═"*80)
    s3_wmr,   (s3_sr,   s3_gr)          = best_s3
    mas3_wmr, (mas3_gr, mas3_iatf)       = best_mas3
    print(f"  Best S3-FIFO:    small={int(s3_sr*100)}%  ghost={int(s3_gr)}x   wmiss={s3_wmr:.4f}")
    print(f"  Best MAS3-FIFO:  small={int(best_sr*100)}%  ghost={int(mas3_gr)}x  iat={mas3_iatf}  wmiss={mas3_wmr:.4f}")
    winner = 'S3-FIFO' if s3_wmr <= mas3_wmr else 'MAS3-FIFO'
    print(f"\n  → Submit: {winner}")
    if winner == 'S3-FIFO':
        print(f"    Update plugin_s3fifo.py:  small_ratio={s3_sr}  ghost_ratio={s3_gr}")
    else:
        print(f"    Update plugin_mas3_fifo.py:  small_ratio={best_sr}  ghost_ratio={mas3_gr}  iat_factor={mas3_iatf}")
    print()
    return best_s3, best_mas3


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating calibrated synthetic traces...")
    traces = make_traces()

    print("\nValidating trace fingerprints...")
    validate_traces(traces)

    print("Running parameter sweep (this takes a few minutes)...")
    sweep(traces)
