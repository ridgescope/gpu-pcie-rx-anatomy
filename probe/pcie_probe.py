"""Synthetic workloads for measuring the PCIe receive throughput.

One process runs a sequence of modes on one GPU, each for --seconds with --idle seconds
of sleep between them, and prints one JSON line per mode: the window, the iterations and
the bytes this process copied from host to device, which the process counts itself. A child
process polls NVML's own PCIe throughput counter (nvmlDeviceGetPcieThroughput, KB/s over
a 20 ms window) through ctypes for the whole run and writes timestamped samples; the
parent prints its mean per mode. The receive throughput that DCGM reports (field 1010) is
read afterwards over the same windows.

Usage: python pcie_probe.py --tag NAME --out DIR [--seconds 40] [--idle 20]
       python pcie_probe.py --poll UUID FILE    (the child)
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import queue
import random
import subprocess
import sys
import threading
import time


class Worker:
    """A thread that stays alive and runs the functions handed to it, one at a time."""

    def __init__(self, setup=None):
        self.q, self.r, self.setup = queue.Queue(), queue.Queue(), setup
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        if self.setup:
            self.setup()
        while True:
            fn = self.q.get()
            try:
                self.r.put((fn(), None))
            except BaseException as e:  # noqa: BLE001
                self.r.put((0, e))

    def __call__(self, fn):
        self.q.put(fn)
        v, e = self.r.get()
        if e:
            raise e
        return v


def poll(uuid: str, path: str) -> int:
    with open(path, "w", buffering=1) as f:
        try:
            lib = ctypes.CDLL("libnvidia-ml.so.1")
            rc = lib.nvmlInit_v2()
            if rc:
                raise RuntimeError(f"nvmlInit_v2 {rc}")
            h = ctypes.c_void_p()
            rc = lib.nvmlDeviceGetHandleByUUID(uuid.encode(), ctypes.byref(h)) if uuid else 1
            if rc:
                rc = lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(h))
            if rc:
                raise RuntimeError(f"no device handle {rc}")
        except Exception as e:  # noqa: BLE001
            f.write(json.dumps({"error": str(e)}) + "\n")
            return 0
        f.write(json.dumps({"ok": True, "uuid": uuid}) + "\n")
        v = ctypes.c_uint()
        while True:
            row = [time.time()]
            for counter in (1, 0):  # NVML_PCIE_UTIL_RX_BYTES = 1, TX = 0; KB/s over 20 ms
                rc = lib.nvmlDeviceGetPcieThroughput(h, counter, ctypes.byref(v))
                row.append(v.value if rc == 0 else -rc)
            f.write(f"{row[0]:.3f} {row[1]} {row[2]}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll", nargs=2)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=".")
    ap.add_argument("--seconds", type=float, default=40)
    ap.add_argument("--idle", type=float, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--modes", default="")
    ap.add_argument("--suite", default="base", choices=["base", "ops", "repeat", "vgg", "poll", "sweep", "flip", "streams", "cross"])
    ap.add_argument("--n-streams", type=int, default=16, help="suite streams: how many streams to create")
    ap.add_argument("--streams-together", action="store_true", help="suite streams: create them all at once after the warm-up, not one before each first window")
    ap.add_argument("--first-use-shuffled", action="store_true", help="suite streams: first use the streams in a shuffled order (with --streams-together)")
    ap.add_argument("--unused-streams", type=int, default=0, help="suite streams: create this many streams that are never used, before the others")
    ap.add_argument("--visit", default="", help="suite streams: run the pass on these streams only (comma-separated indices); all are still made")
    ap.add_argument("--one-round", action="store_true", help="suite streams: visit every stream once")
    ap.add_argument("--pre-window-sleep", type=float, default=0, help="sleep this long after the warm-up and the creation of the streams, before the first window")
    ap.add_argument("--empty-cache", action="store_true", help="suite streams: torch.cuda.empty_cache() after every window (PyTorch caches device memory per stream)")
    ap.add_argument("--dump-commands", default="", help="directory: after iteration --dump-at of every window, have the ioctl interposer (LD_PRELOAD, NV_IOCTL_TRACE_MAPS) write the process's mapped memory objects there")
    ap.add_argument("--dump-at", type=int, default=60)
    ap.add_argument("--l2-sem", action="store_true", help="after the windows, time the pages of the semaphore object from both L2 partitions in this process (l2_probe.py, cross mode; needs the ioctl interposer)")
    ap.add_argument("--seed", type=int, default=-1, help="seed torch before the tensors and the network are made, and print a checksum of one pass at the end")
    ap.add_argument("--first-sleep", type=float, default=0, help="sleep this long after CUDA starts, before the first cuDNN call")
    ap.add_argument("--first-thread", action="store_true", help="make the first cuDNN call (the warm-up) from another thread")
    ap.add_argument("--pre-streams", type=int, default=0, help="create this many CUDA streams (cudaStreamCreate) before the first cuDNN call")
    ap.add_argument("--no-autograd-threads", action="store_true", help="run backward in the calling thread")
    ap.add_argument("--iter-log", action="store_true", help="write the end time of every iteration to iter-<tag>.txt")
    ap.add_argument("--deterministic", action="store_true", help="torch.backends.cudnn.deterministic = True")
    ap.add_argument("--channels-last", action="store_true", help="the VGG network and its input in NHWC")
    ap.add_argument("--membind", type=int, default=-1, help="bind the process's memory to this NUMA node before CUDA starts")
    a = ap.parse_args()
    if a.poll:
        return poll(*a.poll)
    membind_rc = None
    if a.membind >= 0:  # set_mempolicy(MPOL_BIND, {node}): kernel allocations made for this process follow it
        libc = ctypes.CDLL(None, use_errno=True)
        mask = ctypes.c_ulong(1 << a.membind)
        membind_rc = libc.syscall(238, 2, ctypes.byref(mask), 65)
        if membind_rc != 0:
            membind_rc = -ctypes.get_errno()

    import torch
    import torchvision

    if a.no_autograd_threads:  # by default the backward of CUDA nodes runs on autograd's device thread
        torch.autograd.set_multithreading_enabled(False)
    if a.deterministic:
        torch.backends.cudnn.deterministic = True
    dev = torch.device(a.device)
    cuda = dev.type == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    os.makedirs(a.out, exist_ok=True)
    if a.seed >= 0:
        torch.manual_seed(a.seed)
    uuid = f"GPU-{torch.cuda.get_device_properties(0).uuid}" if cuda else ""
    poll_path = os.path.join(a.out, f"nvml-{a.tag}.txt")
    child = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--poll", uuid, poll_path]) if cuda else None

    B = 128
    host = torch.rand((B, 3, 64, 64))
    pinned = host.clone().pin_memory() if cuda else host.clone()
    x = host.to(dev)
    z = torch.ones(1, device=dev)
    conv = torch.nn.Conv2d(3, 64, 3, padding=1).to(dev)
    vgg = torchvision.models.vgg16(weights=None).features.to(dev).eval()
    for p in vgg.parameters():
        p.requires_grad_(False)
    if a.channels_last:
        vgg = vgg.to(memory_format=torch.channels_last); x = x.contiguous(memory_format=torch.channels_last)
    big = torch.rand((128, 1024, 1024), device=dev) if cuda else torch.rand((8, 64, 64))
    big2 = torch.empty_like(big)
    nb = host.numel() * host.element_size()

    def m_pageable():
        x.copy_(host); sync(); return nb
    def m_pinned():
        x.copy_(pinned); sync(); return nb
    def m_pageable_11hz():
        t = time.perf_counter(); x.copy_(host); sync()
        time.sleep(max(0.0, 1 / 11 - (time.perf_counter() - t))); return nb
    def m_launch():
        for _ in range(1000):
            z.add_(1)
        sync(); return 0
    def m_launch_sync():
        z.add_(1); z.item(); return 0
    def m_sync():
        sync(); return 0
    def m_conv():
        conv.zero_grad(set_to_none=True); conv(x).sum().backward(); sync(); return 0
    def vgg_step(n):
        inp = x[:n].clone(memory_format=torch.preserve_format).requires_grad_(True)
        vgg(inp).sum().backward(); sync(); return 0
    def m_vgg128():
        return vgg_step(128)
    def m_vgg64():
        return vgg_step(64)
    def m_dtod():
        big2.copy_(big); sync(); return 0

    modes = [("idle", None), ("pageable", m_pageable), ("pinned", m_pinned), ("pageable-11hz", m_pageable_11hz),
             ("launch", m_launch), ("launch-sync", m_launch_sync), ("sync", m_sync), ("conv", m_conv),
             ("vgg128", m_vgg128), ("vgg64", m_vgg64), ("dtod", m_dtod)]
    if a.suite == "ops":  # which operation of a convolutional pass makes the receive traffic
        import torch.nn.functional as F
        a64 = torch.rand((B, 64, 64, 64), device=dev) if cuda else torch.rand((2, 64, 8, 8))
        a256 = torch.rand((B, 256, 16, 16), device=dev) if cuda else torch.rand((2, 256, 4, 4))
        a512 = torch.rand((B, 512, 8, 8), device=dev) if cuda else torch.rand((2, 512, 4, 4))
        c64 = torch.nn.Conv2d(64, 64, 3, padding=1).to(dev)
        c256 = torch.nn.Conv2d(256, 256, 3, padding=1).to(dev)
        c512 = torch.nn.Conv2d(512, 512, 3, padding=1).to(dev)

        def fwd(net, t):
            def f():
                with torch.no_grad():
                    net(t)
                sync(); return 0
            return f

        def fb(net, t):
            def f():
                if hasattr(net, "zero_grad"):
                    net.zero_grad(set_to_none=True)
                inp = t.clone().requires_grad_(True)
                net(inp).sum().backward(); sync(); return 0
            return f

        def nocudnn(f):
            def g():
                with torch.backends.cudnn.flags(enabled=False):
                    return f()
            return g

        def bench(f):
            def g():
                with torch.backends.cudnn.flags(enabled=True, benchmark=True):
                    return f()
            return g

        def m_add():
            (a64 + 1.0); sync(); return 0
        def m_zero():
            big2.zero_(); sync(); return 0
        def m_alloc():
            t = torch.empty_like(a64); del t
            if cuda:
                torch.cuda.empty_cache()
            sync(); return 0

        pool = lambda t: F.max_pool2d(t, 2)
        relu = lambda t: F.relu(t)
        modes = [("idle", None),
                 ("vgg-fb", fb(vgg, x)), ("vgg-fwd", fwd(vgg, x)),
                 ("vgg-fb-nocudnn", nocudnn(fb(vgg, x))), ("vgg-fwd-nocudnn", nocudnn(fwd(vgg, x))),
                 ("vgg-fb-benchmark", bench(fb(vgg, x))),
                 ("c64-fwd", fwd(c64, a64)), ("c64-fb", fb(c64, a64)), ("c64-fwd-nocudnn", nocudnn(fwd(c64, a64))),
                 ("c256-fwd", fwd(c256, a256)), ("c256-fb", fb(c256, a256)),
                 ("c512-fwd", fwd(c512, a512)), ("c512-fb", fb(c512, a512)),
                 ("relu-fb", fb(relu, a64)), ("pool-fb", fb(pool, a64)),
                 ("add", m_add), ("zero", m_zero), ("alloc-free", m_alloc)]
    if a.suite == "poll":  # does a copy queued behind running compute make the receive traffic, by its wait
        n = 8192 if cuda else 64
        A = torch.rand((n, n), device=dev)
        small = torch.rand((512, 512), device=dev); small2 = torch.empty_like(small)

        def gemm(k=1):
            for _ in range(k):
                torch.mm(A, A)
        def m_gemm():
            gemm(); sync(); return 0
        def m_gemm_dtod():
            gemm(); big2.copy_(big); sync(); return 0
        def m_gemm_sync_dtod():
            gemm(); sync(); big2.copy_(big); sync(); return 0
        def m_gemm2_dtod():
            gemm(2); big2.copy_(big); sync(); return 0
        def m_gemm_dtod_small():
            gemm(); small2.copy_(small); sync(); return 0
        def m_dtod_gemm():
            big2.copy_(big); gemm(); sync(); return 0
        def m_gemm_fill():
            gemm(); big2.fill_(1.0); sync(); return 0

        def flags(f, **kw):
            def g():
                with torch.backends.cudnn.flags(enabled=True, **kw):
                    return f()
            return g
        v128 = lambda: vgg_step(128)
        modes = [("idle", None), ("gemm", m_gemm), ("gemm-dtod", m_gemm_dtod), ("gemm-sync-dtod", m_gemm_sync_dtod),
                 ("gemm2-dtod", m_gemm2_dtod), ("gemm-dtod-small", m_gemm_dtod_small), ("dtod-gemm", m_dtod_gemm),
                 ("gemm-fill", m_gemm_fill), ("vgg-fb", v128),
                 ("vgg-fb-bench-fp32", flags(v128, benchmark=True, allow_tf32=False)),
                 ("vgg-fb-tf32", flags(v128, benchmark=False, allow_tf32=True)),
                 ("vgg-fb-bench-tf32", flags(v128, benchmark=True, allow_tf32=True))]
    if a.suite == "sweep":  # how the receive traffic of one backward convolution scales
        def conv_fb(ch, bsz, hw):
            c = torch.nn.Conv2d(ch, ch, 3, padding=1).to(dev)
            t = torch.rand((bsz, ch, hw, hw), device=dev)
            def f():
                c.zero_grad(set_to_none=True)
                inp = t.clone().requires_grad_(True)
                c(inp).sum().backward(); sync(); return 0
            return f
        grid = [(64, 8, 64), (64, 32, 64), (64, 128, 64), (64, 128, 16), (64, 128, 32), (64, 32, 128),
                (16, 128, 64), (32, 128, 64), (128, 128, 64), (256, 128, 64)] if cuda else [(8, 2, 8)]
        modes = [("idle", None)] + [(f"c{ch}-b{bsz}-s{hw}", conv_fb(ch, bsz, hw)) for ch, bsz, hw in grid] + [("vgg-1", lambda: vgg_step(128))]
    if a.suite == "vgg":  # one window of the pass, for many short processes under cuDNN's logging
        modes = [("idle", None), ("vgg-1", lambda: vgg_step(128))]
    if a.suite == "flip":  # can the rate move inside a live process
        workers, streams = {}, {}
        v = lambda: vgg_step(128)
        def in_thread(key):  # the whole iteration on another host thread: its own cuDNN handle
            def f():
                if key not in workers:
                    workers[key] = Worker()
                return workers[key](v)
            return f
        def in_stream(key):  # the iteration on an explicit stream of PyTorch's pool
            def f():
                if not cuda:
                    return v()
                if key not in streams:
                    streams[key] = torch.cuda.Stream()
                with torch.cuda.stream(streams[key]):
                    inp = x.clone(memory_format=torch.preserve_format).requires_grad_(True)
                    vgg(inp).sum().backward()
                sync(); return 0
            return f
        def calling_thread():  # backward in the calling thread, for this iteration only
            torch.autograd.set_multithreading_enabled(False)
            try:
                return vgg_step(128)
            finally:
                torch.autograd.set_multithreading_enabled(True)
        modes = [("v-1", v), ("idle", None), ("v-2", v), ("thread-a", in_thread("a")), ("v-3", v),
                 ("thread-b", in_thread("b")), ("v-4", v), ("stream-1", in_stream(1)), ("v-5", v),
                 ("stream-2", in_stream(2)), ("v-6", v), ("calling", calling_thread), ("v-7", v),
                 ("thread-a2", in_thread("a")), ("v-8", v), ("stream-1b", in_stream(1)), ("stream-2b", in_stream(2)),
                 ("thread-b2", in_thread("b")), ("calling-2", calling_thread), ("v-9", v)]
    post_warm = None
    def on_stream(get):  # the whole iteration under one stream; get() returns it, or None for the default
        def f():
            s = get() if cuda else None
            if s is None:
                return vgg_step(128)
            with torch.cuda.stream(s):
                inp = x.clone(memory_format=torch.preserve_format).requires_grad_(True)
                vgg(inp).sum().backward()
            sync(); return 0
        return f
    after_window = {}
    if a.suite == "streams":  # does each stream carry a rate of its own
        ext = {}
        rt = ctypes.CDLL("libcudart.so.12") if cuda else None
        def raw_stream():
            h = ctypes.c_void_p()
            rc = rt.cudaStreamCreateWithFlags(ctypes.byref(h), ctypes.c_uint(1))  # cudaStreamNonBlocking
            if rc != 0:
                raise RuntimeError(f"cudaStreamCreateWithFlags {rc}")
            sid = ctypes.c_ulonglong(0)
            try:
                rc2 = rt.cudaStreamGetId(h, ctypes.byref(sid))
            except Exception:  # noqa: BLE001
                rc2 = -1
            return h, (sid.value if rc2 == 0 else None)
        def make(i):  # one stream outside PyTorch's pool, wrapped for torch
            if i not in ext and cuda:
                h, sid = raw_stream()
                ext[i] = torch.cuda.ExternalStream(h.value, device=dev)
                print(json.dumps({"probe": "stream", "tag": a.tag, "i": i, "handle": hex(h.value), "cuda_id": sid,
                                  "t": round(time.time(), 4), "ns": time.monotonic_ns()}), flush=True)
            return ext.get(i)
        unused = []
        def together():
            for k in range(a.unused_streams if cuda else 0):
                h, sid = raw_stream(); unused.append(h)
                print(json.dumps({"probe": "stream", "tag": a.tag, "i": -1 - k, "handle": hex(h.value), "cuda_id": sid,
                                  "t": round(time.time(), 4), "ns": time.monotonic_ns()}), flush=True)
            if a.streams_together:
                for i in range(a.n_streams):
                    make(i)
        post_warm = together
        rnd = random.Random(os.getpid())
        first = [int(i) for i in a.visit.split(",")] if a.visit else list(range(a.n_streams))
        if a.first_use_shuffled:
            rnd.shuffle(first)
        second = list(first); rnd.shuffle(second)
        print(json.dumps({"probe": "order", "tag": a.tag, "first": first, "second": second}), flush=True)
        v = lambda: vgg_step(128)
        modes = [("v-0", v)]
        for rnd_no, order in enumerate((first,) if a.one_round else (first, second)):
            for n, i in enumerate(order):
                modes.append((f"x{i:02d}" + ("b" if rnd_no else ""), on_stream(lambda i=i: make(i))))
                if n % 8 == 7:
                    modes.append((f"v-{len([m for m in modes if m[0].startswith('v-')])}", v))
        if a.empty_cache and cuda:
            for name, _ in modes:
                after_window[name] = torch.cuda.empty_cache
    if a.suite == "cross":  # does the value stay with the producer stream under other partner threads (experiment 1)
        pool = {}
        def producer(k):
            if k == "d":
                return None
            if k not in pool:
                pool[k] = torch.cuda.Stream()
            return pool[k]
        no_mt = lambda: torch.autograd.set_multithreading_enabled(False)  # thread-local: backward in the calling thread
        hosts = {"a": Worker(no_mt), "b": Worker(no_mt), "c": Worker(no_mt)}
        def cell(k, w):
            f = on_stream(lambda: producer(k))
            if w == "m":
                def g():
                    torch.autograd.set_multithreading_enabled(False)
                    try:
                        return f()
                    finally:
                        torch.autograd.set_multithreading_enabled(True)
                return g
            return lambda: hosts[w](f)
        cells = [(k, w) for k in ("d", "p1", "p2") for w in ("m", "a", "b", "c")]
        again = cells[:]; random.Random(os.getpid()).shuffle(again)
        modes = ([("v-1", lambda: vgg_step(128))] + [(f"{k}-{w}", cell(k, w)) for k, w in cells]
                 + [("v-2", lambda: vgg_step(128))] + [(f"{k}-{w}-2", cell(k, w)) for k, w in again]
                 + [("v-3", lambda: vgg_step(128))])
    if a.suite == "repeat":  # is the level of the VGG pass a state of the process, and what moves it
        vfb = lambda: vgg_step(128)
        modes = [("idle", None), ("vgg-1", vfb), ("pageable", m_pageable), ("vgg-2", vfb), ("pinned", m_pinned),
                 ("vgg-3", vfb), ("launch", m_launch), ("vgg-4", vfb), ("dtod", m_dtod), ("vgg-5", vfb)]
    if a.modes:
        keep = a.modes.split(",")
        modes = [m for m in modes if m[0] in keep]
    if cuda:  # where the process sits: CPUs, memory nodes, the GPU's NUMA node, the device memory
        facts = {"probe": "facts", "tag": a.tag, "suite": a.suite, "pid": os.getpid(),
                 "cpus": sorted(os.sched_getaffinity(0)), "mem_get_info": list(torch.cuda.mem_get_info()),
                 "reserved": torch.cuda.memory_reserved(), "x_ptr": x.data_ptr(), "host_ptr": host.data_ptr(),
                 "cudnn": torch.backends.cudnn.version(), "cudnn_benchmark": torch.backends.cudnn.benchmark,
                 "env": {k: v for k, v in os.environ.items() if k.startswith(("CUDNN_", "CUDA_", "PYTORCH_", "SLURM_CPU", "SLURM_STEP_ID"))}}
        try:
            for line in open("/proc/self/status"):
                if line.startswith(("Cpus_allowed_list", "Mems_allowed_list")):
                    facts[line.split(":")[0]] = line.split(":")[1].strip()
            pr = torch.cuda.get_device_properties(0)
            bdf = f"{pr.pci_domain_id:04x}:{pr.pci_bus_id:02x}:{pr.pci_device_id:02x}.0"
            facts["gpu_bdf"] = bdf
            facts["gpu_numa_node"] = open(f"/sys/bus/pci/devices/{bdf}/numa_node").read().strip()
            facts["node_cpulists"] = {n: open(f"/sys/devices/system/node/{n}/cpulist").read().strip()
                                      for n in sorted(os.listdir("/sys/devices/system/node")) if n.startswith("node")}
            facts["numa_host_pages"] = [l.strip()[:160] for l in open("/proc/self/numa_maps") if f"{host.data_ptr():x}"[:9] in l][:2]
        except Exception as e:  # noqa: BLE001
            facts["facts_error"] = str(e)
        print(json.dumps(facts), flush=True)
    pre = []
    if cuda and a.pre_streams:  # streams of our own, created before cuDNN creates its helper streams
        try:
            rt = ctypes.CDLL("libcudart.so.12")
            for _ in range(a.pre_streams):
                h = ctypes.c_void_p()
                pre.append((rt.cudaStreamCreate(ctypes.byref(h)), h))
            print(json.dumps({"probe": "pre-streams", "tag": a.tag, "rc": [rc for rc, _ in pre]}), flush=True)
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"probe": "pre-streams", "tag": a.tag, "error": str(e)}), flush=True)
    if a.first_sleep:
        time.sleep(a.first_sleep)
    def warm():  # every mode once before any window; the flip suite warms its baseline only
        for name, fn in (modes[:1] if a.suite in ("flip", "streams", "cross") else modes):
            if fn:
                fn()
        if post_warm:
            post_warm()
    if a.first_thread:
        Worker()(lambda: (warm(), 0)[1])
    else:
        warm()
    sync()
    if a.pre_window_sleep:
        print(json.dumps({"probe": "pre-window-sleep", "tag": a.tag, "from": round(time.time(), 3), "seconds": a.pre_window_sleep}), flush=True)
        time.sleep(a.pre_window_sleep)
    time.sleep(a.idle)
    windows = []; iters = []
    for name, fn in modes:
        start = time.time(); n = 0; copied = 0
        st0 = torch.cuda.memory_stats() if cuda else {}
        if fn is None:
            time.sleep(a.seconds)
        else:
            while time.time() - start < a.seconds:
                copied += fn(); n += 1
                if a.dump_commands and n == a.dump_at and not (a.suite == "streams" and name.endswith("b")):
                    d = os.path.join(a.dump_commands, f"{a.tag}-{name}"); os.makedirs(d, exist_ok=True)
                    lib = ctypes.CDLL(os.environ["LD_PRELOAD"].split(":")[0])
                    print(json.dumps({"probe": "dump", "tag": a.tag, "mode": name, "dir": d, "t": round(time.time(), 4),
                                      "objects": lib.nv_trace_dump(d.encode())}), flush=True)
                if a.iter_log:
                    iters.append((name, time.time()))
        sync()
        end = time.time()
        if fn is not None and name in after_window:
            after_window[name]()
        rec = {"probe": "mode", "tag": a.tag, "mode": name, "start": round(start, 3), "end": round(end, 3),
               "iterations": n, "h2d_bytes": copied, "h2d_MBps": round(copied / (end - start) / 1e6, 2),
               "per_iteration_ms": round((end - start) / n * 1000, 3) if n else None}
        if cuda:  # cudaMalloc churn: segments the caching allocator took from the driver in the window
            st1 = torch.cuda.memory_stats()
            rec["segments_allocated"] = st1.get("segment.all.allocated", 0) - st0.get("segment.all.allocated", 0)
            rec["reserved_GB"] = round(torch.cuda.memory_reserved() / 1e9, 2)
        windows.append(rec)
        print(json.dumps(rec), flush=True)
        time.sleep(a.idle)
    if a.seed >= 0:  # the same numbers must come out whatever was changed underneath
        inp = x.clone(memory_format=torch.preserve_format).requires_grad_(True)
        loss = vgg(inp).sum(); loss.backward(); sync()
        print(json.dumps({"probe": "check", "tag": a.tag, "loss": loss.item(), "grad_abs_sum": inp.grad.abs().sum().item(),
                          "grad_sum": inp.grad.double().sum().item()}), flush=True)
    if a.iter_log:
        with open(os.path.join(a.out, f"iter-{a.tag}.txt"), "w") as f:
            f.writelines(f"{m} {t:.4f}\n" for m, t in iters)
    try:  # where the process's memory sits at the end: pages per NUMA node, by kind of mapping
        tot = {}; kinds = {}
        for line in open("/proc/self/numa_maps"):
            f = line.split()
            kind = "nvidia" if "nvidia" in line else "file" if "file=" in line else "heap" if "heap" in f else "stack" if "stack" in f else "anon"
            for tok in f:
                if tok[0] == "N" and "=" in tok and tok[1:tok.index("=")].isdigit():
                    n, v = tok.split("="); tot[n] = tot.get(n, 0) + int(v)
                    kinds.setdefault(kind, {}); kinds[kind][n] = kinds[kind].get(n, 0) + int(v)
        free = {n: open(f"/sys/devices/system/node/{n}/meminfo").read().split("MemFree:")[1].split()[0]
                for n in sorted(os.listdir("/sys/devices/system/node")) if n.startswith("node")}
        print(json.dumps({"probe": "numa", "tag": a.tag, "membind": a.membind, "membind_rc": membind_rc,
                          "pages": tot, "by_kind": kinds, "node_free_kB": free,
                          "cpu_now": os.sched_getcpu() if hasattr(os, "sched_getcpu") else None}), flush=True)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"probe": "numa", "tag": a.tag, "error": str(e), "membind_rc": membind_rc}), flush=True)
    if a.l2_sem:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import l2_probe
        l2_probe.main(["--out", a.out, "--tag", a.tag, "--object", "sem", "--cross", "0,4",
                       "--time-sms", "0,1,2,3,4,5,6,7", "--reps", "5"])
    if child:
        child.terminate(); child.wait()
        rows = []
        for line in open(poll_path):
            p = line.split()
            if len(p) == 3 and not line.startswith("{"):
                rows.append((float(p[0]), int(p[1]), int(p[2])))
            elif line.startswith("{"):
                print(json.dumps({"probe": "nvml", "tag": a.tag, **json.loads(line)}), flush=True)
        for w in windows:
            v = [(r, t) for ts, r, t in rows if w["start"] + 1 <= ts <= w["end"] - 1 and r >= 0 and t >= 0]
            if v:
                print(json.dumps({"probe": "nvml-mode", "tag": a.tag, "mode": w["mode"], "samples": len(v),
                                  "rx_MBps": round(sum(r for r, _ in v) / len(v) / 1e3, 2),
                                  "tx_MBps": round(sum(t for _, t in v) / len(v) / 1e3, 2),
                                  "rx_max_MBps": round(max(r for r, _ in v) / 1e3, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
