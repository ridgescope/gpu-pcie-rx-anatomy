"""PCIe receive traffic from CUDA streams that wait, without cuDNN.

One stream (the producer) runs one large matrix product. Three other streams wait for the
end of that product with wait_event and have N small kernels queued behind the wait; each
small kernel adds 1 to a tensor of one element. While the product runs, the GPU receives
gigabytes a second over PCIe although the process copies nothing to it. The script measures
the receive throughput against N.

N changes from wait to wait: the list --kernels is gone through --passes times, in a new
random order every time. A child process polls NVML's PCIe throughput counter
(nvmlDeviceGetPcieThroughput, KB/s over a 20 ms window) and writes timestamped samples. For
each wait the parent takes the median of the samples that fall inside it, --margin seconds
away from its ends, and prints the mean of these medians for every N. The counter belongs to
the GPU, not to the process: run the script on a GPU that does nothing else.

On the H100 we measured, the level of the traffic depends on the physical page that holds
the producer stream's semaphores, so separate runs of this script can print different
curves, and so can different producers of one run (--producers).

Usage: python wait_probe.py [--kernels 1,2,3,4,6,8,10,14,20,30] [--passes 20]
                            [--producers 0] [--waiters 3] [--size 22000] [--out DIR]
       python wait_probe.py --poll UUID FILE    (the child)

Needs PyTorch with CUDA and libnvidia-ml.so.1. No privileges.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import tempfile
import time


def poll(uuid: str, path: str) -> int:
    with open(path, "w", buffering=1) as f:
        lib = ctypes.CDLL("libnvidia-ml.so.1")
        rc = lib.nvmlInit_v2()
        if rc:
            f.write(json.dumps({"error": f"nvmlInit_v2 {rc}"}) + "\n")
            return 1
        h = ctypes.c_void_p()
        rc = lib.nvmlDeviceGetHandleByUUID(uuid.encode(), ctypes.byref(h))
        if rc:
            f.write(json.dumps({"error": f"nvmlDeviceGetHandleByUUID {rc}"}) + "\n")
            return 1
        f.write(json.dumps({"ok": True, "uuid": uuid}) + "\n")
        v = ctypes.c_uint()
        while True:
            row = [time.time()]
            for counter in (1, 0):  # NVML_PCIE_UTIL_RX_BYTES = 1, NVML_PCIE_UTIL_TX_BYTES = 0
                rc = lib.nvmlDeviceGetPcieThroughput(h, counter, ctypes.byref(v))
                row.append(v.value if rc == 0 else -1)
            f.write(f"{row[0]:.3f} {row[1]} {row[2]}\n")


def read_samples(path: str) -> list[tuple[float, int, int]]:
    out = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) == 3 and not line.startswith("{"):
                t, rx, tx = float(p[0]), int(p[1]), int(p[2])
                if rx >= 0 and tx >= 0:
                    out.append((t, rx, tx))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poll", nargs=2, metavar=("UUID", "FILE"), help=argparse.SUPPRESS)
    ap.add_argument("--kernels", default="1,2,3,4,6,8,10,14,20,30", help="numbers of small kernels behind the wait")
    ap.add_argument("--passes", type=int, default=20, help="times the list of --kernels is gone through for every producer")
    ap.add_argument("--producers", default="0", help="which of the created streams run the product, in turn (0 to 27)")
    ap.add_argument("--waiters", type=int, default=3, help="streams that wait; 0 measures the producer alone")
    ap.add_argument("--size", type=int, default=22000, help="side of the square matrices of the product")
    ap.add_argument("--margin", type=float, default=0.03, help="seconds dropped at both ends of a wait")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="directory for the NVML samples and the per-wait records (default: a temporary one)")
    a = ap.parse_args()
    if a.poll:
        return poll(*a.poll)

    import torch

    ns = [int(x) for x in a.kernels.split(",")]
    producers = [int(x) for x in a.producers.split(",")]
    assert ns and min(ns) >= 1 and len(set(ns)) == len(ns), "--kernels: distinct numbers from 1 up"
    assert producers and min(producers) >= 0 and max(producers) <= 27, "--producers: indexes from 0 to 27"
    assert 0 <= a.waiters <= 4 and a.passes >= 1 and a.size >= 1 and a.margin >= 0
    rnd = random.Random(a.seed)
    dev = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    out_dir = a.out or tempfile.mkdtemp(prefix="wait-probe-")
    os.makedirs(out_dir, exist_ok=True)

    # Streams. PyTorch hands out up to 32 streams of its pool; the first 28 are candidates for
    # the producer and the following ones wait.
    streams = [torch.cuda.Stream(device=dev) for _ in range(28)]
    waiters = [torch.cuda.Stream(device=dev) for _ in range(a.waiters)]
    x = torch.randn(a.size, a.size, device=dev)
    y = torch.randn(a.size, a.size, device=dev)
    z = torch.empty(a.size, a.size, device=dev)
    cells = [torch.zeros(1, device=dev) for _ in waiters]
    torch.cuda.synchronize()   # the matrices are filled on the default stream; the products run on others

    # Warm up: one product on every producer and one small kernel on every waiter, so that no
    # measured wait contains a first use.
    for p in producers:
        with torch.cuda.stream(streams[p]):
            torch.mm(x, y, out=z)
        torch.cuda.synchronize()
    for w, c in zip(waiters, cells):
        with torch.cuda.stream(w):
            c.add_(1.0)
    torch.cuda.synchronize()

    uuid = f"GPU-{torch.cuda.get_device_properties(0).uuid}"
    poll_path = os.path.join(out_dir, "nvml.txt")
    child = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--poll", uuid, poll_path])

    def stop_child():
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()

    def on_term(signum, frame):   # so that a SIGTERM also ends the child
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, on_term)

    waits = []
    alive = False
    try:
        deadline = time.time() + 15   # samples have to be coming before the first wait
        while time.time() < deadline and len(read_samples(poll_path) if os.path.exists(poll_path) else []) < 5:
            if child.poll() is not None:
                break
            time.sleep(0.2)
        if child.poll() is not None or len(read_samples(poll_path)) < 5:
            print("the NVML poller gives no samples: " + (open(poll_path).read().strip()[:300] if os.path.exists(poll_path) else "no file"), file=sys.stderr)
            return 1
        for p in producers:
            for ps in range(a.passes):
                order = list(ns)
                rnd.shuffle(order)
                for n in order:
                    with torch.cuda.stream(streams[p]):
                        torch.mm(x, y, out=z)      # the long kernel
                        ev = torch.cuda.Event()    # no timing: the wait is on a semaphore of this stream
                        ev.record(streams[p])
                    for w, c in zip(waiters, cells):
                        w.wait_event(ev)           # the wait ...
                        with torch.cuda.stream(w):
                            for _ in range(n):     # ... and n kernels queued behind it
                                c.add_(1.0)
                    t_queued = time.time()
                    ev.synchronize()
                    t_done = time.time()
                    torch.cuda.synchronize()
                    waits.append({"producer": p, "pass": ps, "N": n, "queued": t_queued, "done": t_done})
                    time.sleep(0.05)
        time.sleep(0.2)
        alive = child.poll() is None
    finally:
        stop_child()
    if not alive:
        print("the NVML poller ended before the last wait", file=sys.stderr)
        return 1

    samples = read_samples(poll_path)
    with open(os.path.join(out_dir, "waits.jsonl"), "w") as f:
        for w in waits:
            lo, hi = w["queued"] + a.margin, w["done"] - a.margin
            rx = [s[1] for s in samples if lo <= s[0] <= hi]
            tx = [s[2] for s in samples if lo <= s[0] <= hi]
            w["samples"] = len(rx)
            w["rx_gbs"] = statistics.median(rx) / 1e6 if rx else None   # KB/s to GB/s
            w["tx_gbs"] = statistics.median(tx) / 1e6 if tx else None
            f.write(json.dumps(w) + "\n")
    empty = sum(w["samples"] == 0 for w in waits)
    if empty > len(waits) // 10:
        print(f"{empty} of {len(waits)} waits have no NVML sample inside them: the product is too short for --margin, or the poller stalled", file=sys.stderr)
        return 1

    product = sorted(w["done"] - w["queued"] for w in waits)[len(waits) // 2]
    print(f"{torch.cuda.get_device_name(0)}; a wait lasts {product * 1e3:.0f} ms (size {a.size}); "
          f"{a.waiters} waiting streams; records in {out_dir}")
    print("receive throughput in GB/s, mean over the waits of each wait's median sample (number of waits)")
    print("producer " + " ".join(f"N={n:<9d}" for n in ns))
    for p in producers:
        cols = []
        for n in ns:
            v = [w["rx_gbs"] for w in waits if w["producer"] == p and w["N"] == n and w["rx_gbs"] is not None]
            cols.append(f"{sum(v) / len(v):5.2f} ({len(v):2d})  " if v else "    - ( 0)  ")
        print(f"{p:8d} " + " ".join(cols))
    return 0


if __name__ == "__main__":
    sys.exit(main())
