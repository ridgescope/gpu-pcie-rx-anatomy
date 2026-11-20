#!/usr/bin/env python3
"""A probe of the GPU's L2 partition bit for device memory.

For every SM of the card and every 4 KiB page of a few 2 MiB-aligned regions of device
memory, it times a chain of the page's 32 lines of 128 B, each read once with L2 emptied
before (a miss) or, with --no-flush, after an earlier read (a hit). The output is the array
cycles[region, sm, rep, page] with the facts beside it.

The probe requires PyTorch and NumPy. NVRTC, which ships with the PyTorch wheels,
compiles the kernel, and the CUDA driver API loads and launches it in PyTorch's context.
Both are called through ctypes.

Usage: python l2_probe.py --out DIR [--regions 4] [--reps 5] [--flush-mb 512] [--no-flush]
                          [--max-sms N] [--seed 0] [--tag T]
                          [--object sem|pb] [--cross W1,W2 --time-sms S1,S2,...]

--object reads, in place of the probe's own device memory, the first 2 MiB of an object
libcuda made at context creation: "sem", the GPU-cached system-memory object that holds the
semaphores, or "pb", the pushbuffer object. Its GPU address is taken from the log of the
ioctl interposer (ioctl-trace, NV_IOCTL_TRACE_MAPS=1), which must be loaded. The probe only reads the object.
--cross: after the sweep a page is read from SM W (the warming read) and then timed from SM
S; the output is cycles[w, s, rep, page]. It tells a line cached in the reader's partition
from one cached in the other where a miss would cross the PCIe link.
"""
import argparse, ctypes, glob, json, os, random, sys, time

import numpy as np
import torch

SRC = r"""
extern "C" __global__ void l2probe(unsigned char* base, const unsigned int* porder,
                                   const unsigned int* lorder, int npages, int nlines,
                                   unsigned int target, unsigned int* flag,
                                   unsigned long long* out, unsigned int* smids,
                                   unsigned int zmask)
{
    unsigned int sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    smids[blockIdx.x] = sm;
    if (sm != target) return;
    if (atomicCAS(flag, 0u, 1u) != 0u) return;
    unsigned int drain = 0;
    unsigned int v = 0;   /* zmask is 0 at run time: the next address waits for the last value, whatever it is */
    /* one read for the TLB; it warms one line of page 0 */
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(base));
    for (int p = 0; p < npages; p++) {
        unsigned char* pg = base + ((unsigned long long)porder[p] << 12);
        unsigned long long t0, t1;
        asm volatile("and.b32 %0, %1, %2;" : "=r"(drain) : "r"(v), "r"(zmask));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        for (int i = 0; i < nlines; i++) {
            const unsigned char* a = pg + ((unsigned long long)lorder[i] << 7) + (v & zmask);
            asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(a));
        }
        /* an instruction that needs the last value, so that the timestamp is not taken
           before the last load has returned */
        asm volatile("and.b32 %0, %1, %2;" : "=r"(drain) : "r"(v), "r"(zmask));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        out[porder[p]] = t1 - t0 + drain;
    }
}
"""


def find(pattern):
    for d in sys.path:
        hits = sorted(glob.glob(os.path.join(d, pattern)))
        if hits:
            return hits[-1]
    raise SystemExit("not found: " + pattern)


def compile_ptx():
    rtc = ctypes.CDLL(find("nvidia/cuda_nvrtc/lib/libnvrtc.so*"))
    prog = ctypes.c_void_p()
    assert rtc.nvrtcCreateProgram(ctypes.byref(prog), SRC.encode(), b"l2probe.cu", 0, None, None) == 0
    rc = rtc.nvrtcCompileProgram(prog, 0, None)
    n = ctypes.c_size_t()
    rtc.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
    log = ctypes.create_string_buffer(n.value)
    rtc.nvrtcGetProgramLog(prog, log)
    if rc != 0:
        raise SystemExit("nvrtc: " + log.value.decode(errors="replace"))
    rtc.nvrtcGetPTXSize(prog, ctypes.byref(n))
    ptx = ctypes.create_string_buffer(n.value)
    assert rtc.nvrtcGetPTX(prog, ptx) == 0
    return ptx.raw


def object_base(kind):
    d, tag = os.environ.get("NV_IOCTL_TRACE_DIR"), os.environ.get("NV_IOCTL_TRACE_TAG", "")
    path = os.path.join(d or ".", "ioctl-%s-%d.log" % (tag, os.getpid()))
    want = {"sem": ("attr2=5 ", "size=200000"), "pb": ("attr2=9 ", "attr=4a800000")}[kind]
    handle, base, size, cpu = None, None, None, None
    for line in open(path, errors="replace"):
        f = dict(t.split("=", 1) for t in line.split() if "=" in t)
        if line.startswith("A<") and f.get("class") == "3e" and f.get("type") == "0" and all(w in line + " " for w in want) and handle is None:
            handle, size = f["hNew"], int(f["size"], 16)
        if line.startswith("U<") and " EXTERNAL " in line and handle and f.get("hMemory") == handle:
            base = int(f["base"], 16)
        if line.startswith("M<") and handle and f.get("hMemory") == handle and cpu is None:
            cpu = int(f["addr"], 16)
    if base is None:
        raise SystemExit("l2_probe: no GPU address for the %s object in %s" % (kind, path))
    return base, size, handle, cpu


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="l2")
    ap.add_argument("--regions", type=int, default=4)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--flush-mb", type=int, default=512)
    ap.add_argument("--no-flush", action="store_true")
    ap.add_argument("--max-sms", type=int, default=0)
    ap.add_argument("--blocks", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--object", choices=["sem", "pb"])
    ap.add_argument("--cross")
    ap.add_argument("--time-sms")
    ap.add_argument("--hold", type=float, default=0, help="seconds to stay alive after the measurement")
    ap.add_argument("--lines", type=int, default=32, help="lines of 128 B read per page (32 is the whole page)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    rnd = random.Random(a.seed)
    dev = torch.device("cuda")
    NP, NL, R2 = 512, a.lines, 2 << 20

    mem = torch.zeros((a.regions + 1) * R2, dtype=torch.uint8, device=dev)
    base0 = (mem.data_ptr() + R2 - 1) // R2 * R2
    obj = None
    if a.object:
        torch.cuda.synchronize()
        base0, size, handle, cpu = object_base(a.object)
        obj = {"kind": a.object, "size": size, "handle": handle, "gpu_addr": hex(base0), "cpu_addr": hex(cpu) if cpu else None}
        a.regions = 1
    big = torch.zeros(a.flush_mb << 18, dtype=torch.float32, device=dev)
    porder = torch.zeros(NP, dtype=torch.int32, device=dev)
    lorder = torch.zeros(32, dtype=torch.int32, device=dev)
    flag = torch.zeros(1, dtype=torch.int32, device=dev)
    out = torch.zeros(NP, dtype=torch.int64, device=dev)
    smids = torch.full((a.blocks,), -1, dtype=torch.int32, device=dev)
    torch.cuda.synchronize()

    cu = ctypes.CDLL("libcuda.so.1")
    mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
    ptx = compile_ptx()
    rc = cu.cuModuleLoadData(ctypes.byref(mod), ptx)
    assert rc == 0, "cuModuleLoadData %d" % rc
    assert cu.cuModuleGetFunction(ctypes.byref(fn), mod, b"l2probe") == 0

    def launch(base, target):
        vals = [ctypes.c_void_p(base), ctypes.c_void_p(porder.data_ptr()), ctypes.c_void_p(lorder.data_ptr()),
                ctypes.c_int(NP), ctypes.c_int(NL), ctypes.c_uint(target), ctypes.c_void_p(flag.data_ptr()),
                ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(smids.data_ptr()), ctypes.c_uint(0)]
        arr = (ctypes.c_void_p * len(vals))(*[ctypes.cast(ctypes.pointer(v), ctypes.c_void_p) for v in vals])
        rc = cu.cuLaunchKernel(fn, a.blocks, 1, 1, 1, 1, 1, 0, None, arr, None)
        assert rc == 0, "cuLaunchKernel %d" % rc
        torch.cuda.synchronize()

    def set_orders():
        po = list(range(NP)); rnd.shuffle(po)
        lo = list(range(32)); rnd.shuffle(lo)
        porder.copy_(torch.tensor(po, dtype=torch.int32)); lorder.copy_(torch.tensor(lo, dtype=torch.int32))

    set_orders()
    launch(base0, 0xFFFFFFFF)                       # no SM matches: collects the SM ids only
    sms = sorted(set(int(x) for x in smids.cpu().tolist() if x >= 0))
    if a.max_sms:
        sms = sms[:: max(1, len(sms) // a.max_sms)][: a.max_sms]
    facts = {"probe": "l2-facts", "tag": a.tag, "device": torch.cuda.get_device_name(0), "sms": sms,
             "base": hex(base0), "regions": a.regions, "reps": a.reps, "flush_mb": 0 if a.no_flush else a.flush_mb,
             "torch": torch.__version__, "seed": a.seed, "start": time.time(), "object": obj,
             "cross": a.cross, "time_sms": a.time_sms, "pid": os.getpid()}
    print(json.dumps(facts), flush=True)

    missed = 0
    if a.cross:
        ws = [int(x) for x in a.cross.split(",")]
        ts = [int(x) for x in a.time_sms.split(",")]
        cyc = np.zeros((len(ws), len(ts), a.reps, NP), dtype=np.float32)
        for rep in range(a.reps):
            for wi, w in enumerate(ws):
                for ti in rnd.sample(range(len(ts)), len(ts)):
                    for attempt in range(5):
                        set_orders()
                        big.sum(); torch.cuda.synchronize()
                        flag.zero_(); launch(base0, w)
                        ok = int(flag.item()) == 1
                        flag.zero_(); out.zero_(); launch(base0, ts[ti])
                        if ok and int(flag.item()) == 1:
                            break
                    else:
                        missed += 1
                    cyc[wi, ti, rep] = out.cpu().numpy().astype(np.float32) / NL
            print(json.dumps({"probe": "l2-rep", "tag": a.tag, "rep": rep, "t": time.time(), "missed": missed,
                              "median_cycles_per_load": float(np.median(cyc[:, :, rep]))}), flush=True)
        np.save(os.path.join(a.out, "l2-%s.npy" % a.tag), cyc)
        json.dump(facts, open(os.path.join(a.out, "l2-%s.json" % a.tag), "w"))
        time.sleep(a.hold)
        return
    cyc = np.zeros((a.regions, len(sms), a.reps, NP), dtype=np.float32)
    for rep in range(a.reps):
        for r in range(a.regions):
            base = base0 + r * R2
            for si in rnd.sample(range(len(sms)), len(sms)):
                for attempt in range(5):
                    set_orders()
                    flag.zero_(); out.zero_()
                    if a.no_flush:
                        launch(base, sms[si])       # the read that warms; timed again below
                        flag.zero_(); out.zero_()
                    else:
                        big.sum(); torch.cuda.synchronize()
                    launch(base, sms[si])
                    if int(flag.item()) == 1:
                        break
                else:
                    missed += 1
                cyc[r, si, rep] = out.cpu().numpy().astype(np.float32) / NL
        print(json.dumps({"probe": "l2-rep", "tag": a.tag, "rep": rep, "t": time.time(), "missed": missed,
                          "median_cycles_per_load": float(np.median(cyc[:, :, rep]))}), flush=True)
    np.save(os.path.join(a.out, "l2-%s.npy" % a.tag), cyc)
    json.dump(facts, open(os.path.join(a.out, "l2-%s.json" % a.tag), "w"))
    time.sleep(a.hold)


if __name__ == "__main__":
    main()
