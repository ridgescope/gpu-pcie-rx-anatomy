#!/usr/bin/env python3
"""Decode the command buffers a process dumped with ioctl-trace (nv_trace_dump).

Usage: analysis/pcie-cmd.py IOCTL_LOG DUMP_DIR [--last N] [--verbose] [--json FILE]

--json writes, per channel, every pushbuffer segment of the last N GPFIFO entries (address,
length in words, the number of acquires in it; releases and reductions are not counted) and every acquired semaphore address with its
count and payload range.

IOCTL_LOG is the interposer's log of the process (ioctl-<tag>-<pid>.log, with
NV_IOCTL_TRACE_MAPS), DUMP_DIR the directory nv_trace_dump wrote. For every channel the
script reads USERD (GPGet, GPPut), the last N GPFIFO entries before GPPut and the pushbuffer
segments they point to, decodes the method headers and reports the host methods of the
channel class (NVC86F_*, kernel-open/nvidia-uvm/clc86f.h of NVIDIA's open module): semaphore
acquires and releases with their addresses, WFI, YIELD, NON_STALL_INTERRUPT.
"""
import collections, glob, json, os, re, struct, sys

log, ddir = sys.argv[1], sys.argv[2]
last = int(sys.argv[sys.argv.index("--last") + 1]) if "--last" in sys.argv else 1024
verbose = "--verbose" in sys.argv
js_path = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
js = {"channels": []}
kv = re.compile(r"(\w+)=(\S+)")

chans, tsg_eng, tokens, ext, pools, objcls = [], {}, {}, [], [], {}
for line in open(log):
    d = dict(kv.findall(line))
    if line.startswith("A<") and d.get("status") == "0":
        if " TSG " in line:
            tsg_eng[d["hNew"]] = int(d["eng"])
        elif " CHANNEL " in line:
            chans.append(d)
        elif " MEMORY " in line:
            objcls[d["hNew"]] = (d["class"], d["attr"], d["attr2"], d["size"])
    elif line.startswith("C<") and "token" in d:
        tokens[d["hObject"]] = d["token"]
    elif line.startswith("U<") and " EXTERNAL " in line and d.get("status") == "0":
        ext.append((int(d["base"], 16), int(d["len"], 16), d["hMemory"], d["caching"]))
    elif line.startswith("U<") and " SEMPOOL " in line:
        pools.append((int(d["base"], 16), int(d["len"], 16), d["caching"]))

mem = []  # (va, bytes, name): a dumped object counts at every GPU VA it is mapped to
cpu = {}
for f in glob.glob(os.path.join(ddir, "obj-*.bin")):
    h, addr = os.path.basename(f)[4:-4].split("-")
    cpu[h] = open(f, "rb").read()
for base, ln, h, caching in ext:
    if h in cpu:
        mem.append((base, cpu[h][:ln], f"obj {h} class {objcls.get(h, ('?',))[0]} caching {caching}"))
for f in glob.glob(os.path.join(ddir, "pool-*.bin")):
    i, addr = os.path.basename(f)[5:-4].split("-")
    mem.append((int(addr, 16), open(f, "rb").read(), f"sempool {i}"))

def read(va, n):
    for base, data, name in mem:
        if base <= va and va + n <= base + len(data):
            return data[va - base:va - base + n], name
    return None, None

def where(va):
    for base, data, name in mem:
        if base <= va < base + len(data):
            return f"{name}+{va - base:#x}"
    for base, ln, h, caching in ext:
        if base <= va < base + ln:
            return f"obj {h} class {objcls.get(h, ('?',))[0]} (not dumped)+{va - base:#x}"
    return "unmapped"

HOST = {0x00: "SET_OBJECT", 0x08: "NOP", 0x20: "NON_STALL_INTERRUPT", 0x28: "MEM_OP_A", 0x2c: "MEM_OP_B",
        0x30: "MEM_OP_C", 0x34: "MEM_OP_D", 0x50: "SET_REFERENCE", 0x5c: "SEM_ADDR_LO", 0x60: "SEM_ADDR_HI",
        0x64: "SEM_PAYLOAD_LO", 0x68: "SEM_PAYLOAD_HI", 0x6c: "SEM_EXECUTE", 0x78: "WFI", 0x80: "YIELD"}
OPS = {0: "ACQUIRE", 1: "RELEASE", 2: "ACQ_STRICT_GEQ", 3: "ACQ_CIRC_GEQ", 4: "ACQ_AND", 5: "ACQ_NOR", 6: "REDUCTION"}

def methods(buf):
    """Yield (subchannel, method, data) for every method write in a pushbuffer segment."""
    w = struct.unpack(f"<{len(buf) // 4}I", buf[:len(buf) // 4 * 4]); i = 0
    while i < len(w):
        h = w[i]; i += 1
        op, cnt, sub, m = h >> 29, (h >> 16) & 0x1fff, (h >> 13) & 7, (h & 0xfff) << 2
        if op == 4:
            yield sub, m, cnt
        elif op in (1, 3, 5):
            for k in range(cnt):
                if i >= len(w):
                    return
                yield sub, m + (4 * k if op == 1 else (4 if op == 5 and k else 0)), w[i]; i += 1
        elif op == 7:
            return

for n, c in enumerate(chans):
    gp, uo = int(c["gpoff"], 16), int(c["userdoff"], 16)
    ub = next((b for b, ln, h, _ in ext if h == c["userd"]), None)
    flags = int(c["flags"], 16)
    head = (f"channel {n} handle {c['hNew']} token {tokens.get(c['hNew'], '?')} engine "
            f"{tsg_eng.get(c['hParent'], '?')} runqueue {(flags >> 4) & 1}")
    userd, _ = read(ub + uo, 0x90) if ub is not None else (None, None)
    ring, rname = read(gp, int(c["entries"]) * 8)
    if userd is None or ring is None:
        print(head, "| USERD or GPFIFO not in the dump"); continue
    gpget, gpput = struct.unpack_from("<II", userd, 0x88)
    N = int(c["entries"])
    sem = {"lo": 0, "hi": 0, "plo": 0, "phi": 0}
    ops = collections.Counter(); targets = collections.Counter(); other = collections.Counter(); segs = 0; missing = 0
    seq = []; seg_rec = []; sem_rec = {}
    for k in range(min(last, N), 0, -1):
        e0, e1 = struct.unpack_from("<II", ring, ((gpput - k) % N) * 8)
        ln = (e1 >> 10) & 0x1fffff
        if ln == 0:
            continue
        va = ((e1 & 0xff) << 32) | (e0 & ~3)
        buf, name = read(va, ln * 4)
        segs += 1
        if buf is None:
            missing += 1; other[f"segment in {where(va)}"] += 1; continue
        other[f"segment in {name.split('+')[0]}"] += 1
        n_acq = 0
        for sub, m, data in methods(buf):
            if m >= 0x100:
                continue
            if m == 0x5c: sem["lo"] = data
            elif m == 0x60: sem["hi"] = data
            elif m == 0x64: sem["plo"] = data
            elif m == 0x68: sem["phi"] = data
            elif m == 0x6c:
                addr = ((sem["hi"] & 0x1ffffff) << 32) | (sem["lo"] & ~3)
                op = OPS.get(data & 7, str(data & 7))
                tag = op + (" SWITCH_TSG" if data >> 12 & 1 else "") + (" WFI" if data >> 20 & 1 else "") + (" 64BIT" if data >> 24 & 1 else "")
                ops[tag] += 1; targets[(op, where(addr).split("+")[0], addr)] += 1
                seq.append((k, tag, addr, sem["plo"])); n_acq += op.startswith("ACQ")
                r = sem_rec.setdefault(addr, [0, sem["plo"], sem["plo"], data])
                r[0] += 1; r[1] = min(r[1], sem["plo"]); r[2] = max(r[2], sem["plo"])
            elif m in HOST:
                ops[HOST[m]] += 1
            else:
                ops[f"host {m:#x}"] += 1
        seg_rec.append((va, ln, n_acq))
    js["channels"].append({"n": n, "handle": c["hNew"], "token": tokens.get(c["hNew"]), "engine": tsg_eng.get(c["hParent"]),
                           "runqueue": (flags >> 4) & 1, "gpget": gpget, "gpput": gpput, "segments": seg_rec,
                           "semaphores": {hex(a): r for a, r in sem_rec.items()}})
    print(f"{head} | GPGet {gpget} GPPut {gpput} | segments {segs} (unreadable {missing}) | {dict(other)}")
    print("   methods:", dict(ops))
    by = collections.defaultdict(lambda: [0, set()])
    for (op, obj, addr), cnt in targets.items():
        by[(op, obj)][0] += cnt; by[(op, obj)][1].add(addr)
    for (op, obj), (cnt, addrs) in sorted(by.items()):
        a = sorted(addrs)
        print(f"   {op:14} x{cnt:5} on {obj}: {len(a)} addresses {', '.join(hex(x) for x in a[:6])}{' ...' if len(a) > 6 else ''}")
    if verbose:
        for k, tag, addr, p in seq[-24:]:
            print(f"      -{k:4} {tag:28} {addr:#x} payload {p:#x}")
if js_path:
    js["external"] = [(hex(b), hex(l), h, objcls.get(h, ("?",) * 4)[0], objcls.get(h, ("?",) * 4)[1], objcls.get(h, ("?",) * 4)[2]) for b, l, h, _ in ext]
    json.dump(js, open(js_path, "w"))
print("semaphore pools (UVM):", [(hex(b), hex(l), "caching " + c) for b, l, c in pools])
