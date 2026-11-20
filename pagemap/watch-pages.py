# Read-only watcher for the semaphore pages: for every
# probe process (pcie_probe.py or l2_probe.py) on the host, AGE seconds after it is first seen and again every REPEAT
# seconds, write the physical frame number and the kpageflags word of every page of its
# /dev/nvidia* mappings (up to 64 MiB each). It reads /proc/<pid>/maps, /proc/<pid>/pagemap,
# /proc/kpageflags, /proc/<pid>/numa_maps and /proc/zoneinfo, and nothing else.
import json, os, struct, sys, time
OUT = sys.argv[1]; DEADLINE = time.time() + float(sys.argv[2]); AGE = float(sys.argv[3]); REPEAT = float(sys.argv[4])
seen = {}; last = {}
def cmd(pid):
    try: return open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
    except Exception: return []
def snap(pid, tag):
    rec = {"pid": pid, "tag": tag, "t": time.time(), "vmas": []}
    try:
        pm = open(f"/proc/{pid}/pagemap", "rb"); kf = open("/proc/kpageflags", "rb")
        for line in open(f"/proc/{pid}/maps"):
            f = line.split()
            if len(f) < 6 or "nvidia" not in f[5] or "uvm" in f[5]: continue
            a, b = (int(x, 16) for x in f[0].split("-")); n = (b - a) >> 12
            v = {"start": hex(a), "pages": n, "perms": f[1], "off": f[2], "path": f[5]}
            if n <= 16384:
                pm.seek((a >> 12) * 8); raw = pm.read(n * 8); ent = struct.unpack(f"<{len(raw)//8}Q", raw)
                v["pfn"] = [(e & ((1 << 55) - 1)) if e >> 63 else 0 for e in ent]
                if n <= 512:
                    fl = []
                    for p in v["pfn"]:
                        if p:
                            kf.seek(p * 8); fl.append(struct.unpack("<Q", kf.read(8))[0])
                        else:
                            fl.append(0)
                    v["flags"] = fl
            rec["vmas"].append(v)
        rec["numa_nvidia"] = [" ".join(t for t in l.split() if t[0] == "N" or t[:2] in ("7f", "20", "10") or t.startswith(("mapped", "kernelpagesize"))) for l in open(f"/proc/{pid}/numa_maps") if "nvidia" in l and "uvm" not in l]
    except Exception as e:
        rec["error"] = repr(e)
    return rec
with open(OUT, "a", buffering=1) as out:
    out.write(json.dumps({"zoneinfo": [l.strip() for l in open("/proc/zoneinfo") if l.startswith("Node") or "start_pfn" in l or "spanned" in l]}) + "\n")
    while time.time() < DEADLINE and not os.path.exists("/tmp/stop"):
        for d in os.listdir("/proc"):
            if not d.isdigit(): continue
            c = cmd(d)
            if c and any(os.path.basename(x) in (b"pcie_probe.py", b"l2_probe.py") for x in c[1:]) and b"--tag" in c and b"--poll" not in c:
                pid = int(d); seen.setdefault(pid, time.time())
                due = seen[pid] + AGE if pid not in last else last[pid] + REPEAT
                if time.time() >= due:
                    tag = c[c.index(b"--tag") + 1].decode(); out.write(json.dumps(snap(pid, tag)) + "\n"); last[pid] = time.time()
        time.sleep(2)
