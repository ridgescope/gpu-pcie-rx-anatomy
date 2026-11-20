#!/usr/bin/env python3
"""Every window of a multi-window probe job with its rate and its streams on channels.

Reads ROOT/job.out (the probe's JSON lines) and ROOT/cupti/*-<pid>.log.gz (the
"KT" lines of a CUPTI subscriber that logs CUpti_ActivityKernel9.channelID), writes ROOT/windows.csv: tag, mode, rx and tx in MB/s, ms
per iteration, and the layout as "s<stream>>ch<channel>:<kernels>", with "L" after the
stream whose kernels exceed 1 ms. Usage: analysis/pcie-windows.py ROOT [TAG_PREFIX]
"""
import collections, csv, glob, gzip, json, os, sys

root = sys.argv[1]; prefix = sys.argv[2] if len(sys.argv) > 2 else ""
info = {}
for line in open(os.path.join(root, "job.out"), errors="replace"):
    i = line.find('{"probe"')
    if i < 0:
        continue
    try:
        r = json.loads(line[i:])
    except ValueError:
        continue
    d = info.setdefault(r.get("tag"), {"w": {}, "rx": {}, "tx": {}, "ms": {}})
    if r["probe"] == "facts":
        d["pid"] = r["pid"]
    elif r["probe"] == "mode":
        d["w"][r["mode"]] = (r["start"], r["end"]); d["ms"][r["mode"]] = r["per_iteration_ms"]
    elif r["probe"] == "nvml-mode":
        d["rx"][r["mode"]] = r["rx_MBps"]; d["tx"][r["mode"]] = r["tx_MBps"]
rows = []
for tag in sorted(t for t in info if t and t.startswith(prefix) and "pid" in info[t]):
    d = info[tag]
    fs = glob.glob(os.path.join(root, "cupti", f"*-{d['pid']}.log.gz"))
    wins = sorted((a, b, m) for m, (a, b) in d["w"].items() if m != "idle")
    agg = {m: collections.defaultdict(lambda: [0, 0]) for _, _, m in wins}
    if fs:
        k = 0
        for line in gzip.open(fs[0], "rt"):
            if not line.startswith("KT "):
                continue
            p = line.split(None, 6); t = int(p[1]) / 1e9; dur = int(p[2]) - int(p[1])
            while k < len(wins) and t >= wins[k][1]:
                k += 1
            for a, b, m in wins[max(0, k - 1):k + 2]:
                if a <= t < b:
                    x = agg[m][(int(p[3][2:]), int(p[4][3:]))]; x[0] += 1; x[1] += dur if dur > 1e6 else 0
                    break
    for a, b, m in wins:
        lay = " ".join(f"s{s}>ch{ch}:{x[0]}{'L' if x[1] else ''}" for (s, ch), x in sorted(agg[m].items()))
        rows.append({"tag": tag, "mode": m, "rx_MBps": d["rx"].get(m, ""), "tx_MBps": d["tx"].get(m, ""),
                     "ms": d["ms"].get(m, ""), "layout": lay})
with open(os.path.join(root, "windows.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "windows")
