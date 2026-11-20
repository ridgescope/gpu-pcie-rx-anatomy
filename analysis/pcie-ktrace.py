#!/usr/bin/env python3
"""Which channel each stream's kernels ran on, per process of job x-pcie-ktrace.

Reads results/x-pcie-ktrace/job.out (the probe's JSON lines: the pid and the rates per
tag), results/x-pcie-ktrace/cupti/*-<pid>.log(.gz) (the "KT"/"MT" lines of
a CUPTI subscriber that logs CUpti_ActivityKernel9.channelID) and results/x-pcie-ktrace/ioctl/ (the channel table). Writes
streams.csv (one row per process, stream and channel) and prints one line per process.
Usage: analysis/pcie-ktrace.py [results/x-pcie-ktrace]
"""
import collections, csv, glob, gzip, json, os, re, sys

root = sys.argv[1] if len(sys.argv) > 1 else "results/x-pcie-ktrace"
info = {}
for line in open(os.path.join(root, "job.out"), errors="replace"):
    i = line.find('{"probe"')
    if i < 0:
        continue
    try:
        r = json.loads(line[i:])
    except ValueError:
        continue
    d = info.setdefault(r.get("tag"), {})
    if r["probe"] == "facts":
        d["pid"] = r["pid"]
    elif r.get("mode") == "vgg-1" and r["probe"] == "mode":
        d["ms"], d["start"], d["end"] = r["per_iteration_ms"], r["start"], r["end"]
    elif r.get("mode") == "vgg-1" and r["probe"] == "nvml-mode":
        d["rx"] = r["rx_MBps"]
bypid = {d["pid"]: t for t, d in info.items() if "pid" in d}
rows = []
for f in sorted(glob.glob(os.path.join(root, "cupti", "*.log*"))):
    mt = re.search(r"-(\d+)\.log", f)
    tag = bypid.get(int(mt.group(1))) if mt else None
    pid = int(mt.group(1)) if mt else None
    if not tag:
        continue
    agg = collections.defaultdict(lambda: [0, 0, 0, collections.Counter()])  # count, sum ns, long ns, names
    for line in (gzip.open(f, "rt") if f.endswith(".gz") else open(f)):
        if not line.startswith("KT "):
            continue
        p = line.split(None, 7)
        dur = int(p[2]) - int(p[1])
        a = agg[(int(p[3][2:]), int(p[4][3:]), int(p[5][3:]))]
        a[0] += 1; a[1] += dur
        if dur >= 1_000_000:
            a[2] += dur
        a[3][p[7].strip()[:60] if len(p) > 7 else "?"] += dur
    info[tag]["streams"] = agg
    for (s, ch, ct), a in sorted(agg.items()):
        top = a[3].most_common(1)[0]
        rows.append({"tag": tag, "rx_MBps": info[tag].get("rx", ""), "ms": info[tag].get("ms", ""), "stream": s,
                     "channel": ch, "channel_type": ct, "kernels": a[0], "kernel_ms": round(a[1] / 1e6, 1),
                     "long_kernel_ms": round(a[2] / 1e6, 1), "top_kernel": top[0], "top_kernel_ms": round(top[1] / 1e6, 1)})
if rows:
    with open(os.path.join(root, "streams.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for tag in sorted(info, key=lambda t: (t.rsplit("-", 1)[0], t)):
    d = info[tag]
    if "rx" not in d:
        continue
    m = " ".join(f"s{s}>ch{ch}:{a[0]}k/{a[1]/1e6:.0f}ms/long{a[2]/1e6:.0f}" for (s, ch, ct), a in sorted(d.get("streams", {}).items()))
    print(f'{tag:9} rx={d["rx"]:8.0f} ms={d.get("ms", 0):6.2f}  {m}')
