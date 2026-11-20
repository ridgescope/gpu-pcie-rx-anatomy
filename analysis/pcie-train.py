#!/usr/bin/env python3
"""Simulus's own training under the default connections and under two (job x-pcie-train).

Usage: analysis/pcie-train.py ROOT
ROOT holds job.out (the arms' start lines and the per-epoch records), series.csv (the job's
DCGM series at 5 s steps, with a column pcie_rx_MBps) and cupti/*-<pid>.log.gz of
the traced arm ("KT" lines: start ns, end ns, s=<stream> ch=<channel> ...). Writes
ROOT/epochs.csv (per arm and epoch: seconds per phase and the mean of the PCIe receive throughput inside
each phase) and ROOT/train-streams.csv (the traced arm: per phase, stream and channel, the
kernels, their time and the time of kernels longer than 1 ms), and prints both in short.
The first epoch of every arm is left out of the means.
"""
import collections, csv, datetime, glob, gzip, json, os, re, statistics as st, sys

root = sys.argv[1]
PH = ["collection", "tokenizer", "world_model", "actor_critic"]
arms, cur = [], None
for line in open(os.path.join(root, "job.out"), errors="replace"):
    m = re.search(r"X arm (\S+) \(connections (\S+) epochs (\d+) ktrace (\d)\) start \S+ (\d+)", line)
    if m:
        cur = {"arm": m.group(1), "conn": m.group(2), "kt": m.group(4) == "1", "epochs": []}; arms.append(cur); continue
    m = re.search(r"wrapper: connections \S+ pid (\d+)", line)
    if m and cur:
        cur["pid"] = int(m.group(1))
    i = line.find('{"rs":"step"')
    if i >= 0 and cur:
        r = json.loads(line[i:])
        end = datetime.datetime.fromisoformat(r["t"].replace("Z", "+00:00")).timestamp()
        t = end - r["wall"]; win = {}
        for p in PH:
            win[p] = (t, t + r["phases"][p]); t += r["phases"][p]
        cur["epochs"].append({"step": r["step"], "wall": r["wall"], "phases": r["phases"], "win": win})
series = []
for r in csv.DictReader(open(os.path.join(root, "series.csv"))):
    try:
        ts = float(r["ts"])
    except ValueError:
        ts = datetime.datetime.fromisoformat(r["ts"].replace("Z", "+00:00")).timestamp()
    if r.get("pcie_rx_MBps") not in (None, ""):
        series.append((ts, float(r["pcie_rx_MBps"])))
def receive_mean(a, b):
    v = [x for t, x in series if a + 5 <= t <= b - 2]
    return round(st.mean(v), 1) if v else ""
rows = []
for a in arms:
    for e in a["epochs"]:
        row = {"arm": a["arm"], "connections": a["conn"], "epoch": e["step"], "wall_s": round(e["wall"], 2)}
        for p in PH:
            row[p + "_s"] = round(e["phases"][p], 2)
        for p in PH[1:]:
            row[p + "_rx_MBps"] = receive_mean(*e["win"][p])
        rows.append(row)
w = csv.DictWriter(open(os.path.join(root, "epochs.csv"), "w", newline=""), fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("arm        conn  epochs  wall_s  tokenizer_s  world_model_s  actor_critic_s | rx MB/s: tok   wm    ac")
for a in arms:
    r = [x for x in rows if x["arm"] == a["arm"] and x["epoch"] > 1]
    f = lambda k: st.mean(x[k] for x in r)
    g = lambda k: (lambda v: "%6.0f" % st.mean(v) if v else "   n/a")([x[k] for x in r if x[k] != ""])
    print("%-10s %-7s %d      %6.2f  %6.2f       %6.2f         %6.2f         |      %s %s %s" % (
        a["arm"], a["conn"], len(r), f("wall_s"), f("tokenizer_s"), f("world_model_s"), f("actor_critic_s"),
        g("tokenizer_rx_MBps"), g("world_model_rx_MBps"), g("actor_critic_rx_MBps")))
kt = next((a for a in arms if a["kt"]), None)
fs = glob.glob(os.path.join(root, "cupti", "*-%d.log.gz" % kt["pid"])) if kt else []
if fs:
    wins = [(e["win"][p][0], e["win"][p][1], e["step"], p) for e in kt["epochs"] for p in PH]
    agg = collections.defaultdict(lambda: [0, 0, 0, collections.Counter()]); n = last = 0
    for line in gzip.open(fs[0], "rt"):
        if not line.startswith("KT "):
            continue
        p = line.split(None, 7); t0 = int(p[1]) / 1e9; dur = int(p[2]) - int(p[1]); n += 1; last = t0
        ph = next((ph for a_, b_, s_, ph in wins if a_ <= t0 < b_), None)
        if ph is None:
            continue
        a = agg[(ph, int(p[3][2:]), int(p[4][3:]))]
        a[0] += 1; a[1] += dur
        if dur >= 1_000_000:
            a[2] += dur
        a[3][(p[7].strip() if len(p) > 7 else "?")[:70]] += dur
    out = []
    for (ph, s, ch), a in sorted(agg.items(), key=lambda kv: (PH.index(kv[0][0]), kv[0][1])):
        top = a[3].most_common(1)[0]
        out.append({"phase": ph, "stream": s, "channel": ch, "kernels": a[0], "kernel_s": round(a[1] / 1e9, 2),
                    "long_kernel_s": round(a[2] / 1e9, 2), "top_kernel": top[0], "top_kernel_s": round(top[1] / 1e9, 2)})
    w = csv.DictWriter(open(os.path.join(root, "train-streams.csv"), "w", newline=""), fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print("\ntraced arm %s: %d kernel records, the last at +%.0f s of the first epoch's start; epochs %s" % (
        kt["arm"], n, last - kt["epochs"][0]["win"]["collection"][0], [e["step"] for e in kt["epochs"]]))
    for r in out:
        print("  %-12s stream %-4d channel %d  kernels %8d  %7.2f s  long %6.2f s  %s" % (r["phase"], r["stream"], r["channel"], r["kernels"], r["kernel_s"], r["long_kernel_s"], r["top_kernel"][:58]))
