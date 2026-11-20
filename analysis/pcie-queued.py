#!/usr/bin/env python3
"""The receive throughput against the number of kernels queued behind a wait (data/queued-kernels.csv).

Every row is one measurement window of one process. In a window one stream runs a long matrix
product again and again, three streams wait for it, and N small kernels are queued behind each
wait; N changes from wait to wait, and rx_gbs_N<N> is the window's mean receive throughput over
its waits with that N. wait_on tells what the three streams wait for: a semaphore of the
producer stream (partition: the L2 partition of the page that holds it, a or b, from load times
in the same process), a word of our own in device or host memory, or nothing (the producer
alone). layout names the channels of the waiting streams, the runqueue of the producer and
whether the producer shares a channel with a waiter; windows are compared within one process
and one layout.

Run from the repository root: analysis/pcie-queued.py
"""
import csv, statistics as st
from collections import defaultdict

NS = [0, 1, 2, 3, 4, 6, 8, 10, 14, 20, 30]
rows = list(csv.DictReader(open("data/queued-kernels.csv")))
val = lambda r, n: float(r[f"rx_gbs_N{n}"]) if r[f"rx_gbs_N{n}"] else None
def mean_curve(sel):
    out = []
    for n in NS:
        v = [val(r, n) for r in sel if val(r, n) is not None]
        out.append(f"{n}:{st.mean(v):.2f}" if v else None)
    return " ".join(x for x in out if x)

sem = [r for r in rows if r["wait_on"] == "semaphore"]
print("mean of the windows, GB/s by N")
for p in "ab":
    sel = [r for r in sem if r["partition"] == p]
    print(f"  semaphore page in partition {p}, {len(sel)} windows from {len({r['process'] for r in sel})} processes: {mean_curve(sel)}")
for m in ("device", "host"):
    sel = [r for r in rows if r["wait_on"] == "word" and r["word_memory"] == m]
    print(f"  word in {m} memory, {len(sel)} windows: {mean_curve(sel)}")
alone = [val(r, 0) for r in rows if r["wait_on"] == "producer-alone"]
print(f"  producer alone, {len(alone)} windows: {st.mean(alone):.4f}")

# partition b against partition a inside one process and one layout
cells = defaultdict(lambda: defaultdict(list))
for r in sem:
    cells[(r["process"], r["layout"])][r["partition"]].append(r)
print("partition b over partition a, same process and layout: median over processes (processes with b above a / processes)")
for n in NS[1:]:
    per = defaultdict(list)
    for (proc, _), d in cells.items():
        if d["a"] and d["b"]:
            a = st.mean(val(r, n) for r in d["a"]); b = st.mean(val(r, n) for r in d["b"])
            per[proc].append((b / a, b - a))
    ratio = [st.mean(x[0] for x in v) for v in per.values()]
    diff = [st.mean(x[1] for x in v) for v in per.values()]
    print(f"  N={n:<2d} ratio {st.median(ratio):.2f}  difference {st.median(diff):+.2f} GB/s  ({sum(d > 0 for d in diff)}/{len(diff)})")

print("mean over N = 10, 14, 20, 30 of a window (the plateau), GB/s")
for p in "ab":
    pl = [st.mean(val(r, n) for n in (10, 14, 20, 30)) for r in sem if r["partition"] == p]
    print(f"  partition {p}: {min(pl):.2f} to {max(pl):.2f}, {len(pl)} windows")
