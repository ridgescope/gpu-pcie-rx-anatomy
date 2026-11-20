#!/usr/bin/env python3
"""Traffic and L2 partition of the same semaphore pages in one process.

This script was committed before the job it scores was submitted, and its scoring rules
have not been edited since.

Usage: analysis/pcie-l2traffic.py ROOT
ROOT holds probe/l2-<tag>.npy and .json (l2_probe.py run inside the traffic
probe after its windows: cycles[warm SM, timing SM, rep, page] of the semaphore object, warm
SMs 0 and 4, timing SMs 0 to 7) and windows.csv (analysis/pcie-windows.py: each
window's receive rate and the channel of the stream with the long kernels).

The home partition of a page: "a" (SMs 0 to 3) when the page, warmed from SM 0 and timed
from SMs 4 to 7, is slower than warmed from SM 4 and timed from SMs 0 to 3; else "b". A
process is scored only if every one of its 512 pages has a contrast of at least 100 cycles
(the SM groups hold on its card). The prediction, fixed from jobs 5868 and 5889: a window is
high when the page of its stream's semaphores is at home in "b". Pages: our streams 0 to 2 at
offset 0x26000 of the object, 3 to 23 at 0x2f000, 24 to 45 at 0x2e000, 46 and later at
0x37000, the default stream (windows "v-*") at 0x27000. A window is read as high when its
rate exceeds 5,700 MB/s with the long kernels on an even channel, 4,000 MB/s on an odd one.
Every window of a scored process is scored. Writes ROOT/l2-traffic.csv.
"""
import collections, csv, json, os, sys
import numpy as np

root = sys.argv[1]

def page_of(mode):
    if mode.startswith("v-"):
        return 0x27
    k = int(mode[1:3])
    return 0x26 if k < 3 else 0x37 if k >= 46 else (0x2ff84 - (k - 3) * 0xc0) >> 12

home, bad = {}, []
rows = []
for w in csv.DictReader(open(os.path.join(root, "windows.csv"))):
    tag, mode = w["tag"], w["mode"]
    if tag not in home and tag not in bad:
        f = os.path.join(root, "probe", "l2-%s.npy" % tag)
        if not os.path.exists(f):
            bad.append(tag); continue
        m = np.median(np.load(f), axis=2)
        c = m[0][4:8].mean(axis=0) - m[1][0:4].mean(axis=0)
        if abs(c).min() < 100:
            bad.append(tag); continue
        home[tag] = c
    if tag in bad:
        continue
    ch = next((int(tok.split(">ch")[1].split(":")[0]) for tok in w["layout"].split() if tok.endswith("L")), None)
    pg = page_of(mode); c = home[tag][pg]
    pred = "L" if c > 0 else "H"                 # contrast > 0: home a
    seen = None if ch is None else ("H" if float(w["rx_MBps"]) > (5700 if ch % 2 == 0 else 4000) else "L")
    rows.append({"tag": tag, "mode": mode, "page": hex(pg), "home": "a" if c > 0 else "b", "contrast_cycles": round(float(c), 1),
                 "predicted": pred, "channel": ch, "rx_MBps": w["rx_MBps"], "observed": seen, "agree": seen == pred if seen else None})
out = csv.DictWriter(open(os.path.join(root, "l2-traffic.csv"), "w", newline=""), fieldnames=list(rows[0]))
out.writeheader(); out.writerows(rows)
scored = [r for r in rows if r["observed"]]
print("processes scored: %d; not scored (no array, or the SM groups do not hold): %s" % (len(home), bad))
print("windows scored: %d, agree: %d; windows without a channel: %d" % (len(scored), sum(r["agree"] for r in scored), len(rows) - len(scored)))
pages = collections.defaultdict(list)
for r in scored:
    pages[(r["tag"], r["page"])].append(r["agree"])
print("pages: %d, all windows agree: %d" % (len(pages), sum(all(v) for v in pages.values())))
print("pages at home in b (predicted high): %d, in a: %d" % (len({k for k in pages if next(r for r in scored if (r['tag'], r['page']) == k)['home'] == 'b'}),
                                                              len({k for k in pages if next(r for r in scored if (r['tag'], r['page']) == k)['home'] == 'a'})))
for r in scored:
    if not r["agree"]:
        print("  disagree:", r)
