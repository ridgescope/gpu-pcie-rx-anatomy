#!/usr/bin/env python3
"""The value of each semaphore page against its physical frame.

Usage: analysis/pcie-frames.py ROOT [PAGES_JSONL] [--check OTHER_ROOT]
ROOT holds job.out (the probe's lines) and pages.jsonl or pages.jsonl.gz (pagemap/watch-pages.py
run on the node). The semaphore object is the 512-page /dev/nvidia* mapping at 0x204600000;
the block of our stream k lies in the page at offset 0x26000 (k < 3), 0x2f000 (3 to 23),
0x2e000 (24 to 45), 0x37000 (46, 47: decoded in job 5850), the default stream's at 0x27000.
A window is high if its rate exceeds 4,800 MB/s; a page takes the majority of its windows.
Writes ROOT/page-frames.csv and prints the pages by value with tests of single address bits,
then solves over GF(2) for a set of frame bits whose parity equals the value on every page
(pages with windows of both values are left out), and checks it leaving one process out.
With --check OTHER_ROOT it predicts the pages of OTHER_ROOT/page-frames.csv from the fit.
"""
import collections, json, os, statistics, sys

import gzip
root = sys.argv[1]
pj = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else os.path.join(root, "pages.jsonl")
if not os.path.exists(pj):
    pj += ".gz"
opener = gzip.open if pj.endswith(".gz") else open
rx = {}
for line in open(os.path.join(root, "job.out"), errors="replace"):
    i = line.find('{"probe"')
    if i < 0:
        continue
    try:
        r = json.loads(line[i:])
    except ValueError:
        continue
    if r["probe"] == "nvml-mode":
        rx.setdefault(r["tag"], {})[r["mode"]] = r["rx_MBps"]

def page_of(k):
    return 0x26 if k < 3 else 0x37 if k >= 46 else (0x2ff84 - (k - 3) * 0xc0) >> 12

rows, done = [], set()
for line in opener(pj, "rt"):
    s = json.loads(line)
    if "tag" not in s or s["tag"] not in rx or s["tag"] in done:
        continue
    v = next((v for v in s["vmas"] if v["start"] == "0x204600000" and "pfn" in v), None)
    if v is None:
        continue
    done.add(s["tag"]); w = rx[s["tag"]]; vals = collections.defaultdict(list)
    for m, r in w.items():
        if m.startswith("x"):
            vals[page_of(int(m[1:3]))].append(r)
        elif m.startswith("v-"):
            vals[0x27].append(r)
    for pg in sorted(vals):
        hi = sum(1 for r in vals[pg] if r > 4800)
        cls = "H" if hi * 2 > len(vals[pg]) else "L"
        rows.append({"tag": s["tag"], "page": hex(pg), "windows": len(vals[pg]), "high_windows": hi,
                     "median_rx_MBps": round(statistics.median(vals[pg])), "value": cls, "pfn": hex(v["pfn"][pg]),
                     "flags": hex(v["flags"][pg]), "prev_pfn": hex(v["pfn"][pg - 1]), "next_pfn": hex(v["pfn"][pg + 1])})
import csv
with open(os.path.join(root, "page-frames.csv"), "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
H = [int(r["pfn"], 16) for r in rows if r["value"] == "H"]; L = [int(r["pfn"], 16) for r in rows if r["value"] == "L"]
print(f"{len(done)} processes, {len(rows)} pages: {len(H)} high, {len(L)} low; unclear pages (30 to 70 % high windows):",
      sum(1 for r in rows if 0.3 < r["high_windows"] / r["windows"] < 0.7))
for bit in range(0, 28):
    h1 = sum(p >> bit & 1 for p in H); l1 = sum(p >> bit & 1 for p in L)
    if H and L:
        print(f"  frame bit {bit:2} (address bit {bit + 12}): set in {h1}/{len(H)} high, {l1}/{len(L)} low")

NB = 28
def fit(X):
    piv = {}
    for p, y in X:
        row = p | (1 << NB)
        for col in sorted(piv):
            if row >> col & 1:
                row ^= piv[col][0]; y ^= piv[col][1]
        if row == 0:
            if y:
                return None
            continue
        col = (row & -row).bit_length() - 1; piv[col] = (row, y)
    sol = 0
    for col in sorted(piv, reverse=True):
        row, v = piv[col]
        for c in range(col + 1, NB + 1):
            if row >> c & 1 and sol >> c & 1:
                v ^= 1
        if v:
            sol |= 1 << col
    return sol
par = lambda sol, p: (bin(p & sol & ((1 << NB) - 1)).count("1") + (sol >> NB & 1)) & 1
clean = [(int(r["pfn"], 16), 1 if r["value"] == "H" else 0, r["tag"]) for r in rows if r["high_windows"] in (0, r["windows"])]
sol = fit([(p, y) for p, y, _ in clean])
if sol is None:
    print("no set of frame bits has a parity equal to the value on all", len(clean), "pages")
else:
    bits = [i for i in range(NB) if sol >> i & 1]
    print(f"parity of frame bits {bits} (address bits {[b + 12 for b in bits]}), constant {sol >> NB & 1}: high when odd; fits",
          sum(par(sol, p) == y for p, y, _ in clean), "of", len(clean))
    good = tot = 0
    for t in sorted({t for _, _, t in clean}):
        s = fit([(p, y) for p, y, tt in clean if tt != t])
        for p, y, tt in clean:
            if tt == t:
                tot += 1; good += s is not None and par(s, p) == y
    print("leaving one process out:", good, "of", tot, "pages predicted")
    if "--check" in sys.argv:
        other = list(csv.DictReader(open(os.path.join(sys.argv[sys.argv.index("--check") + 1], "page-frames.csv"))))
        o = [(int(r["pfn"], 16), 1 if r["value"] == "H" else 0) for r in other]
        print("pages of the other run predicted:", sum(par(sol, p) == y for p, y in o), "of", len(o))
