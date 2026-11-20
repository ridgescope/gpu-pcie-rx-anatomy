#!/usr/bin/env python3
"""The L2 partition of each page of the semaphore object against its physical frame.

Usage: analysis/pcie-l2sem.py ROOT
ROOT holds probe/l2-sem-*.npy and .json (probe/l2_probe.py, cross mode: cycles
[warm SM, timing SM, rep, page], warm SMs 0 and 4, timing SMs 0 to 7) and pages.jsonl (the
watcher's frames). SMs 0 to 3 are of one L2 partition ("a") and 4 to 7 of the other ("b")
(job 5888). A page warmed from a and timed from b is slow when its home is a; warmed from b
and timed from a, slow when its home is b. Writes ROOT/sem-pages.csv and prints the
agreement of the home partition with the parity of the frame under the mask 0x13a78b3.
"""
import csv, glob, gzip, json, os, sys
import numpy as np

root = sys.argv[1]
MASK = 0x13a78b3
snaps = {}
pj = os.path.join(root, "pages.jsonl")
for line in (gzip.open(pj + ".gz", "rt") if not os.path.exists(pj) else open(pj)):
    s = json.loads(line)
    if "tag" in s:
        snaps.setdefault(s["tag"], s)
rows = []
for f in sorted(glob.glob(os.path.join(root, "probe", "l2-sem-*.npy"))):
    tag = os.path.basename(f)[3:-4]
    facts = json.load(open(f[:-4] + ".json"))
    m = np.median(np.load(f), axis=2)
    ab = m[0][4:8].mean(axis=0)       # warmed from a (SM 0), timed from b
    ba = m[1][0:4].mean(axis=0)       # warmed from b (SM 4), timed from a
    aa = m[0][0:4].mean(axis=0); bb = m[1][4:8].mean(axis=0)
    vma = next((v for v in snaps.get(tag, {}).get("vmas", []) if int(v["start"], 16) == int((facts.get("object") or {}).get("cpu_addr") or facts["base"], 16) and "pfn" in v), None)
    for p in range(512):
        pfn = vma["pfn"][p] if vma else 0
        rows.append({"tag": tag, "page": p, "pfn": hex(pfn), "warm_a_time_b": round(float(ab[p]), 1), "warm_b_time_a": round(float(ba[p]), 1),
                     "warm_a_time_a": round(float(aa[p]), 1), "warm_b_time_b": round(float(bb[p]), 1),
                     "home": "a" if ab[p] > ba[p] else "b", "contrast": round(float(ab[p] - ba[p]), 1),
                     "parity": (bin(pfn & MASK).count("1") & 1) if pfn else ""})
w = csv.DictWriter(open(os.path.join(root, "sem-pages.csv"), "w", newline=""), fieldnames=list(rows[0]))
w.writeheader(); w.writerows(rows)
c = np.array([r["contrast"] for r in rows])
print("pages", len(rows), "with a frame", sum(r["parity"] != "" for r in rows), "distinct frames", len({r["pfn"] for r in rows if r["parity"] != ""}))
print("contrast (warm a, time b) - (warm b, time a): |c| min %.1f, 1%% %.1f, median %.1f" % (abs(c).min(), np.percentile(abs(c), 1), np.median(abs(c))))
for name in ("warm_a_time_b", "warm_b_time_a", "warm_a_time_a", "warm_b_time_b"):
    for h in "ab":
        v = [r[name] for r in rows if r["home"] == h]
        print("  %s, home %s: mean %.0f (min %.0f max %.0f) n=%d" % (name, h, np.mean(v), min(v), max(v), len(v)))
ok = [r for r in rows if r["parity"] != ""]
n11 = sum(r["home"] == "a" and r["parity"] == 1 for r in ok); n10 = sum(r["home"] == "a" and r["parity"] == 0 for r in ok)
n01 = sum(r["home"] == "b" and r["parity"] == 1 for r in ok); n00 = sum(r["home"] == "b" and r["parity"] == 0 for r in ok)
print("home a: odd parity %d, even %d; home b: odd %d, even %d" % (n11, n10, n01, n00))
for tag in sorted({r["tag"] for r in ok}):
    t = [r for r in ok if r["tag"] == tag]
    print("  %s: odd parity and home a, or even and home b: %d of %d" % (tag, sum((r["home"] == "a") == (r["parity"] == 1) for r in t), len(t)))
# which frame bits vary, and a free GF(2) fit of home on the frame bits
pf = np.array([int(r["pfn"], 16) for r in ok]); y = np.array([1 if r["home"] == "a" else 0 for r in ok])
var = [b for b in range(40) if 0 < ((pf >> b) & 1).sum() < len(pf)]
A = np.array([[(p >> b) & 1 for b in var] + [1] for p in pf], dtype=np.uint8); yv = y.astype(np.uint8).copy()
A = np.concatenate([A, yv[:, None]], axis=1); rnk = 0; piv = []
for col in range(len(var) + 1):
    r = next((i for i in range(rnk, len(A)) if A[i, col]), None)
    if r is None: continue
    A[[rnk, r]] = A[[r, rnk]]
    for i in range(len(A)):
        if i != rnk and A[i, col]: A[i] ^= A[rnk]
    piv.append(col); rnk += 1
cons = not any(A[i, -1] and not A[i, :-1].any() for i in range(len(A)))
sol = {piv[i]: int(A[i, -1]) for i in range(rnk)}
print("frame bits that vary:", var)
print("GF(2) fit of home on them: rank %d of %d unknowns, consistent %s" % (rnk, len(var) + 1, cons))
if cons:
    bits = [var[c] for c in sol if c < len(var) and sol[c]]
    print("  frame bits in the solution:", bits, "mask", hex(sum(1 << b for b in bits)), "constant", sol.get(len(var), 0), "(free columns: %d)" % (len(var) + 1 - rnk))
