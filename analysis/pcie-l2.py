#!/usr/bin/env python3
"""Reads the L2 probe's array (probe/l2_probe.py): cycles[region, sm, rep, page].

Usage: analysis/pcie-l2.py FILE.npy
Prints the spread of the load time over SMs and pages, the first components of the
SM-by-page matrix after removing each SM's and each page's mean, and, where the pages fall
into two classes, the fit of the class over GF(2) on bits 12 to 20 of the page's offset.
"""
import itertools, json, sys
import numpy as np

f = sys.argv[1]
c = np.load(f)
facts = json.load(open(f[:-4] + ".json"))
R, S, N, P = c.shape
m = np.median(c, axis=2)                                  # region, sm, page
print("file", f, "shape", c.shape, "device", facts["device"], "flush_mb", facts["flush_mb"])
print("cycles per load: median %.1f, 1%% %.1f, 99%% %.1f" % (np.median(m), np.percentile(m, 1), np.percentile(m, 99)))
print("rep-to-rep sd of a cell (median over cells): %.2f" % np.median(c.std(axis=2)))
sm_mean = m.mean(axis=(0, 2)); pg_mean = m.mean(axis=1)   # per sm; per region,page
print("per-SM mean: min %.1f max %.1f sd %.2f" % (sm_mean.min(), sm_mean.max(), sm_mean.std()))
print("per-page mean (over SMs): min %.1f max %.1f sd %.2f" % (pg_mean.min(), pg_mean.max(), pg_mean.std()))

def bits(v):
    return [(v >> b) & 1 for b in range(9)]               # page index bit b = address bit 12 + b

def solve(rows, rhs):
    """Least-inconsistent GF(2) fit by exhaustive search over the 2^9 masks, constant free."""
    best = None
    for mask in range(1, 512):
        par = np.array([bin(p & mask).count("1") & 1 for p in rows])
        agree = int((par == rhs).sum()); agree = max(agree, len(rhs) - agree)
        if best is None or agree > best[0]:
            best = (agree, mask)
    return best

for r in range(R):
    x = m[r] - m[r].mean(axis=1, keepdims=True) - m[r].mean(axis=0, keepdims=True) + m[r].mean()
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    print("region %d: singular values %s; residual sd %.2f" % (r, np.round(s[:5], 1), x.std()))
    sm_side, pg_side = u[:, 0] * s[0], vt[0]
    g = sm_side > 0
    print("  first component: SMs %d against %d; SM ids of the smaller side: %s" % (
        g.sum(), (~g).sum(), [facts["sms"][i] for i in np.where(g if g.sum() <= (~g).sum() else ~g)[0]][:70]))
    # the page classes as the two SM groups see them
    if 0 < g.sum() < S:
        d = m[r][g].mean(axis=0) - m[r][~g].mean(axis=0)  # per page: group A minus group B
        srt = np.sort(d); gap = np.diff(srt); k = int(gap.argmax())
        print("  page contrast between the SM groups: min %.1f max %.1f; largest gap %.1f between %.1f and %.1f (%d pages below, %d above)" % (
            d.min(), d.max(), gap[k], srt[k], srt[k + 1], k + 1, P - k - 1))
        cls = (d > (srt[k] + srt[k + 1]) / 2).astype(int)
        agree, mask = solve(list(range(P)), cls)
        print("  best parity over bits 12..20 for that split: address bits %s, %d of %d pages" % (
            [12 + b for b in range(9) if mask >> b & 1], agree, P))
    # per-page mean latency itself (any SM): classes by address bits
    pm = m[r].mean(axis=0)
    for b in range(9):
        on = np.array([p >> b & 1 for p in range(P)]) == 1
        print("  bit %2d: mean load time with the bit set minus clear, all SMs: %+.2f" % (12 + b, pm[on].mean() - pm[~on].mean()), end=";")
        print(" SM-side first component %+.2f" % (pg_side[on].mean() - pg_side[~on].mean()))
