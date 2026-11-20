#!/usr/bin/env python3
"""Summary tables of the L2 probe (analysis/pcie-l2.py prints the analysis).

Usage: analysis/pcie-l2-summary.py DIR OUT_PREFIX
For every l2-*.npy in DIR: the SMs split by the sign of the first component of the
SM-by-page matrix (the group that holds SM 0 is "a"), and per region and page the mean
cycles per load seen by each group, the class (1 where group a is the slower), and the
parity of address bits 12, 13, 16, 17, 19 of the page's offset in its 2 MiB region.
Writes OUT_PREFIX-pages.csv and OUT_PREFIX-sms.csv.
"""
import csv, glob, json, os, sys
import numpy as np

d, out = sys.argv[1], sys.argv[2]
MASK = sum(1 << b for b in (0, 1, 4, 5, 7))
pages, sms = [], []
for f in sorted(glob.glob(os.path.join(d, "l2-*.npy"))):
    tag = os.path.basename(f)[3:-4]
    facts = json.load(open(f[:-4] + ".json"))
    m = np.median(np.load(f), axis=2)
    for r in range(m.shape[0]):
        x = m[r] - m[r].mean(1, keepdims=True) - m[r].mean(0, keepdims=True) + m[r].mean()
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        g = u[:, 0] > 0
        if not g[0]:
            g = ~g
        if r == 0:
            for i, smid in enumerate(facts["sms"]):
                sms.append({"tag": tag, "sm": smid, "group": "a" if g[i] else "b", "mean_cycles": round(float(m[:, i].mean()), 1)})
        a, b = m[r][g].mean(0), m[r][~g].mean(0)
        for p in range(m.shape[2]):
            pages.append({"tag": tag, "region": r, "region_base": hex(int(facts["base"], 16) + (r << 21)), "page": p,
                          "cycles_group_a": round(float(a[p]), 1), "cycles_group_b": round(float(b[p]), 1),
                          "class": int(a[p] > b[p]), "parity_12_13_16_17_19": bin(p & MASK).count("1") & 1,
                          "s1_over_s2": round(float(s[0] / s[1]), 1)})
for name, rows in (("pages", pages), ("sms", sms)):
    w = csv.DictWriter(open("%s-%s.csv" % (out, name), "w", newline=""), fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(len(pages), "page rows,", len(sms), "SM rows")
