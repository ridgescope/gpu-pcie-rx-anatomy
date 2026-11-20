#!/usr/bin/env python3
"""The prediction test of the parity rule.

This script was committed before the job it scores was submitted, and its scoring rules
have not been edited since.

The rule (job 5852): a stream reads the high value when the frame number of the page that
holds its semaphores has odd parity under the mask 0x13a78b3, the low value otherwise.

Usage: analysis/pcie-predict.py ROOT
ROOT holds pages.jsonl(.gz) (the watcher's snapshots), job.out (the probe's lines) and
windows.csv (analysis/pcie-windows.py: each window's rate and the channel of the stream
with the long kernels).

The prediction of a process is made from the first snapshot of that process, and counts only
if the snapshot's time precedes the start of the process's first window; otherwise the
process is reported as "no prediction" and none of its windows is scored. The semaphore
object is the 512-page /dev/nvidia* mapping at 0x204600000. Pages: our streams 0 to 2 at
offset 0x26000, 3 to 23 at 0x2f000, 24 to 45 at 0x2e000, 46 and later at 0x37000, the default
stream (windows "v-*") at 0x27000. A window is read as high when its rate exceeds 5,700 MB/s
with the long kernels on an even channel, 4,000 MB/s on an odd channel. Every window with a
prediction is scored.
"""
import collections, csv, gzip, json, os, sys

MASK = 0x13a78b3
root = sys.argv[1]
pj = os.path.join(root, "pages.jsonl")
if not os.path.exists(pj):
    pj += ".gz"
first = {}
for line in (gzip.open(pj, "rt") if pj.endswith(".gz") else open(pj)):
    s = json.loads(line)
    if "tag" in s and s["tag"] not in first:
        first[s["tag"]] = s
start = {}
for line in open(os.path.join(root, "job.out"), errors="replace"):
    i = line.find('{"probe": "mode"')
    if i >= 0:
        r = json.loads(line[i:]); start[r["tag"]] = min(start.get(r["tag"], 1e18), r["start"])

def page_of(mode):
    if mode.startswith("v-"):
        return 0x27
    k = int(mode[1:3])
    return 0x26 if k < 3 else 0x37 if k >= 46 else (0x2ff84 - (k - 3) * 0xc0) >> 12

rows, noped = [], []
for w in csv.DictReader(open(os.path.join(root, "windows.csv"))):
    tag, mode = w["tag"], w["mode"]
    s = first.get(tag)
    v = next((v for v in s["vmas"] if v["start"] == "0x204600000" and "pfn" in v), None) if s else None
    if v is None or s["t"] >= start[tag]:
        if tag not in noped:
            noped.append(tag)
        continue
    ch = next((int(tok.split(">ch")[1].split(":")[0]) for tok in w["layout"].split() if tok.endswith("L")), None)
    pfn = v["pfn"][page_of(mode)]
    pred = "H" if bin(pfn & MASK).count("1") & 1 else "L"
    seen = None if ch is None or not pfn else ("H" if float(w["rx_MBps"]) > (5700 if ch % 2 == 0 else 4000) else "L")
    rows.append({"tag": tag, "mode": mode, "page": hex(page_of(mode)), "pfn": hex(pfn), "snapshot_unix": round(s["t"], 1),
                 "first_window_unix": start[tag], "predicted": pred, "channel": ch, "rx_MBps": w["rx_MBps"],
                 "observed": seen, "agree": None if seen is None else pred == seen})
with open(os.path.join(root, "predictions.csv"), "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
scored = [r for r in rows if r["agree"] is not None]
pages = collections.defaultdict(list)
for r in scored:
    pages[(r["tag"], r["page"])].append(r["agree"])
print("processes with a prediction made before their first window:", len({r["tag"] for r in rows}), "; without:", noped)
print("windows scored:", len(scored), "; agree:", sum(r["agree"] for r in scored), "; windows not scorable (no channel or no frame):", len(rows) - len(scored))
print("pages:", len(pages), "; all windows agree:", sum(all(v) for v in pages.values()), "; all disagree:", sum(not any(v) for v in pages.values()),
      "; mixed:", sum(any(v) and not all(v) for v in pages.values()))
print("predicted high / low pages:", sum(1 for (t, p) in pages if any(r["predicted"] == "H" for r in scored if r["tag"] == t and r["page"] == p)),
      "/", sum(1 for (t, p) in pages if any(r["predicted"] == "L" for r in scored if r["tag"] == t and r["page"] == p)))
for r in scored:
    if not r["agree"]:
        print("  disagrees:", r)
