#!/usr/bin/env python3
"""The channel table of every process of job x-pcie-chan against its receive rate.

Reads results/x-pcie-chan/job.out (the probe's JSON lines) and
results/x-pcie-chan/ioctl/ioctl-<tag>-<pid>.log (ioctl-trace/nv_ioctl_trace.c), and
writes results/x-pcie-chan/channels.csv (one row per channel) and processes.csv (one row
per process). Usage: analysis/pcie-chan.py [results/x-pcie-chan]
"""
import collections, csv, glob, json, os, re, sys

root = sys.argv[1] if len(sys.argv) > 1 else "results/x-pcie-chan"
rate = {}
for line in open(os.path.join(root, "job.out"), errors="replace"):
    i = line.find('{"probe"')
    if i < 0:
        continue
    try:
        r = json.loads(line[i:])
    except ValueError:
        continue
    if r.get("mode") != "vgg-1":
        continue
    d = rate.setdefault(r["tag"], {})
    if r["probe"] == "mode":
        d["ms"] = r["per_iteration_ms"]
    elif r["probe"] == "nvml-mode":
        d["rx"], d["tx"] = r["rx_MBps"], r["tx_MBps"]

kv = re.compile(r"(\w+)=(\S+)")
files = collections.defaultdict(list)
for f in glob.glob(os.path.join(root, "ioctl", "ioctl-*.log")):
    tag = re.match(r"ioctl-(.*)-\d+\.log$", os.path.basename(f)).group(1)
    files[tag].append(f)

chan_rows, proc_rows = [], []
for tag in sorted(files, key=lambda t: (t.split("-")[-1], t)):
    f = max(files[tag], key=os.path.getsize)  # the probe itself; the smaller file is the NVML poller
    tsg, share, chans, copies, ctrls = {}, {}, [], {}, collections.Counter()
    for line in open(f):
        d = dict(kv.findall(line))
        if line.startswith("C<"):
            ctrls[d["cmd"]] += 1
        if not line.startswith("A<") or d.get("status") != "0":
            continue
        if " TSG " in line:
            tsg[d["hNew"]] = int(d["eng"])
        elif " CTXSHARE " in line:
            share[d["hNew"]] = (int(d["flags"], 16), int(d["subctx"]))
        elif " CHANNEL " in line:
            chans.append(d)
        elif " COPY " in line:
            copies[d["hParent"]] = int(d["engine"])
    t0 = None
    for n, c in enumerate(chans):
        fl = int(c["flags"], 16)
        t = float(line.split()[1]) if False else None
        row = {"tag": tag, "n": n, "tid": c["tid"], "tsg": c["hParent"], "tsg_engine": tsg.get(c["hParent"], ""),
               "handle": c["hNew"], "flags": c["flags"], "runqueue": (fl >> 4) & 1, "thread": (fl >> 28) & 3,
               "entries": c["entries"], "gpoff": c["gpoff"], "ctxshare": c["ctxshare"],
               "subctx_flags": share.get(c["ctxshare"], ("", ""))[0], "subctx": share.get(c["ctxshare"], ("", ""))[1],
               "userd": c["userd"], "userdoff": c["userdoff"], "eng": c["eng"], "cid": c["cid"],
               "copy_engine": copies.get(c["hNew"], "")}
        chan_rows.append(row)
    comp = [r for r in chan_rows if r["tag"] == tag and r["tsg_engine"] not in (11, 12, 13, 14, 15, 16, 17, 18, 19, 20)]
    r = rate.get(tag, {})
    proc_rows.append({"tag": tag, "rx_MBps": r.get("rx", ""), "tx_MBps": r.get("tx", ""), "ms": r.get("ms", ""),
                      "channels": len(chans), "tsgs": len(tsg), "ctxshares": len(share),
                      "tsg_engines": " ".join(str(v) for v in tsg.values()),
                      "main_tsg_channels": len(comp),
                      "main_runqueues": "".join(str(x["runqueue"]) for x in comp),
                      "main_threads": "".join(str(x["thread"]) for x in comp),
                      "main_subctx": " ".join(str(x["subctx"]) for x in comp),
                      "main_copy": " ".join(str(x["copy_engine"]) for x in comp),
                      "main_tids": len({x["tid"] for x in comp}),
                      "controls": " ".join(f"{k}:{v}" for k, v in sorted(ctrls.items()))})
for name, rows in (("channels.csv", chan_rows), ("processes.csv", proc_rows)):
    if rows:
        with open(os.path.join(root, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
for p in proc_rows:
    print(f'{p["tag"]:10} rx={p["rx_MBps"]!s:>8} ms={p["ms"]!s:>7} ch={p["channels"]:3} tsg={p["tsgs"]:2} '
          f'main={p["main_tsg_channels"]:2} rq={p["main_runqueues"]} thr={p["main_threads"]} copy=[{p["main_copy"]}] sub=[{p["main_subctx"]}]')
