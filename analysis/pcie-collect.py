#!/usr/bin/env python3
"""Collect the PCIe-probe measurements of every results/x-*/job.out into one table.

Usage: analysis/pcie-collect.py [--out results/x-pcie-summary]

Reads the JSON lines `probe/pcie_probe.py` printed (`mode`, `nvml-mode`, `facts`,
`numa`, `membench`) from each `results/x-*/job.out`, joins them by run, tag and mode, adds
the Slurm job id from `runs/registry.jsonl`, and writes:

- `modes.csv`: one row per (run, process tag, mode): the window, iterations, milliseconds
  per iteration, the host-to-device bytes the process copied, NVML's receive and transmit
  rates (MB/s, mean over the window; the maximum sample), the receive bytes per iteration
  beyond the copies, the card, the CPUs, the memory binding;
- `membench.csv`: the host memory copy rates of the harm job.

The raw NVML samples (timestamp, receive KB/s, transmit KB/s; about 25 per second) are in
`results/<run>/nvml/nvml-<tag>.txt.gz`.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/x-pcie-summary")
    a = ap.parse_args()
    jobs = {}
    if os.path.exists("runs/registry.jsonl"):
        for line in open("runs/registry.jsonl"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("job_name") and r.get("job_id"):
                jobs.setdefault(r["job_name"], set()).add(str(r["job_id"]))
    rows, mem = [], []
    for path in sorted(glob.glob("results/x-*/job.out")):
        run = path.split("/")[1]
        modes, nvml, facts, numa = {}, {}, {}, {}
        for line in open(path, errors="ignore"):
            i = line.find('{"probe"')
            if i < 0:
                continue
            try:
                r = json.loads(line[i:])
            except ValueError:
                continue
            kind = r["probe"]
            if kind == "mode":
                modes[(r["tag"], r["mode"])] = r
            elif kind == "nvml-mode":
                nvml[(r["tag"], r["mode"])] = r
            elif kind == "facts":
                facts[r["tag"]] = r
            elif kind == "numa":
                numa[r["tag"]] = r
            elif kind == "membench":
                mem.append({"run": run, "job": ";".join(sorted(jobs.get(run, []))), **{k: r[k] for k in ("tag", "start", "end", "GBps")}})
        for (tag, mode), w in modes.items():
            n = nvml.get((tag, mode), {})
            f = facts.get(tag, {})
            secs = w["end"] - w["start"]
            its = w["iterations"] / secs if w["iterations"] else 0.0
            rx = n.get("rx_MBps")
            rows.append({
                "run": run, "job": ";".join(sorted(jobs.get(run, []))), "tag": tag, "mode": mode,
                "start": w["start"], "end": w["end"], "iterations": w["iterations"],
                "ms_per_iteration": w.get("per_iteration_ms"), "h2d_MBps": w.get("h2d_MBps"),
                "rx_MBps": rx, "tx_MBps": n.get("tx_MBps"), "rx_max_MBps": n.get("rx_max_MBps"), "nvml_samples": n.get("samples"),
                "rx_beyond_copies_MB_per_iteration": round((rx - (w.get("h2d_MBps") or 0.0)) / its, 4) if rx is not None and its else None,
                "gpu_bdf": f.get("gpu_bdf"), "gpu_numa_node": f.get("gpu_numa_node"), "cpus": f.get("Cpus_allowed_list"),
                "membind": numa.get(tag, {}).get("membind"),
                "env": json.dumps({k: v for k, v in (f.get("env") or {}).items() if k.startswith(("CUDA_DEVICE", "CUDA_LAUNCH", "CUDNN_", "TORCH_"))}) or None,
            })
    os.makedirs(a.out, exist_ok=True)
    for name, data in (("modes.csv", rows), ("membench.csv", mem)):
        if not data:
            continue
        with open(os.path.join(a.out, name), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(data[0]))
            wr.writeheader()
            wr.writerows(data)
        print(f"{name}: {len(data)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
