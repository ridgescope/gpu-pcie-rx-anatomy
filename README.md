# gpu-pcie-rx-anatomy

This repository contains the code and data for the article "Tracing 7 GB/s of Unexplained
PCIe Traffic to the H100's L2 Cache Partitions" on the RidgeScope blog. The article describes PCIe receive traffic
of several gigabytes a second on NVIDIA GPUs during work that makes no host-to-device copy.
On an H100 the traffic has two rates, and the article traces them to the L2 cache partition
of one 4 KiB page of host memory.

The scripts are the ones we used in the investigation. We wrote them for our own job setup
and have not packaged them to run elsewhere, so treat them as a reference for how each
number was produced.

## Machines

- An 8 x NVIDIA H100 80GB HBM3 node, Linux 6.11, NVIDIA open kernel module 570.211.01,
  PyTorch 2.11.0+cu128, CUDA 12.8, cuDNN 9.19. PCIe link 32 GT/s x16.
- One NVIDIA RTX A6000 in a KVM virtual machine, Linux 6.14, open kernel module 570.195.03,
  the same PyTorch and cuDNN. PCIe link 16 GT/s x16. The allocation rewrites were made on
  this machine only.

## Contents

`probe/pcie_probe.py` runs one kind of GPU work in each measurement window. For every
window it records the bytes the process copied and the PCIe receive and transmit rates
that NVML reports. It contains the workload the article calls "the pass" and its variants
on other streams and threads.

`probe/wait_probe.py` reproduces the traffic without cuDNN in one short script. One stream
runs a large matrix product, three streams wait for it with `wait_event`, and N small
kernels are queued behind each wait. It prints the PCIe receive throughput against N.
It needs PyTorch with CUDA and NVML, and no privileges.

`probe/l2_probe.py` is the L2 partition probe. It compiles one CUDA kernel with NVRTC and
launches it through the driver API. The kernel reads the id of its SM, reads a page in a
way that bypasses the L1 cache, and reads the cycle counter around the reads.

`ioctl-trace/nv_ioctl_trace.c` is an `LD_PRELOAD` interposer on `ioctl`. It logs the
objects a process asks the NVIDIA driver to create, how the process maps them, and the
UVM calls that give them GPU addresses. With `NV_IOCTL_REWRITE` it changes one allocation
request before it reaches the kernel. This mode is experimental and can crash the process
or the driver. We used it on the A6000 machine and never on the H100 node. Use it only on
an isolated GPU or virtual machine where a driver reload or a reboot is acceptable. The
probes that read driver objects depend on undocumented mappings that are specific to the
driver version.

`pagemap/watch-pages.py` reads `/proc/<pid>/pagemap` and related files for the probe's
processes and writes its output to a JSONL file. It needs root access.

`analysis/` contains the scripts that produce the tables in `data/` from the logs.

- `pcie-cmd.py` decodes GPFIFO entries and pushbuffer segments from a memory dump.
- `pcie-frames.py` fits the parity rule over GF(2).
- `pcie-predict.py` and `pcie-l2traffic.py` score the prediction tests.
- `pcie-queued.py` prints the receive throughput against the number of queued kernels by
  the partition of the semaphore page, from `queued-kernels.csv`.
- `pcie-l2.py`, `pcie-l2-summary.py` and `pcie-l2sem.py` analyse the arrays of the L2 probe.
- `pcie-article-figures.py` recreates the SVG figures of the article from three of the
  tables, numbers written in the script, and schematic diagrams.
- The remaining scripts build the tables of windows, channels and streams.

## Data

Each file is a table with a header row. Rates are in MB/s. `tag` names a process and
`mode` a measurement window inside it.

| File | What it holds | Section of the article |
|---|---|---|
| `all-windows.csv` | 2,901 measurement windows collected from the jobs, including superseded measurements: rates, iteration time, environment | all |
| `channels-per-process.csv`, `channel-table-vs-rate.csv` | the channels each process created, with runqueue bits, against its rate | The number of CUDA connections |
| `streams-on-channels.csv` | the channel of each stream's kernels, per process | The number of CUDA connections |
| `training-epochs.csv` | the Simulus training with the default connections and with two: seconds per phase and epoch. The means of the PCIe receive throughput per phase mix neighbouring phases, because DCGM reports it every 15 s. | not in the final text of the article |
| `training-streams-by-phase.csv` | the streams and channels of each training phase, from a kernel trace of two epochs | The number of CUDA connections |
| `a6000-interventions.csv` | the allocation rewrites on the A6000 with rates, loss and gradient absolute sum | What the GPU reads from the host |
| `one-process-streams-and-threads.csv` | the pass on different streams and threads of one process | Looking for the cause of the two levels |
| `sixteen-streams.csv`, `creation-order.csv`, `forty-eight-streams.csv` | the pass on each of many streams of one process | same |
| `semaphore-blocks.csv` | the offset and page of each stream's semaphore block | same |
| `page-frames-150.csv` | 150 semaphore pages: physical frame and rate | The physical address of the page |
| `frame-prediction-test.csv` | the prediction from page frames, 168 windows | same |
| `l2-device-memory-h100.csv`, `l2-sm-groups-h100.csv` | read times of device memory from two SM groups, per page | The two partitions of the L2 cache |
| `l2-device-memory-a6000.csv` | the same probe on the A6000 | same |
| `l2-semaphore-pages.csv` | 4,096 semaphore pages: frame, read times, partition, parity | same |
| `l2-traffic-same-process.csv` | traffic and partition of the same pages in one process, 168 windows | same |
| `wait-probe-h100.csv` | `probe/wait_probe.py` on the H100: 24 curves of receive throughput against the queued kernels from eight runs, and one run without waiting streams | How the partition changes the amount of traffic |
| `queued-kernels.csv` | 640 windows of 20 processes: receive throughput against the number of kernels queued behind a wait, by the partition of the awaited semaphore page, for a word of our own in device or host memory, and for the producer alone | How the partition changes the amount of traffic |

The frame mask of the parity rule is `0x13a78b3` (physical address bits 12, 13, 16, 17, 19,
23 to 26, 29, 31 to 33 and 36).

### Reading the tables

- MB/s means decimal megabytes per second. Read times are GPU clock cycles.
- Process tags are local to an experiment. An empty job id means the row was not matched to
  a job.
- The `gpu` column names the card: `h100-0` to `h100-7` for the eight cards of the node and
  `a6000` for the second machine. The collection script writes PCI addresses in this column,
  and we replaced them with these names.
- In the `layout` columns, `s7>ch0:22686L` means stream 7 ran 22,686 kernels on channel 0.
  The `L` marks the stream that carries kernels longer than 1 ms.
- Rate classes: the fit in `page-frames-150.csv` calls a window high above 4,800 MB/s. The
  two prediction tests call a window high above 5,700 MB/s when the long kernels are on an
  even channel and above 4,000 MB/s when they are on an odd channel.
- Semaphore pages: the default stream's block is in the page at offset 0x27000 of the
  semaphore object. The blocks of the streams we created are at 0x26000 (streams 0 to 2),
  0x2f000 (3 to 23), 0x2e000 (24 to 45) and 0x37000 (46 and later).
- L2 partitions: group `a` is the group of SMs that contains SM 0. A page belongs to the
  partition from which it reads faster; the tables call this partition the page's `home`.
  Pages of odd parity belong to partition `b`, and their streams have the high traffic rate.

### Rows that need care

- In `all-windows.csv`, the copy-interference measurements of run `x-pcie-harm` on one card
  are invalid, because the copy windows missed the pass. Use `x-pcie-harm-2` for that
  comparison. The eight-card measurements of `x-pcie-harm` are valid. The job column of
  these rows holds both job ids.
- In `a6000-interventions.csv`, the row `semnc1` is a process whose rewrite did not apply.
  It counts as an unchanged process, which is why the article reports six of them.
- In `page-frames-150.csv`, the pages `(f-6, 0x27)` and `(f-18, 0x2f)` have windows of both
  classes. The table keeps their majority class, and the fit leaves both pages out.
- `all-windows.csv` holds 20 of the 92 windows measured on the A6000.

## Not included

- The raw logs are not included. These are the NVML samples, the kernel traces, the memory
  dumps and the arrays of the L2 probe.
- The CUPTI subscriber that logged the hardware channel of each kernel is part of the
  RidgeScope agent and is not included. It logs the field
  `CUpti_ActivityKernel9.channelID`.
- The job scripts for our cluster are not included.
- Some evidence the article cites is not in the tables: the per-epoch totals of the
  earlier training runs and the DCGM series, the cuDNN API logs and the comparison of metric names,
  the allocation and mapping logs and the decoded command streams, 72 of the 92 A6000
  windows, the host CPU and interrupt series, and the outputs of the shuffle tests.

## Notes

- The docstrings of `pcie-predict.py` and `pcie-l2traffic.py` state that each script was
  committed before the job it scores. This refers to our internal repository. Its history
  is not part of this repository.
- The job numbers and run names in the files are our own identifiers.
- Addresses such as `0x204600000` in the tables are virtual addresses of the process or
  the GPU. The `pfn` values are physical frame numbers of host pages.

## Licence

The code in `probe/`, `ioctl-trace/`, `pagemap/` and `analysis/` is released under the MIT
licence (see `LICENSE`). The tables in `data/` are released under the Creative Commons
Attribution 4.0 International licence (CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/).

The structure offsets and method numbers in `ioctl-trace/nv_ioctl_trace.c` and
`analysis/pcie-cmd.py` were read from the headers of NVIDIA's open GPU kernel modules
(tag 570.211.01), which are published under the MIT licence.
