#!/usr/bin/env python3
"""The SVG figures of the article, drawn from the tables in data/.
Run from the repository root: analysis/pcie-article-figures.py OUT.json writes one SVG string
per figure.
"""
import csv, json, random, sys
OUT = sys.argv[1]
F = '"Iowan Old Style", "Charter", "Georgia", serif'
def style(sfx, extra=""):
    return f"""    <style>
      .lbl{sfx} {{ font: 12px {F}; fill: #1b1b1b; }}
      .lbl{sfx}-b {{ font: 13px {F}; fill: #1b1b1b; font-weight: 700; }}
      .sm{sfx} {{ font: 10.5px {F}; fill: #6b6b6b; }}
      .ax{sfx} {{ stroke: #c9c9c9; stroke-width: 1; }}
{extra}    </style>"""
figs = {}

# --- A: receive throughput against copies, per epoch
s = "A"
h = lambda gb: gb / 40 * 150
figs["gauge"] = f"""<svg viewBox="0 0 720 250" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Bar chart of gigabytes per training epoch. Host-to-device copies counted by CUPTI: 15.5 GB. PCIe receive traffic estimated from the receive throughput that DCGM reports, in other runs of the same schedule: 34.5 to 38.6 GB.">
{style(s, f"      .cp{s} {{ fill: #6b8fb0; }}\n      .gg{s} {{ fill: #9c4040; }}\n      .gr{s} {{ fill: #9c4040; opacity: 0.45; }}\n")}
    <text x="360" y="22" class="lbl{s}-b" text-anchor="middle">One training epoch: bytes copied to the GPU against bytes received over PCIe</text>
    <line class="ax{s}" x1="150" y1="200" x2="600" y2="200" />
    <line class="ax{s}" x1="150" y1="{200-h(20):.1f}" x2="600" y2="{200-h(20):.1f}" />
    <line class="ax{s}" x1="150" y1="{200-h(40):.1f}" x2="600" y2="{200-h(40):.1f}" />
    <text x="144" y="204" class="sm{s}" text-anchor="end">0</text>
    <text x="144" y="{204-h(20):.1f}" class="sm{s}" text-anchor="end">20 GB</text>
    <text x="144" y="{204-h(40):.1f}" class="sm{s}" text-anchor="end">40 GB</text>
    <rect class="cp{s}" x="220" y="{200-h(15.5):.1f}" width="110" height="{h(15.5):.1f}" />
    <rect class="gg{s}" x="420" y="{200-h(34.5):.1f}" width="110" height="{h(34.5):.1f}" />
    <rect class="gr{s}" x="420" y="{200-h(38.6):.1f}" width="110" height="{h(38.6)-h(34.5):.1f}" />
    <text x="275" y="{194-h(15.5):.1f}" class="lbl{s}" text-anchor="middle">15.5 GB</text>
    <text x="475" y="{194-h(38.6):.1f}" class="lbl{s}" text-anchor="middle">34.5 to 38.6 GB</text>
    <text x="275" y="220" class="sm{s}" text-anchor="middle">host-to-device copies</text>
    <text x="275" y="234" class="sm{s}" text-anchor="middle">(CUPTI memcpy records)</text>
    <text x="475" y="220" class="sm{s}" text-anchor="middle">PCIe receive</text>
    <text x="475" y="234" class="sm{s}" text-anchor="middle">(receive throughput × epoch time)</text>
  </svg>"""

# --- B: rate against connections
s = "B"
conn = {"1": [34, 31, 30, 28, 28, 29], "2": [36, 37, 38, 37, 36, 37], "3": [5616, 5932, 3225, 5740, 5479, 3272],
        "4": [4451, 6947, 7116, 4410, 6387, 6916], "8": [7927, 7888, 4456, 7578, 4606, 7279],
        "16": [6622, 4417, 4440, 4149, 7679, 6983], "32": [6904, 6956, 7367, 7060, 7180, 6781]}
x0, x1, y0, y1 = 80, 680, 250, 50
ys = lambda v: y0 - v / 8000 * (y0 - y1)
parts = []
for i, (k, vals) in enumerate(conn.items()):
    cx = x0 + (i + 0.5) * (x1 - x0) / len(conn)
    parts.append(f'    <text x="{cx:.0f}" y="268" class="sm{s}" text-anchor="middle">{k}{" (default)" if k == "8" else ""}</text>')
    for j, v in enumerate(sorted(vals)):
        parts.append(f'    <circle class="{"lo" if v < 1000 else "pt"}{s}" cx="{cx - 20 + j * 8:.1f}" cy="{ys(v):.1f}" r="4" />')
grid = "".join(f'    <line class="ax{s}" x1="{x0}" y1="{ys(v):.1f}" x2="{x1}" y2="{ys(v):.1f}" />\n    <text x="{x0-6}" y="{ys(v)+4:.1f}" class="sm{s}" text-anchor="end">{v//1000 if v else 0}{" GB/s" if v == 8000 else ""}</text>\n' for v in (0, 2000, 4000, 6000, 8000))
figs["connections"] = f"""<svg viewBox="0 0 720 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Dot plot of PCIe receive throughput against CUDA_DEVICE_MAX_CONNECTIONS, six processes per setting. At 1 and 2 connections every process reads about 30 megabytes per second. From 3 connections on, the processes read gigabytes per second and fall into a low group and a high group; at 32 all six are in the high group.">
{style(s, f"      .pt{s} {{ fill: #9c4040; }}\n      .lo{s} {{ fill: #6b8fb0; }}\n")}
    <text x="360" y="22" class="lbl{s}-b" text-anchor="middle">PCIe receive throughput of the same pass, by CUDA_DEVICE_MAX_CONNECTIONS</text>
    <text x="360" y="38" class="sm{s}" text-anchor="middle">six processes per setting, one dot each; iteration time 24.2 to 24.6 ms in all of them</text>
{grid}{chr(10).join(parts)}
    <text x="380" y="288" class="sm{s}" text-anchor="middle">CUDA_DEVICE_MAX_CONNECTIONS</text>
  </svg>"""

# --- C: channels on runqueues
s = "C"
def chan(x, y, label, sub, cls):
    return f'    <rect class="{cls}{s}" x="{x}" y="{y}" width="92" height="40" rx="3" />\n    <text x="{x+46}" y="{y+17}" class="lbl{s}" text-anchor="middle">{label}</text>\n    <text x="{x+46}" y="{y+32}" class="sm{s}" text-anchor="middle">{sub}</text>\n'
c = f"""<svg viewBox="0 0 720 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Diagram of CUDA streams on compute channels and runqueues. With two connections, channel 0 on runqueue 0 carries the long-kernel stream and one helper, channel 1 on runqueue 1 the other two helpers. With three connections, channel 0 and channel 2 both sit on runqueue 0: a helper stream's channel shares a runqueue with the channel of the long kernels.">
{style(s, f"      .rq{s} {{ fill: #f2f2ee; }}\n      .lk{s} {{ fill: #c9d8e6; stroke: #6b8fb0; }}\n      .hp{s} {{ fill: #ffffff; stroke: #b9b9b4; }}\n      .hot{s} {{ fill: #f0d9d6; stroke: #9c4040; }}\n")}
    <text x="180" y="24" class="lbl{s}-b" text-anchor="middle">2 connections: no traffic</text>
    <text x="540" y="24" class="lbl{s}-b" text-anchor="middle">3 connections: 3 to 6 GB/s</text>
"""
for ox, n in ((20, 2), (380, 3)):
    c += f'    <rect class="rq{s}" x="{ox}" y="44" width="320" height="100" />\n    <text x="{ox+10}" y="62" class="sm{s}">runqueue 0</text>\n'
    c += f'    <rect class="rq{s}" x="{ox}" y="160" width="320" height="100" />\n    <text x="{ox+10}" y="178" class="sm{s}">runqueue 1</text>\n'
    if n == 2:
        c += chan(ox + 20, 84, "channel 0", "stream 7 (long), 31", "lk") + chan(ox + 20, 200, "channel 1", "streams 13, 30", "hp")
    else:
        c += chan(ox + 20, 84, "channel 0", "stream 7 (long)", "lk") + chan(ox + 130, 84, "channel 2", "stream 30", "hot") + chan(ox + 20, 200, "channel 1", "streams 13, 31", "hp")
c += f"""    <text x="360" y="284" class="sm{s}" text-anchor="middle">stream 7 carries cuDNN's long kernels; 13, 30 and 31 are its FFT helper streams and wait on it through semaphores</text>
  </svg>"""
figs["channels"] = c

# --- G: where the objects live
s = "G"
figs["objects"] = f"""<svg viewBox="0 0 720 270" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Diagram of the three memory objects behind a CUDA channel. In host memory: the pushbuffer, 56 MiB, holding the commands, not cached by the GPU; and the semaphore object, 2 MiB of 4 KiB pages, cached by the GPU in its L2. In device memory: the GPFIFO and USERD, which point at pushbuffer segments. The GPU fetches the pushbuffer across PCIe.">
{style(s, f"      .host{s} {{ fill: #f2f2ee; }}\n      .dev{s} {{ fill: #e6eef5; }}\n      .box{s} {{ fill: #ffffff; stroke: #b9b9b4; }}\n      .hot{s} {{ fill: #f0d9d6; stroke: #9c4040; }}\n      .arr{s} {{ stroke: #9c4040; stroke-width: 2; fill: none; }}\n")}
    <rect class="host{s}" x="20" y="40" width="330" height="200" />
    <rect class="dev{s}" x="390" y="40" width="310" height="200" />
    <text x="30" y="58" class="lbl{s}-b">host memory</text>
    <text x="400" y="58" class="lbl{s}-b">GPU</text>
    <rect class="hot{s}" x="40" y="74" width="290" height="62" rx="3" />
    <text x="52" y="94" class="lbl{s}">pushbuffer: the commands of every channel</text>
    <text x="52" y="110" class="sm{s}">56 MiB at 8 connections, not GPU-cached</text>
    <text x="52" y="124" class="sm{s}">kernel launches, with semaphore acquires between them</text>
    <rect class="box{s}" x="40" y="154" width="290" height="62" rx="3" />
    <text x="52" y="174" class="lbl{s}">semaphore object: one 4 KiB page per ~21 streams</text>
    <text x="52" y="190" class="sm{s}">2 MiB of unrelated 4 KiB pages, GPU-cached (in L2)</text>
    <text x="52" y="204" class="sm{s}">a stream's block of eight semaphores lives in one page</text>
    <rect class="box{s}" x="410" y="74" width="270" height="62" rx="3" />
    <text x="422" y="94" class="lbl{s}">host unit (PBDMA), selected by the runqueue</text>
    <text x="422" y="110" class="sm{s}">fetches pushbuffer segments, executes host methods,</text>
    <text x="422" y="124" class="sm{s}">switches channel when an acquire fails</text>
    <rect class="box{s}" x="410" y="154" width="270" height="62" rx="3" />
    <text x="422" y="174" class="lbl{s}">GPFIFO and USERD (device memory)</text>
    <text x="422" y="190" class="sm{s}">GPFIFO: segment descriptors; USERD: progress;</text>
    <text x="422" y="204" class="sm{s}">neither is read from the host</text>
    <path class="arr{s}" d="M330 100 L404 100" /><path class="arr{s}" d="M396 94 L406 100 L396 106" />
    <text x="367" y="90" class="sm{s}" text-anchor="middle">PCIe</text>
    <text x="360" y="258" class="sm{s}" text-anchor="middle">read from the process's own mappings on both cards; the red object is the one whose re-fetching is the traffic</text>
  </svg>"""

# --- D: sixteen streams of one run, each measured twice
s = "D"
def long_ch(r):
    return next(int(t.split(">ch")[1].split(":")[0]) for t in r["layout"].split() if t.endswith("L"))
rows = [r for r in csv.DictReader(open("data/sixteen-streams.csv")) if r["tag"] == "sa-1" and r["mode"].startswith("x")]
x0, x1, y0, y1 = 70, 690, 250, 56
xs = lambda k: x0 + (k + 0.5) * (x1 - x0) / 16
ys = lambda v: y0 - v / 8000 * (y0 - y1)
bg = [f'    <rect class="{c}{s}" x="{xs(a)-17:.1f}" y="{y1}" width="{xs(b)-xs(a)+34:.1f}" height="{y0-y1}" />' for a, b, c in ((0, 2, "bandL"), (3, 15, "bandH"))]
parts = []
for r in rows:
    k = int(r["mode"][1:3]); v = float(r["rx_MBps"])
    parts.append(f'    <circle class="{"ev" if long_ch(r) % 2 == 0 else "od"}{s}" cx="{xs(k) + (5 if r["mode"].endswith("b") else -5):.1f}" cy="{ys(v):.1f}" r="3.6" />')
grid = "".join(f'    <line class="ax{s}" x1="{x0}" y1="{ys(v):.1f}" x2="{x1}" y2="{ys(v):.1f}" />\n    <text x="{x0-6}" y="{ys(v)+4:.1f}" class="sm{s}" text-anchor="end">{v//1000}{" GB/s" if v == 8000 else ""}</text>\n' for v in (0, 2000, 4000, 6000, 8000))
xticks = chr(10).join(f'    <text x="{xs(k):.1f}" y="268" class="sm{s}" text-anchor="middle">{k}</text>' for k in range(16))
dot_style = lambda s: f"      .ev{s} {{ fill: #9c4040; }}\n      .od{s} {{ fill: #ffffff; stroke: #9c4040; stroke-width: 1.4; }}\n      .bandL{s} {{ fill: #f2f2ee; }}\n      .bandH{s} {{ fill: #f6e9e7; }}\n"
legend = lambda s, y: f'    <circle class="ev{s}" cx="206" cy="{y}" r="4" /><text x="218" y="{y+4}" class="lbl{s}">on an even-numbered channel</text>\n    <circle class="od{s}" cx="406" cy="{y}" r="4" /><text x="418" y="{y+4}" class="lbl{s}">on an odd-numbered channel</text>'
figs["streams16"] = f"""<svg viewBox="0 0 720 324" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Dot plot of the PCIe receive throughput on each of sixteen CUDA streams of one run, by the order the streams were created, two measurements per stream. Streams 0 to 2 are at the low level and streams 3 to 15 at the high level. Within each group a stream receives less on an odd-numbered channel than on an even-numbered one.">
{style(s, dot_style(s))}
    <text x="360" y="22" class="lbl{s}-b" text-anchor="middle">PCIe receive throughput on each of sixteen streams of one run</text>
    <text x="360" y="38" class="sm{s}" text-anchor="middle">two measurements per stream, one dot each</text>
{chr(10).join(bg)}
{grid}{chr(10).join(parts)}
{xticks}
    <text x="380" y="286" class="sm{s}" text-anchor="middle">stream, in order of creation</text>
{legend(s, 306)}
  </svg>"""

# --- P: 48 streams by the page that holds their semaphores, three runs
s = "P"
rows = list(csv.DictReader(open("data/forty-eight-streams.csv")))
pages = [(0, 2), (3, 23), (24, 45), (46, 47)]
names = ("0x26000", "0x2f000", "0x2e000", "0x37000", "0x27000")
held = ("streams 0 to 2", "streams 3 to 23", "streams 24 to 45", "streams 46 and 47", "the default stream")
x0, pw, ph = 70, 620, 84; cw = pw / 5
tops = [62, 166, 270]
bg, parts, grid = [], [], ""
for j, tag in enumerate(("p-1", "p-2", "p-3")):
    y1 = tops[j]; y0 = y1 + ph
    ys = lambda v: y0 - v / 8800 * ph
    cols = [[] for _ in range(5)]
    for r in rows:
        if r["tag"] != tag:
            continue
        v = float(r["rx_MBps"]); even = long_ch(r) % 2 == 0
        if r["mode"].startswith("x"):
            k = int(r["mode"][1:3])
            cols[next(i for i, (a, b) in enumerate(pages) if a <= k <= b)].append((k, v, even))
        elif r["mode"].startswith("v"):
            cols[4].append((int(r["mode"][2:]), v, even))
    parts.append(f'    <text x="{x0}" y="{y1-6}" class="lbl{s}">Run {j+1}</text>')
    parts.append(f'    <rect class="fr{s}" x="{x0}" y="{y1}" width="{pw}" height="{ph}" />')
    for v in (0, 2000, 4000, 6000, 8000):
        grid += f'    <line class="ax{s}" x1="{x0}" y1="{ys(v):.1f}" x2="{x0+pw}" y2="{ys(v):.1f}" />\n'
        if v % 4000 == 0:
            grid += f'    <text x="{x0-6}" y="{ys(v)+4:.1f}" class="sm{s}" text-anchor="end">{v//1000}{" GB/s" if v == 8000 and j == 0 else ""}</text>\n'
    for i, col in enumerate(cols):
        cx = x0 + (i + 0.5) * cw
        ev = sorted(v for _, v, e in col if e)
        bg.append(f'    <rect class="{"bandH" if ev[len(ev) // 2] > 5500 else "bandL"}{s}" x="{cx - cw/2 + 3:.1f}" y="{y1}" width="{cw - 6:.1f}" height="{ph}" />')
        ks = sorted({k for k, _, _ in col}); n = max(len(ks) - 1, 1)
        for k, v, e in col:
            dx = (ks.index(k) / n - 0.5) * (cw - 30) if len(ks) > 1 else 0
            parts.append(f'    <circle class="{"ev" if e else "od"}{s}" cx="{cx + dx:.1f}" cy="{ys(v):.1f}" r="3" />')
        if j == 2:
            parts.append(f'    <text x="{cx:.1f}" y="{y0+17}" class="lbl{s}" text-anchor="middle">{names[i]}</text>')
            parts.append(f'    <text x="{cx:.1f}" y="{y0+31}" class="sm{s}" text-anchor="middle">{held[i]}</text>')
figs["pages"] = f"""<svg viewBox="0 0 720 442" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Three dot plots, one per run of the script with 48 CUDA streams. The horizontal axis lists five pages of the area that holds the semaphores, by their offset in the area: 0x26000 with streams 0 to 2, 0x2f000 with streams 3 to 23, 0x2e000 with streams 24 to 45, 0x37000 with streams 46 and 47, and 0x27000 with the default stream. In every run all the streams of one page are at one level. The same page is at the low level in one run and at the high level in another.">
{style(s, dot_style(s) + f"      .fr{s} {{ fill: none; stroke: #b9b9b4; stroke-width: 1; }}\n")}
    <text x="360" y="22" class="lbl{s}-b" text-anchor="middle">PCIe receive throughput of 48 streams, by the page that holds their semaphores</text>
{chr(10).join(bg)}
{grid}{chr(10).join(parts)}
    <text x="380" y="404" class="sm{s}" text-anchor="middle">offset of the page in the area that holds the semaphores</text>
{legend(s, 424)}
  </svg>"""

# --- E: partition maps
s = "E"
LOW = sum(1 << b for b in (0, 1, 4, 5, 7))
sem = {int(r["page"]): r["home"] for r in csv.DictReader(open("data/l2-semaphore-pages.csv")) if r["tag"] == "sem-1"}
def grid_map(ox, oy, f):
    d = {0: [], 1: []}
    for p in range(512):
        d[1 if f(p) else 0].append(f"M{ox + (p % 32) * 9.6:.1f} {oy + (p // 32) * 9.6:.1f}h8.6v8.6h-8.6z")
    return f'    <path class="pa{s}" d="{"".join(d[0])}" />\n    <path class="pb{s}" d="{"".join(d[1])}" />'
figs["partitions"] = f"""<svg viewBox="0 0 720 250" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Two maps of 512 consecutive 4 KiB pages, coloured by the L2 partition each page is at home in, as measured by read times. Left: a 2 MiB region of device memory, a regular pattern given by the parity of address bits 12, 13, 16, 17 and 19. Right: the semaphore object of one process, an irregular pattern, because its pages are unrelated pages of host memory; each page still follows the parity of its own physical address.">
{style(s, f"      .pa{s} {{ fill: #6b8fb0; }}\n      .pb{s} {{ fill: #9c4040; }}\n")}
    <text x="180" y="22" class="lbl{s}-b" text-anchor="middle">Device memory, one 2 MiB region</text>
    <text x="540" y="22" class="lbl{s}-b" text-anchor="middle">The semaphore object of one process</text>
{grid_map(26, 36, lambda p: bin(p & LOW).count("1") & 1)}
{grid_map(386, 36, lambda p: sem[p] == "b")}
    <text x="180" y="208" class="sm{s}" text-anchor="middle">512 pages in address order, one contiguous region</text>
    <text x="540" y="208" class="sm{s}" text-anchor="middle">512 pages in address order; scattered physical frames</text>
    <rect class="pa{s}" x="236" y="224" width="9" height="9" /><text x="250" y="233" class="sm{s}">at home in partition a</text>
    <rect class="pb{s}" x="386" y="224" width="9" height="9" /><text x="400" y="233" class="sm{s}">at home in partition b</text>
  </svg>"""

# --- F: traffic by home partition, one process
s = "F"
rows = list(csv.DictReader(open("data/l2-traffic-same-process.csv")))
rnd = random.Random(3)
cols = [("a", 0, "home a, even channel"), ("a", 1, "home a, odd channel"), ("b", 0, "home b, even channel"), ("b", 1, "home b, odd channel")]
x0, x1, y0, y1 = 80, 680, 250, 50
ys = lambda v: y0 - v / 10000 * (y0 - y1)
parts = []
for i, (h, par, name) in enumerate(cols):
    cx = x0 + (i + 0.5) * (x1 - x0) / 4
    pts = [float(r["rx_MBps"]) for r in rows if r["home"] == h and int(r["channel"]) % 2 == par]
    for v in pts:
        parts.append(f'    <circle class="{"pa" if h == "a" else "pb"}{s}" cx="{cx + rnd.uniform(-46, 46):.1f}" cy="{ys(v):.1f}" r="3" />')
    parts.append(f'    <text x="{cx:.0f}" y="268" class="sm{s}" text-anchor="middle">{name}</text>')
    parts.append(f'    <text x="{cx:.0f}" y="282" class="sm{s}" text-anchor="middle">{len(pts)} windows</text>')
grid = "".join(f'    <line class="ax{s}" x1="{x0}" y1="{ys(v):.1f}" x2="{x1}" y2="{ys(v):.1f}" />\n    <text x="{x0-6}" y="{ys(v)+4:.1f}" class="sm{s}" text-anchor="end">{v//1000}{" GB/s" if v == 10000 else ""}</text>\n' for v in (0, 2000, 4000, 6000, 8000, 10000))
figs["traffic"] = f"""<svg viewBox="0 0 720 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Strip plot of the PCIe receive throughput of 168 measurement windows in twelve processes, grouped by the L2 partition in which the stream's semaphore page is at home and by the parity of the channel. Windows of pages at home in partition a read 3.0 to 4.8 gigabytes per second; windows of pages at home in partition b read 5.0 to 9.2. The groups do not overlap within a channel parity.">
{style(s, f"      .pa{s} {{ fill: #6b8fb0; opacity: 0.85; }}\n      .pb{s} {{ fill: #9c4040; opacity: 0.85; }}\n")}
    <text x="360" y="22" class="lbl{s}-b" text-anchor="middle">Receive throughput of a stream, by the L2 partition of its semaphore page</text>
    <text x="360" y="38" class="sm{s}" text-anchor="middle">168 windows in twelve processes; partition measured afterwards in the same process, by read times alone</text>
{grid}{chr(10).join(parts)}
  </svg>"""

# The diagram of the logging library between the CUDA library and the kernel driver.
F = '"Iowan Old Style", "Charter", "Georgia", serif'
def m_style(s):
    return f"""    <style>
      .lbl{s} {{ font: 12px {F}; fill: #1b1b1b; }}
      .lbl{s}-b {{ font: 13px {F}; fill: #1b1b1b; font-weight: 700; }}
      .sm{s} {{ font: 10.5px {F}; fill: #6b6b6b; }}
      .box{s} {{ fill: #ffffff; stroke: #b9b9b4; }}
      .ours{s} {{ fill: #f0d9d6; stroke: #9c4040; }}
      .zone{s} {{ fill: #f2f2ee; }}
      .dev{s} {{ fill: #e6eef5; }}
      .arr{s} {{ stroke: #1b1b1b; stroke-width: 1.4; fill: none; }}
      .red{s} {{ stroke: #9c4040; stroke-width: 2; fill: none; }}
      .dash{s} {{ stroke: #b9b9b4; stroke-width: 1; stroke-dasharray: 4 3; fill: none; }}
    </style>"""
def m_box(s, x, y, w, h, title, subs=(), cls="box"):
    o = f'    <rect class="{cls}{s}" x="{x}" y="{y}" width="{w}" height="{h}" rx="3" />\n'
    o += f'    <text x="{x + w/2:.0f}" y="{y + 18}" class="lbl{s}" text-anchor="middle">{title}</text>\n'
    for i, t in enumerate(subs):
        o += f'    <text x="{x + w/2:.0f}" y="{y + 33 + i*13}" class="sm{s}" text-anchor="middle">{t}</text>\n'
    return o
def m_harr(s, x1, x2, y, cls="arr"):
    return f'    <path class="{cls}{s}" d="M{x1} {y} L{x2} {y}" /><path class="{cls}{s}" d="M{x2-8} {y-5} L{x2} {y} L{x2-8} {y+5}" />\n'
def m_varr(s, x, y1, y2, cls="arr"):
    return f'    <path class="{cls}{s}" d="M{x} {y1} L{x} {y2}" /><path class="{cls}{s}" d="M{x-5} {y2-8} L{x} {y2} L{x+5} {y2-8}" />\n'
# --- L: where the logging library sits
s = "V3"
v = f"""<svg viewBox="0 0 720 330" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Diagram of layers from top to bottom. Inside the process: the PyTorch script, cuDNN, the CUDA library, and our library, which is loaded with LD_PRELOAD. Below the process: the NVIDIA kernel driver, then the GPU. Every ioctl request from the CUDA library passes through our library, which writes it to a log and can change it before it reaches the kernel driver.">
{m_style(s)}
    <text x="360" y="22" class="lbl{s}-b" text-anchor="middle">Where our library sits</text>
    <rect class="zone{s}" x="60" y="36" width="300" height="196" />
    <text x="70" y="52" class="sm{s}">the process</text>
"""
v += m_box(s, 110, 58, 200, 30, "PyTorch script")
v += m_box(s, 110, 98, 200, 30, "cuDNN")
v += m_box(s, 110, 138, 200, 30, "CUDA library")
v += m_box(s, 110, 186, 200, 34, "our library (LD_PRELOAD)", (), "ours")
v += m_varr(s, 210, 88, 97) + m_varr(s, 210, 128, 137) + m_varr(s, 210, 168, 185) + m_varr(s, 210, 220, 251)
v += f'    <text x="218" y="180" class="sm{s}">ioctl</text>\n    <text x="218" y="242" class="sm{s}">ioctl</text>\n'
v += m_box(s, 110, 252, 200, 30, "NVIDIA kernel driver")
v += m_varr(s, 210, 282, 291)
v += m_box(s, 110, 292, 200, 30, "GPU", (), "dev")
v += f'    <path class="red{s}" d="M312 203 L398 203" />\n'
v += f'    <text x="406" y="184" class="lbl{s}">Every request for a channel or a buffer</text>\n    <text x="406" y="200" class="lbl{s}">passes through it. It writes the request</text>\n    <text x="406" y="216" class="lbl{s}">to a log.</text>\n'
v += f'    <text x="406" y="240" class="lbl{s}">In the test runs it changes one request:</text>\n    <text x="406" y="256" class="lbl{s}">where a buffer is placed, or whether</text>\n    <text x="406" y="272" class="lbl{s}">the GPU may cache it.</text>\n'
v += f'    <text x="406" y="80" class="sm{s}">Kernel launches do not go this way: the CUDA</text>\n    <text x="406" y="93" class="sm{s}">library writes them into the pushbuffer directly.</text>\n'
v += "  </svg>"
figs["library"] = v
# --- Q, W: receive throughput against the number of kernels queued behind a wait
qrows = list(csv.DictReader(open("data/queued-kernels.csv")))
QN = [0, 1, 2, 3, 4, 6, 8, 10, 14, 20, 30]
def q_curve(sel):
    out = []
    for n in QN:
        v = sorted(float(r[f"rx_gbs_N{n}"]) for r in sel if r[f"rx_gbs_N{n}"])
        if v:
            out.append((n, sum(v) / len(v), v[int(0.1 * (len(v) - 1))], v[int(round(0.9 * (len(v) - 1)))]))
    return out
def q_fig(s, title, sub, ymax, step, series, aria):
    X = lambda n: 80 + n / 30 * 560
    Y = lambda v: 250 - v / ymax * 200
    o = f'<svg viewBox="0 0 720 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{aria}">\n'
    o += style(s, "".join(f"      .ln{k}{s} {{ stroke: {c}; stroke-width: 2; fill: none; }}\n      .pt{k}{s} {{ fill: {c}; }}\n      .bd{k}{s} {{ fill: {c}; opacity: 0.16; }}\n" for k, (_, c, _) in enumerate(series))) + "\n"
    o += f'    <text x="360" y="22" class="lbl{s}-b" text-anchor="middle">{title}</text>\n    <text x="360" y="38" class="sm{s}" text-anchor="middle">{sub}</text>\n'
    v = 0
    while v <= ymax:
        o += f'    <line class="ax{s}" x1="80" y1="{Y(v):.1f}" x2="640" y2="{Y(v):.1f}" />\n    <text x="74" y="{Y(v)+4:.1f}" class="sm{s}" text-anchor="end">{v}{" GB/s" if v == ymax else ""}</text>\n'
        v += step
    for n in QN:
        o += f'    <text x="{X(n):.1f}" y="268" class="sm{s}" text-anchor="middle">{n}</text>\n'
    o += f'    <text x="360" y="288" class="sm{s}" text-anchor="middle">kernels queued behind each wait</text>\n'
    for k, (label, _, c) in enumerate(series):
        o += f'    <path class="bd{k}{s}" d="M' + " L".join(f"{X(n):.1f} {Y(hi):.1f}" for n, _, _, hi in c) + " L" + " L".join(f"{X(n):.1f} {Y(lo):.1f}" for n, _, lo, _ in reversed(c)) + ' Z" />\n'
    for k, (label, _, c) in enumerate(series):
        o += f'    <path class="ln{k}{s}" d="M' + " L".join(f"{X(n):.1f} {Y(m):.1f}" for n, m, _, _ in c) + '" />\n'
        o += "".join(f'    <circle class="pt{k}{s}" cx="{X(n):.1f}" cy="{Y(m):.1f}" r="3" />\n' for n, m, _, _ in c)
        o += f'    <text x="648" y="{Y(c[-1][1])+4:.1f}" class="lbl{s}">{label}</text>\n'
    return o + "  </svg>"
qa = q_curve([r for r in qrows if r["wait_on"] == "semaphore" and r["partition"] == "a"])
qb = q_curve([r for r in qrows if r["wait_on"] == "semaphore" and r["partition"] == "b"])
figs["queued"] = q_fig("Q", "PCIe receive throughput against the number of kernels queued behind a wait",
    "420 windows of 20 processes; lines are the means, bands hold 80 percent of the windows",
    12, 2, [("partition b", "#9c4040", qb), ("partition a", "#6b8fb0", qa)],
    "Line chart of the PCIe receive throughput against the number of small kernels queued behind each wait, from 1 to 30, for waits on a semaphore page in L2 partition a and in partition b. " +
    "Partition a: " + ", ".join(f"{m:.2f} at {n}" for n, m, _, _ in qa) + " gigabytes per second. Partition b: " + ", ".join(f"{m:.2f} at {n}" for n, m, _, _ in qb) + ". With one kernel partition b is slightly below partition a; from two kernels on it is above.")
wd = q_curve([r for r in qrows if r["wait_on"] == "word" and r["word_memory"] == "device"])
wh = q_curve([r for r in qrows if r["wait_on"] == "word" and r["word_memory"] == "host"])
figs["words"] = q_fig("W", "The same measurement with a wait on a word of our own",
    "40 windows of 20 processes for each placement of the word; lines are the means, bands hold 80 percent of the windows",
    20, 4, [("host memory", "#9c4040", wh), ("GPU memory", "#6b8fb0", wd)],
    "Line chart of the PCIe receive throughput against the number of small kernels queued behind each wait, from 0 to 30, when the streams wait for a word in GPU memory or for a word in host memory that the GPU does not cache. " +
    "GPU memory: " + ", ".join(f"{m:.2f} at {n}" for n, m, _, _ in wd) + " gigabytes per second. Host memory: " + ", ".join(f"{m:.2f} at {n}" for n, m, _, _ in wh) + ".")
json.dump(figs, open(OUT, "w"))
print({k: len(v) for k, v in figs.items()})
