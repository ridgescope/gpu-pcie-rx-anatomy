/*
 * nv_ioctl_trace: an LD_PRELOAD interposer that logs the objects a process asks the
 * NVIDIA driver to create.
 *
 * It wraps ioctl() in the process it is loaded into. Without NV_IOCTL_REWRITE it does not
 * modify ioctl arguments. For NV_ESC_RM_ALLOC (and RM controls on channel classes) writes one
 * line before and one after the call. It touches no other process and no kernel state.
 *
 * Layouts from NVIDIA's open kernel modules, tag 570.211.01:
 *   NVOS21/NVOS64_PARAMETERS: hRoot 0, hObjectParent 4, hObjectNew 8, hClass 12,
 *     pAllocParms 16; paramsSize 24 and status 28 (NVOS21, 32 bytes) or paramsSize 32
 *     and status 40 (NVOS64, 48 bytes).
 *   NV_CHANNEL_ALLOC_PARAMS: gpFifoOffset 8, gpFifoEntries 16, flags 20,
 *     hContextShare 24, hVASpace 28, hUserdMemory[0] 32, userdOffset[0] 64,
 *     engineType 128, cid 132.
 *   NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS: hVASpace 8, engineType 12.
 *   NV_CTXSHARE_ALLOCATION_PARAMETERS: hVASpace 0, flags 4, subctxId 8.
 *   NV_MEMORY_ALLOCATION_PARAMS: type 4, flags 8, attr 24, attr2 28, size 64.
 *   NVB0B5_ALLOCATION_PARAMETERS (a copy-engine object): version 0, engineType 4.
 *   The clients pass paramsSize 0, so the class alone selects the layout.
 *   NVOS54_PARAMETERS (control): hClient 0, hObject 4, cmd 8, params 16, paramsSize 24,
 *     status 28.
 *
 * With NV_IOCTL_TRACE_MAPS set it also logs how the process maps those objects: the
 * NV_ESC_RM_MAP_MEMORY ioctl (NVOS33 with an fd: hMemory 8, offset 16, length 24, fd 48)
 * and the mmap() on that fd that follows it (the CPU address), and NV_ESC_RM_MAP_MEMORY_DMA
 * (NVOS46: hDma 8, hMemory 12, offset 16, length 24, flags 32, dmaOffset 40: the GPU
 * virtual address; libcuda does not use it). libcuda maps objects into the GPU's address
 * space through UVM instead, so the ioctls on /dev/nvidia-uvm are logged too ("U<"): every
 * one with its number, and with their fields UVM_MAP_EXTERNAL_ALLOCATION (33: base 0, length
 * 8, offset 16, perGpuAttributes[256] of 36 bytes from 24, hClient 9252, hMemory 9256),
 * UVM_ALLOC_SEMAPHORE_POOL (68: base 0, length 8, attributes from 16) and
 * UVM_REGISTER_CHANNEL (27: hChannel 24, base 32, length 40). libcuda's mmap() does not go
 * through the interposable symbol, so the CPU address of an RM mapping is found by reading
 * /proc/self/maps at the next ioctl: the new /dev/nvidia* region of the mapped length.
 * nv_trace_dump(dir) then writes the bytes of every CPU mapping of a
 * system-memory or video-memory object (classes 0x3e, 0x40) to DIR/obj-<hMemory>-<addr>.bin
 * with write(), which returns EFAULT for an address that is gone instead of faulting. It
 * reads memory objects only, never the usermode (doorbell) region or other device mappings.
 * analysis/pcie-cmd.py decodes the GPFIFOs and pushbuffers from the dump.
 *
 * Interventions. Use them only on a machine that may crash without consequences.
 * NV_IOCTL_REWRITE changes one allocation before it reaches the driver. "sem-nocache": the
 * 2 MiB system-memory object that holds the stream semaphores (class 0x3e, attr 0x2a800000,
 * attr2 0x5 as granted; requested as attr 0x3a000000, attr2 0x4, GPU caching on) is asked for
 * with GPU caching off (attr2 0x8). "pb-cache": the pushbuffer object (requested as attr
 * 0x5a000000, attr2 0, 8 MiB or more) with GPU caching on (attr2 0x4). "sem-vidmem" and
 * "pb-vidmem": the same objects as video memory, as libcuda requests its own video-memory
 * objects (class 0x40, flags 0x1c101, attr 0x18000000, attr2 0). Several may be given,
 * separated by commas. Every rewrite is logged ("R>").
 *
 * Environment: NV_IOCTL_TRACE_DIR (the log directory; unset: no log),
 * NV_IOCTL_TRACE_TAG (a name for the file).
 * Build: cc -O2 -fPIC -shared -o libnv_ioctl_trace.so nv_ioctl_trace.c
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define NV_MAGIC 'F'
#define NV_ESC_RM_CONTROL 0x2A
#define NV_ESC_RM_ALLOC 0x2B
#define NV_ESC_RM_MAP_MEMORY 0x4E
#define NV_ESC_RM_MAP_MEMORY_DMA 0x57
#define MAX_LINES 50000
#define MAX_OBJ 4096
#define MAX_MAP 1024

static int maps_on = -1;
static unsigned char nv_fd[4096];
static struct { uint32_t h, cls; } objs[MAX_OBJ];
static int n_objs;
static struct { uint32_t h; uint64_t off, len; int fd; } pend[64];  /* RM_MAP_MEMORY awaiting its mmap */
static int n_pend;
static struct { uint32_t h, cls; uint64_t off; uintptr_t addr; size_t len; int live; } maps[MAX_MAP];
static int n_maps;

static unsigned char fd_kind[4096]; /* 0 unknown, 1 /dev/nvidia-uvm, 2 other */
static uintptr_t seen[4096];
static int n_seen;
static struct { uintptr_t base; size_t len; } pools[256];
static int n_pools;

static int is_uvm(int fd)
{
    if (fd < 0 || fd >= (int)sizeof fd_kind) return 0;
    if (!fd_kind[fd]) {
        char link[64], path[256];
        snprintf(link, sizeof link, "/proc/self/fd/%d", fd);
        ssize_t n = readlink(link, path, sizeof path - 1);
        path[n > 0 ? n : 0] = 0;
        fd_kind[fd] = strstr(path, "nvidia-uvm") ? 1 : 2;
    }
    return fd_kind[fd] == 1;
}

static void resolve_pending(void);

static int want_maps(void)
{
    if (maps_on < 0) maps_on = getenv("NV_IOCTL_TRACE_MAPS") != NULL;
    return maps_on;
}

static uint32_t obj_class(uint32_t h)
{
    for (int i = n_objs - 1; i >= 0; i--)
        if (objs[i].h == h) return objs[i].cls;
    return 0;
}

static FILE *out;
static int out_state; /* 0 not tried, 1 open, 2 off */
static long lines;

static uint32_t u32(const void *p, size_t off) { uint32_t v; memcpy(&v, (const char *)p + off, 4); return v; }
static uint64_t u64(const void *p, size_t off) { uint64_t v; memcpy(&v, (const char *)p + off, 8); return v; }

static FILE *log_file(void)
{
    if (out_state == 0) {
        const char *dir = getenv("NV_IOCTL_TRACE_DIR");
        const char *tag = getenv("NV_IOCTL_TRACE_TAG");
        out_state = 2;
        if (dir && *dir) {
            char path[1024];
            snprintf(path, sizeof path, "%s/ioctl-%s-%d.log", dir, tag ? tag : "x", (int)getpid());
            out = fopen(path, "a");
            if (out) { setvbuf(out, NULL, _IOLBF, 0); out_state = 1; }
        }
    }
    return out_state == 1 && lines < MAX_LINES ? out : NULL;
}

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* NV_IOCTL_REWRITE: change one allocation in place; returns 1 if it did. a: NVOS21/64, p: NV_MEMORY_ALLOCATION_PARAMS. */
static int rewrite_alloc(void *a)
{
    static int done_sem, done_pb;
    const char *w = getenv("NV_IOCTL_REWRITE");
    if (!w || !*w || !a || u32(a, 12) != 0x3e) return 0;
    void *p = (void *)(uintptr_t)u64(a, 16);
    if (!p) return 0;
    uint32_t attr = u32(p, 24), attr2 = u32(p, 28), v; uint64_t size = u64(p, 64);
    /* as libcuda requests them (the driver returns 0x2a800000/0x5 and 0x4a800000/0x9) */
    int sem = !done_sem && attr == 0x3a000000 && attr2 == 0x4 && size == 0x200000;
    int pb = !done_pb && attr == 0x5a000000 && attr2 == 0x0 && size >= 0x800000;
    int vid = 0;
    if (sem && strstr(w, "sem-nocache")) { v = 0x8; memcpy((char *)p + 28, &v, 4); done_sem = 1; }
    else if (pb && strstr(w, "pb-cache")) { v = 0x4; memcpy((char *)p + 28, &v, 4); done_pb = 1; }
    else if ((sem && strstr(w, "sem-vidmem")) || (pb && strstr(w, "pb-vidmem"))) { vid = 1; if (sem) done_sem = 1; else done_pb = 1; }
    else return 0;
    if (vid) {
        v = 0x40; memcpy((char *)a + 12, &v, 4);
        v = 0x1c101; memcpy((char *)p + 8, &v, 4);
        v = 0x18000000; memcpy((char *)p + 24, &v, 4);
        v = 0x0; memcpy((char *)p + 28, &v, 4);
    }
    FILE *f = log_file();
    if (f) { lines++; fprintf(f, "R> %.6f rewrite=%s was attr=%x attr2=%x size=%llx now class=%x flags=%x attr=%x attr2=%x\n", now(), w,
                     attr, attr2, (unsigned long long)size, u32(a, 12), u32(p, 8), u32(p, 24), u32(p, 28)); }
    return 1;
}

static void log_alloc(const char *when, int fd, unsigned long req, const void *a, long rc)
{
    FILE *f = log_file();
    if (!f || !a) return;
    unsigned size = (unsigned)((req >> 16) & 0x3fff);
    uint32_t cls = u32(a, 12);
    const void *p = (const void *)(uintptr_t)u64(a, 16);
    uint32_t psize = size >= 48 ? u32(a, 32) : u32(a, 24);
    uint32_t status = size >= 48 ? u32(a, 40) : u32(a, 28);
    lines++;
    fprintf(f, "%s %.6f tid=%ld fd=%d rc=%ld hRoot=%x hParent=%x hNew=%x class=%x psize=%u status=%x",
            when, now(), (long)syscall(SYS_gettid), fd, rc, u32(a, 0), u32(a, 4), u32(a, 8), cls, psize, status);
    if (p && (cls & 0xff) == 0x6f)
        fprintf(f, " CHANNEL flags=%x entries=%u gpoff=%llx ctxshare=%x vaspace=%x userd=%x userdoff=%llx eng=%u cid=%u",
                u32(p, 20), u32(p, 16), (unsigned long long)u64(p, 8), u32(p, 24), u32(p, 28), u32(p, 32),
                (unsigned long long)u64(p, 64), u32(p, 128), u32(p, 132));
    else if (p && cls == 0xa06c)
        fprintf(f, " TSG vaspace=%x eng=%u", u32(p, 8), u32(p, 12));
    else if (p && cls == 0x9067)
        fprintf(f, " CTXSHARE vaspace=%x flags=%x subctx=%u", u32(p, 0), u32(p, 4), u32(p, 8));
    else if (p && (cls & 0xff) == 0xb5)
        fprintf(f, " COPY version=%x engine=%u", u32(p, 0), u32(p, 4));
    if (when[1] == '<' && status == 0 && (cls == 0x3e || cls == 0x40 || cls == 0x3f) && n_objs < MAX_OBJ) {
        objs[n_objs].h = u32(a, 8); objs[n_objs].cls = cls; n_objs++;
    }
    if (p && (cls == 0x3e || cls == 0x40 || cls == 0x3f))
        fprintf(f, " MEMORY type=%x flags=%x attr=%x attr2=%x size=%llx", u32(p, 4), u32(p, 8), u32(p, 24),
                u32(p, 28), (unsigned long long)u64(p, 64));
    fputc('\n', f);
}

static int control_wanted(uint32_t cmd)
{
    uint32_t c = cmd >> 16;
    return c == 0xa06c || c == 0xa06f || c == 0xc36f || c == 0xc86f || c == 0x906f || c == 0x9067;
}

int ioctl(int fd, unsigned long req, ...)
{
    va_list ap;
    va_start(ap, req);
    void *arg = va_arg(ap, void *);
    va_end(ap);

    if (want_maps() && n_pend) { int e0 = errno; resolve_pending(); errno = e0; }
    if (want_maps() && req < 0x400 && is_uvm(fd)) {  /* UVM ioctls are plain numbers */
        long urc = syscall(SYS_ioctl, fd, req, arg);
        int e = errno;
        FILE *f = log_file();
        if (f && arg) {
            lines++;
            fprintf(f, "U< %.6f tid=%ld rc=%ld nr=%lu", now(), (long)syscall(SYS_gettid), urc, req);
            if (req == 33)
                fprintf(f, " EXTERNAL base=%llx len=%llx off=%llx maptype=%u caching=%u nattr=%llu hClient=%x hMemory=%x class=%x status=%x",
                        (unsigned long long)u64(arg, 0), (unsigned long long)u64(arg, 8), (unsigned long long)u64(arg, 16),
                        u32(arg, 40), u32(arg, 44), (unsigned long long)u64(arg, 9240), u32(arg, 9252), u32(arg, 9256),
                        obj_class(u32(arg, 9256)), u32(arg, 9260));
            else if (req == 68) {
                fprintf(f, " SEMPOOL base=%llx len=%llx maptype=%u caching=%u nattr=%llu status=%x", (unsigned long long)u64(arg, 0),
                        (unsigned long long)u64(arg, 8), u32(arg, 32), u32(arg, 36), (unsigned long long)u64(arg, 9232), u32(arg, 9240));
                if (urc == 0 && u32(arg, 9240) == 0 && n_pools < 256) { pools[n_pools].base = (uintptr_t)u64(arg, 0); pools[n_pools].len = (size_t)u64(arg, 8); n_pools++; }
            } else if (req == 27)
                fprintf(f, " CHANNEL hClient=%x hChannel=%x base=%llx len=%llx status=%x", u32(arg, 20), u32(arg, 24),
                        (unsigned long long)u64(arg, 32), (unsigned long long)u64(arg, 40), u32(arg, 48));
            fputc('\n', f);
        }
        errno = e;
        return (int)urc;
    }
    int nv = ((req >> 8) & 0xff) == NV_MAGIC;
    unsigned nr = (unsigned)(req & 0xff);
    if (nv && fd >= 0 && fd < (int)sizeof nv_fd) nv_fd[fd] = 1;
    int is_map = nv && arg && want_maps() && ((nr == NV_ESC_RM_MAP_MEMORY && ((req >> 16) & 0x3fff) >= 52) ||
                                              (nr == NV_ESC_RM_MAP_MEMORY_DMA && ((req >> 16) & 0x3fff) >= 48));
    int is_alloc = nv && nr == NV_ESC_RM_ALLOC && ((req >> 16) & 0x3fff) >= 32;
    int is_ctrl = nv && nr == NV_ESC_RM_CONTROL && ((req >> 16) & 0x3fff) >= 32 && arg && control_wanted(u32(arg, 8));

    if (is_alloc) { rewrite_alloc(arg); log_alloc("A>", fd, req, arg, 0); }
    long rc = syscall(SYS_ioctl, fd, req, arg);
    if (is_map) {
        int e = errno;
        FILE *f = log_file();
        if (nr == NV_ESC_RM_MAP_MEMORY) {
            int mfd = (int)u32(arg, 48);
            if (rc == 0 && u32(arg, 40) == 0 && n_pend < 64) {
                pend[n_pend].h = u32(arg, 8); pend[n_pend].off = u64(arg, 16); pend[n_pend].len = u64(arg, 24);
                pend[n_pend].fd = mfd; n_pend++;
            }
            if (f) { lines++; fprintf(f, "P< %.6f tid=%ld rc=%ld hMemory=%x class=%x off=%llx len=%llx mapfd=%d status=%x flags=%x\n",
                             now(), (long)syscall(SYS_gettid), rc, u32(arg, 8), obj_class(u32(arg, 8)),
                             (unsigned long long)u64(arg, 16), (unsigned long long)u64(arg, 24), mfd, u32(arg, 40), u32(arg, 44)); }
        } else if (f) {
            lines++;
            fprintf(f, "G< %.6f tid=%ld rc=%ld hDma=%x hMemory=%x class=%x off=%llx len=%llx flags=%x gpuva=%llx\n", now(),
                    (long)syscall(SYS_gettid), rc, u32(arg, 8), u32(arg, 12), obj_class(u32(arg, 12)),
                    (unsigned long long)u64(arg, 16), (unsigned long long)u64(arg, 24), u32(arg, 32),
                    (unsigned long long)u64(arg, 40));
        }
        errno = e;
    }
    if (is_alloc || is_ctrl) {
        int e = errno;
        if (is_alloc) log_alloc("A<", fd, req, arg, rc);
        else {
            FILE *f = log_file();
            if (f) {
                lines++;
                const void *cp = (const void *)(uintptr_t)u64(arg, 16);
                fprintf(f, "C< %.6f tid=%ld fd=%d rc=%ld hClient=%x hObject=%x cmd=%x psize=%u status=%x", now(),
                        (long)syscall(SYS_gettid), fd, rc, u32(arg, 0), u32(arg, 4), u32(arg, 8), u32(arg, 24), u32(arg, 28));
                /* NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN: one NvU32, the doorbell token of the channel */
                if (u32(arg, 8) == 0xc36f0108 && cp && u32(arg, 24) >= 4) fprintf(f, " token=%x", u32(cp, 0));
                fputc('\n', f);
            }
        }
        errno = e;
    }
    return (int)rc;
}

/* Find the CPU addresses of pending RM mappings: new /dev/nvidia* regions in /proc/self/maps. */
static void resolve_pending(void)
{
    if (!n_pend) return;
    FILE *m = fopen("/proc/self/maps", "r");
    if (!m) return;
    FILE *f = log_file();
    char line[1024];
    while (fgets(line, sizeof line, m)) {
        if (!strstr(line, "/dev/nvidia") || strstr(line, "nvidia-uvm")) continue;
        unsigned long a, b;
        if (sscanf(line, "%lx-%lx", &a, &b) != 2) continue;
        int known = 0;
        for (int i = 0; i < n_seen; i++) if (seen[i] == a) { known = 1; break; }
        if (known) continue;
        if (n_seen < 4096) seen[n_seen++] = a;
        size_t len = b - a;
        for (int i = 0; i < n_pend; i++) {
            if (((pend[i].len + 4095) & ~4095ull) != len) continue;
            if (n_maps < MAX_MAP) {
                maps[n_maps].h = pend[i].h; maps[n_maps].cls = obj_class(pend[i].h); maps[n_maps].off = pend[i].off;
                maps[n_maps].addr = a; maps[n_maps].len = len; maps[n_maps].live = 1; n_maps++;
            }
            if (f) { lines++; fprintf(f, "M< %.6f addr=%lx len=%zx hMemory=%x class=%x objoff=%llx\n", now(), a, len, pend[i].h,
                             obj_class(pend[i].h), (unsigned long long)pend[i].off); }
            for (int k = i; k < n_pend - 1; k++) pend[k] = pend[k + 1];
            n_pend--;
            break;
        }
    }
    fclose(m);
}

/* mmap on an NVIDIA device fd right after NV_ESC_RM_MAP_MEMORY: the CPU address of that object. */
static void note_mmap(void *ret, size_t len, int fd, long off)
{
    if (ret == MAP_FAILED || fd < 0 || fd >= (int)sizeof nv_fd || !nv_fd[fd] || !want_maps()) return;
    uint32_t h = 0; uint64_t ooff = 0;
    for (int i = n_pend - 1; i >= 0; i--)
        if (pend[i].fd == fd) { h = pend[i].h; ooff = pend[i].off; pend[i] = pend[--n_pend]; break; }
    if (h && n_maps < MAX_MAP) {
        maps[n_maps].h = h; maps[n_maps].cls = obj_class(h); maps[n_maps].off = ooff;
        maps[n_maps].addr = (uintptr_t)ret; maps[n_maps].len = len; maps[n_maps].live = 1; n_maps++;
    }
    FILE *f = log_file();
    if (f) { lines++; fprintf(f, "M< %.6f tid=%ld fd=%d addr=%lx len=%zx fileoff=%lx hMemory=%x class=%x objoff=%llx\n", now(),
                     (long)syscall(SYS_gettid), fd, (unsigned long)(uintptr_t)ret, len, off, h, obj_class(h),
                     (unsigned long long)ooff); }
}

void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off)
{
    void *r = (void *)syscall(SYS_mmap, addr, len, prot, flags, fd, off);
    if (fd >= 0) { int e = errno; note_mmap(r, len, fd, (long)off); errno = e; }
    return r;
}

void *mmap64(void *addr, size_t len, int prot, int flags, int fd, off64_t off)
{
    void *r = (void *)syscall(SYS_mmap, addr, len, prot, flags, fd, off);
    if (fd >= 0) { int e = errno; note_mmap(r, len, fd, (long)off); errno = e; }
    return r;
}

int munmap(void *addr, size_t len)
{
    for (int i = 0; i < n_maps; i++)
        if (maps[i].live && maps[i].addr == (uintptr_t)addr) maps[i].live = 0;
    return (int)syscall(SYS_munmap, addr, len);
}

/* Write every live CPU mapping of a system- or video-memory object to DIR. Returns the count. */
int nv_trace_dump(const char *dir)
{
    int n = 0;
    FILE *f = log_file();
    resolve_pending();
    for (int i = 0; i < n_pools; i++) {  /* UVM semaphore pools: ordinary memory of this process */
        char path[1200];
        snprintf(path, sizeof path, "%s/pool-%d-%lx.bin", dir, i, (unsigned long)pools[i].base);
        int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) continue;
        long w = syscall(SYS_write, fd, (const char *)pools[i].base, pools[i].len);
        close(fd);
        if (f) { lines++; fprintf(f, "D< %.6f dir=%s pool=%d addr=%lx len=%zx written=%ld err=%d\n", now(), dir, i,
                         (unsigned long)pools[i].base, pools[i].len, w, w < 0 ? errno : 0); }
        n++;
    }
    for (int i = 0; i < n_maps; i++) {
        if (!maps[i].live || (maps[i].cls != 0x3e && maps[i].cls != 0x40) || maps[i].len > (128u << 20)) continue;
        char path[1200];
        snprintf(path, sizeof path, "%s/obj-%x-%lx.bin", dir, maps[i].h, (unsigned long)maps[i].addr);
        int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) continue;
        size_t done = 0; long w = 0;
        while (done < maps[i].len) {
            size_t chunk = maps[i].len - done; if (chunk > (1u << 20)) chunk = 1u << 20;
            w = syscall(SYS_write, fd, (const char *)maps[i].addr + done, chunk);
            if (w <= 0) break;
            done += (size_t)w;
        }
        close(fd);
        if (f) { lines++; fprintf(f, "D< %.6f dir=%s hMemory=%x class=%x addr=%lx len=%zx objoff=%llx written=%zx err=%d\n", now(), dir,
                         maps[i].h, maps[i].cls, (unsigned long)maps[i].addr, maps[i].len,
                         (unsigned long long)maps[i].off, done, w < 0 ? errno : 0); }
        n++;
    }
    return n;
}
