#!/usr/bin/env python3
"""
nv-monitor - System monitor for NVIDIA GPU systems (Python port)

Displays CPU per-core usage, memory, CPU thermals, GPU utilization,
GPU temperature/power/clock, and GPU processes in a single TUI.

All features of the C version are preserved:
  - CPU per-core delta-based usage from /proc/stat
  - Memory from /proc/meminfo with HugePages fix for DGX Spark
  - CPU temperature (highest thermal zone) and frequency
  - GPU via NVML loaded dynamically with ctypes (libnvidia-ml.so.1)
  - GPU util, temp, power, graphics/memory clocks, VRAM, ENC/DEC, fan
  - GPU process list (compute + graphics) with per-process CPU%
  - ARM big.LITTLE core type labels (Cortex-X925/X725 on Grace)
  - Tegra GPU sysfs fallback (Jetson Orin / Nano / NX / AGX)
  - RDMA/InfiniBand monitoring via /sys/class/infiniband
  - 20-sample history chart with Unicode block elements (▁▂▃▄▅▆▇█)
  - TUI via Python curses (ncurses-equivalent) with color-coded bars
  - CSV logging with configurable interval
  - Prometheus/OpenMetrics exporter on a dedicated thread
  - Bearer token auth for /metrics
  - Headless mode

Build/Run: python3 nv-monitor.py [OPTIONS]
Dependencies: Python 3.7+ stdlib only (curses, ctypes, threading, socket)
"""

import sys
import os
import re
import time
import stat as stat_mod
import signal
import locale
import argparse
import threading
import socket
import pwd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import curses

# ── Version ───────────────────────────────────────────────────────────

VERSION = os.environ.get("NV_MONITOR_VERSION", "dev")

# ── Constants ─────────────────────────────────────────────────────────

MAX_CPUS       = 128
MAX_GPU_PROCS  = 64
REFRESH_MS     = 1000
HISTORY_LEN    = 20

# curses color pair indices (match C version)
CP_RED     = 1
CP_GREEN   = 2
CP_YELLOW  = 3
CP_BLUE    = 4
CP_MAGENTA = 5
CP_CYAN    = 6
CP_WHITE   = 7
CP_DIM     = 8

# ARM CPU part ID → core label
CPU_PART_LABELS: Dict[int, str] = {
    0xd85: "X925",  # Cortex-X925  (Grace performance, DGX Spark)
    0xd87: "X725",  # Cortex-X725  (Grace efficiency, DGX Spark)
    0xd42: "A78A",  # Cortex-A78AE (Jetson Orin)
    0xd44: "X4",
    0xd43: "A720",
    0xd46: "A725",
    0xd41: "A78",
    0xd40: "V2",
    0xd0b: "A76",
    0xd0a: "A75",
    0xd07: "A57",   # Jetson TX1/TX2
    0xd03: "A53",   # Jetson Nano (original)
}

# Unicode block elements ▁▂▃▄▅▆▇█ (U+2581..U+2588)
BLOCK_CHARS = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

# ── NVML constants ────────────────────────────────────────────────────

NVML_SUCCESS              = 0
NVML_ERROR_NOT_SUPPORTED  = 3
NVML_TEMPERATURE_GPU      = 0
NVML_CLOCK_GRAPHICS       = 0
NVML_CLOCK_MEM            = 2
NVML_CLOCK_SM             = 1

# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class CpuTick:
    user:    int = 0
    nice:    int = 0
    system:  int = 0
    idle:    int = 0
    iowait:  int = 0
    irq:     int = 0
    softirq: int = 0
    steal:   int = 0

    def total(self) -> int:
        return (self.user + self.nice + self.system + self.idle +
                self.iowait + self.irq + self.softirq + self.steal)

    def idle_total(self) -> int:
        return self.idle + self.iowait


@dataclass
class MemInfo:
    total_kb:     int = 0
    free_kb:      int = 0
    avail_kb:     int = 0
    buffers_kb:   int = 0
    cached_kb:    int = 0
    swap_total_kb: int = 0
    swap_free_kb:  int = 0
    # Derived
    app_kb:       int = 0
    bufcache_kb:  int = 0
    swap_used_kb: int = 0

    def calc(self):
        self.bufcache_kb = self.buffers_kb + self.cached_kb
        self.app_kb = max(0, self.total_kb - self.free_kb - self.bufcache_kb)
        self.swap_used_kb = max(0, self.swap_total_kb - self.swap_free_kb)


@dataclass
class GpuProc:
    pid:       int   = 0
    mem_bytes: int   = 0
    name:      str   = ""
    user:      str   = ""
    type:      str   = "C"   # 'C' = compute, 'G' = graphics
    cpu_pct:   float = 0.0


@dataclass
class RdmaPort:
    device:         str   = ""
    port:           int   = 0
    state:          str   = ""
    rate:           str   = ""
    xmit_bytes:     int   = 0
    recv_bytes:     int   = 0
    xmit_pkts:      int   = 0
    recv_pkts:      int   = 0
    errors:         int   = 0
    xmit_bytes_sec: float = 0.0
    recv_bytes_sec: float = 0.0

# ── NVML dynamic loading via ctypes ───────────────────────────────────

import ctypes

class _NvmlUtilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

class _NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free",  ctypes.c_ulonglong),
        ("used",  ctypes.c_ulonglong),
    ]

class _NvmlProcessInfo(ctypes.Structure):
    _fields_ = [
        ("pid",               ctypes.c_uint),
        ("usedGpuMemory",     ctypes.c_ulonglong),
        ("gpuInstanceId",     ctypes.c_uint),
        ("computeInstanceId", ctypes.c_uint),
    ]


class NVML:
    """Thin wrapper around libnvidia-ml.so loaded dynamically via ctypes.
    Mirrors the C version's dlopen/dlsym pattern."""

    def __init__(self):
        self._lib  = None
        self._ok   = False

    def load(self) -> bool:
        _lib_paths = [
            "libnvidia-ml.so.1",
            "libnvidia-ml.so",
            "/usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1",
            "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
        ]
        for p in _lib_paths:
            try:
                self._lib = ctypes.CDLL(p)
                break
            except OSError:
                continue
        if self._lib is None:
            return False

        def _sym(name: str, *fallbacks):
            """Try versioned symbol first, then base name."""
            for n in (name,) + fallbacks:
                fn = getattr(self._lib, n, None)
                if fn:
                    return fn
            return None

        self._Init               = _sym("nvmlInit_v2", "nvmlInit")
        self._Shutdown           = _sym("nvmlShutdown")
        self._DeviceGetCount     = _sym("nvmlDeviceGetCount_v2", "nvmlDeviceGetCount")
        self._DeviceGetHandle    = _sym("nvmlDeviceGetHandleByIndex_v2", "nvmlDeviceGetHandleByIndex")
        self._DeviceGetName      = _sym("nvmlDeviceGetName")
        self._GetUtil            = _sym("nvmlDeviceGetUtilizationRates")
        self._GetMemInfo         = _sym("nvmlDeviceGetMemoryInfo")
        self._GetTemp            = _sym("nvmlDeviceGetTemperature")
        self._GetPower           = _sym("nvmlDeviceGetPowerUsage")
        self._GetClock           = _sym("nvmlDeviceGetClockInfo")
        self._GetComputeProcs    = _sym("nvmlDeviceGetComputeRunningProcesses_v3",
                                        "nvmlDeviceGetComputeRunningProcesses")
        self._GetGraphicsProcs   = _sym("nvmlDeviceGetGraphicsRunningProcesses_v3",
                                        "nvmlDeviceGetGraphicsRunningProcesses")
        self._GetFan             = _sym("nvmlDeviceGetFanSpeed")
        self._GetEnc             = _sym("nvmlDeviceGetEncoderUtilization")
        self._GetDec             = _sym("nvmlDeviceGetDecoderUtilization")

        if not self._Init:
            return False
        if self._Init() != NVML_SUCCESS:
            return False
        self._ok = True
        return True

    def is_ok(self) -> bool:
        return self._ok

    def device_count(self) -> int:
        if not self._ok or not self._DeviceGetCount:
            return 0
        n = ctypes.c_uint(0)
        if self._DeviceGetCount(ctypes.byref(n)) == NVML_SUCCESS:
            return n.value
        return 0

    def device_handle(self, index: int) -> Optional[ctypes.c_void_p]:
        if not self._ok or not self._DeviceGetHandle:
            return None
        h = ctypes.c_void_p(0)
        if self._DeviceGetHandle(index, ctypes.byref(h)) == NVML_SUCCESS:
            return h
        return None

    def device_name(self, handle) -> str:
        if not self._ok or not self._DeviceGetName:
            return "Unknown"
        buf = ctypes.create_string_buffer(96)
        if self._DeviceGetName(handle, buf, 96) == NVML_SUCCESS:
            return buf.value.decode("utf-8", errors="replace")
        return "Unknown"

    def utilization(self, handle) -> Optional[Tuple[int, int]]:
        """Returns (gpu%, memory%) or None."""
        if not self._ok or not self._GetUtil:
            return None
        u = _NvmlUtilization()
        if self._GetUtil(handle, ctypes.byref(u)) == NVML_SUCCESS:
            return u.gpu, u.memory
        return None

    def temperature(self, handle) -> Optional[int]:
        if not self._ok or not self._GetTemp:
            return None
        t = ctypes.c_uint(0)
        if self._GetTemp(handle, NVML_TEMPERATURE_GPU, ctypes.byref(t)) == NVML_SUCCESS:
            return t.value
        return None

    def power_usage_mw(self, handle) -> Optional[int]:
        if not self._ok or not self._GetPower:
            return None
        pw = ctypes.c_uint(0)
        if self._GetPower(handle, ctypes.byref(pw)) == NVML_SUCCESS:
            return pw.value
        return None

    def clock_mhz(self, handle, clock_type: int) -> Optional[int]:
        if not self._ok or not self._GetClock:
            return None
        clk = ctypes.c_uint(0)
        if self._GetClock(handle, clock_type, ctypes.byref(clk)) == NVML_SUCCESS:
            return clk.value
        return None

    def memory_info(self, handle) -> Optional[Tuple[int, int, int]]:
        """Returns (total, free, used) in bytes, or None."""
        if not self._ok or not self._GetMemInfo:
            return None
        m = _NvmlMemory()
        if self._GetMemInfo(handle, ctypes.byref(m)) == NVML_SUCCESS:
            return m.total, m.free, m.used
        return None

    def fan_speed(self, handle) -> Optional[int]:
        if not self._ok or not self._GetFan:
            return None
        fs = ctypes.c_uint(0)
        if self._GetFan(handle, ctypes.byref(fs)) == NVML_SUCCESS:
            return fs.value
        return None

    def encoder_utilization(self, handle) -> Optional[int]:
        if not self._ok or not self._GetEnc:
            return None
        util   = ctypes.c_uint(0)
        period = ctypes.c_uint(0)
        if self._GetEnc(handle, ctypes.byref(util), ctypes.byref(period)) == NVML_SUCCESS:
            return util.value
        return None

    def decoder_utilization(self, handle) -> Optional[int]:
        if not self._ok or not self._GetDec:
            return None
        util   = ctypes.c_uint(0)
        period = ctypes.c_uint(0)
        if self._GetDec(handle, ctypes.byref(util), ctypes.byref(period)) == NVML_SUCCESS:
            return util.value
        return None

    def compute_processes(self, handle) -> List[Tuple[int, int]]:
        """Returns list of (pid, used_gpu_memory_bytes)."""
        if not self._ok or not self._GetComputeProcs:
            return []
        n     = ctypes.c_uint(MAX_GPU_PROCS)
        procs = (_NvmlProcessInfo * MAX_GPU_PROCS)()
        if self._GetComputeProcs(handle, ctypes.byref(n), procs) != NVML_SUCCESS:
            return []
        result = []
        for i in range(n.value):
            pid = procs[i].pid
            mem = procs[i].usedGpuMemory
            if pid == 0 or pid > 4194304:   # Jetson NVML can return garbage
                continue
            if mem == 0xFFFFFFFFFFFFFFFF:
                mem = 0
            result.append((pid, mem))
        return result

    def graphics_processes(self, handle) -> List[Tuple[int, int]]:
        """Returns list of (pid, used_gpu_memory_bytes)."""
        if not self._ok or not self._GetGraphicsProcs:
            return []
        n     = ctypes.c_uint(MAX_GPU_PROCS)
        procs = (_NvmlProcessInfo * MAX_GPU_PROCS)()
        if self._GetGraphicsProcs(handle, ctypes.byref(n), procs) != NVML_SUCCESS:
            return []
        result = []
        for i in range(n.value):
            pid = procs[i].pid
            mem = procs[i].usedGpuMemory
            if pid == 0 or pid > 4194304:
                continue
            if mem == 0xFFFFFFFFFFFFFFFF:
                mem = 0
            result.append((pid, mem))
        return result

    def shutdown(self):
        if self._ok and self._Shutdown:
            self._Shutdown()
            self._ok = False


# ── Global state ──────────────────────────────────────────────────────

nvml      = NVML()
nvml_ok   = False
gpu_count = 0
use_tegra_gpu = False
cpu_model_name = ""

num_cpus  = 0
prev_ticks: List[CpuTick] = [CpuTick() for _ in range(MAX_CPUS + 1)]
cpu_pct   = [0.0] * (MAX_CPUS + 1)
cpu_parts = [0] * MAX_CPUS

cpu_history = [0.0] * HISTORY_LEN
gpu_history = [0.0] * HISTORY_LEN
history_pos   = 0
history_count = 0

sort_mode     = 0    # 0 = by GPU mem, 1 = by PID
delay_ms      = REFRESH_MS
last_gpu_util = 0.0

log_fp           = None
log_interval_ms  = 1000
no_ui            = False
prom_port        = 0
prom_token: Optional[str] = None
g_quit           = False

# Per-process CPU tracking (dict: pid → ticks snapshot)
prev_proc_snaps: Dict[int, int] = {}
prev_total_cpu_ticks: int = 0

# Tegra GPU
tegra_gpu_available  = False
tegra_gpu_load_path  = ""
tegra_gpu_therm_zone = -1
tegra_gpu_dev        = None   # st_rdev value of GPU device node

# RDMA
rdma_ports: List[RdmaPort] = []
rdma_available   = False
_rdma_prev_xmit: Dict[Tuple[str, int], int] = {}
_rdma_prev_recv: Dict[Tuple[str, int], int] = {}
_rdma_prev_time: Optional[float] = None

# ── sysfs / procfs helpers ────────────────────────────────────────────

def _read_str(path: str) -> Optional[str]:
    """Read a sysfs/procfs file as a stripped string. Returns None on error."""
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except Exception:
        return None


def _read_int(path: str) -> Optional[int]:
    s = _read_str(path)
    if s is None:
        return None
    try:
        return int(s)
    except ValueError:
        return None


# ── CPU model name ────────────────────────────────────────────────────

def read_cpu_model_name() -> str:
    # 1. Device tree (ARM SBCs, DGX Spark)
    val = _read_str("/sys/firmware/devicetree/base/model")
    if val:
        return val.replace("\x00", "").strip()

    # 2. DMI product name (DGX Spark, most x86 systems)
    val = _read_str("/sys/devices/virtual/dmi/id/product_name")
    if val:
        return val.replace("_", " ").strip()

    # 3. x86: /proc/cpuinfo "model name"
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    _, _, v = line.partition(":")
                    return v.strip()
    except Exception:
        pass
    return ""


def read_cpu_part_ids() -> List[int]:
    """Read ARM CPU part hex IDs per core from /proc/cpuinfo."""
    parts = [0] * MAX_CPUS
    try:
        with open("/proc/cpuinfo", "r") as f:
            cur = -1
            for line in f:
                m = re.match(r"processor\s*:\s*(\d+)", line)
                if m:
                    cur = int(m.group(1))
                elif cur >= 0 and cur < MAX_CPUS:
                    m2 = re.match(r"CPU part\s*:\s*(0x[0-9a-fA-F]+)", line)
                    if m2:
                        parts[cur] = int(m2.group(1), 16)
    except Exception:
        pass
    return parts


def cpu_part_label(cpu_idx: int) -> str:
    return CPU_PART_LABELS.get(cpu_parts[cpu_idx], "")


# ── CPU sampling ──────────────────────────────────────────────────────

def _parse_cpu_line(fields: List[str]) -> CpuTick:
    t = CpuTick()
    vals = [int(x) for x in fields[:8]]
    pad  = [0] * (8 - len(vals))
    vals += pad
    t.user, t.nice, t.system, t.idle, t.iowait, t.irq, t.softirq, t.steal = vals
    return t


def read_cpu_ticks() -> Tuple[List[CpuTick], int]:
    """Returns (ticks[0..n], n_cpus). Index 0 = aggregate."""
    ticks = [CpuTick() for _ in range(MAX_CPUS + 1)]
    n = 0
    try:
        with open("/proc/stat", "r") as f:
            for line in f:
                if not line.startswith("cpu"):
                    continue
                parts = line.split()
                if parts[0] == "cpu":
                    ticks[0] = _parse_cpu_line(parts[1:])
                else:
                    cpu_num = int(parts[0][3:])
                    if 0 <= cpu_num < MAX_CPUS:
                        ticks[cpu_num + 1] = _parse_cpu_line(parts[1:])
                        if cpu_num + 1 > n:
                            n = cpu_num + 1
    except Exception:
        pass
    return ticks, n


def compute_cpu_usage():
    global prev_ticks, num_cpus, cpu_pct
    cur_ticks, n = read_cpu_ticks()
    num_cpus = n
    for i in range(n + 1):
        prev_idle  = prev_ticks[i].idle_total()
        cur_idle   = cur_ticks[i].idle_total()
        prev_total = prev_ticks[i].total()
        cur_total  = cur_ticks[i].total()
        totald     = cur_total - prev_total
        idled      = cur_idle  - prev_idle
        cpu_pct[i] = 0.0 if totald == 0 else (totald - idled) / totald * 100.0
    prev_ticks = cur_ticks


# ── Per-process CPU% ──────────────────────────────────────────────────

def _read_proc_stat_ticks(pid: int) -> int:
    """Read utime + stime from /proc/<pid>/stat."""
    try:
        with open(f"/proc/{pid}/stat", "r") as f:
            content = f.read()
        end = content.rfind(")")       # end of comm field
        if end < 0:
            return 0
        # Fields after ") ": state ppid pgroup session tty_nr tpgid flags
        #   minflt cminflt majflt cmajflt utime(11) stime(12) ...
        fields = content[end + 2:].split()
        return int(fields[11]) + int(fields[12])
    except Exception:
        return 0


def _read_total_cpu_ticks() -> int:
    try:
        with open("/proc/stat", "r") as f:
            line = f.readline()
        parts = line.split()
        if parts[0] == "cpu":
            return sum(int(x) for x in parts[1:])
    except Exception:
        pass
    return 0


def calc_proc_cpu_pct(pid: int) -> float:
    cur_ticks = _read_proc_stat_ticks(pid)
    cur_total = _read_total_cpu_ticks()
    total_delta = cur_total - prev_total_cpu_ticks
    if pid in prev_proc_snaps and total_delta > 0:
        proc_delta = cur_ticks - prev_proc_snaps[pid]
        return proc_delta / total_delta * 100.0 * num_cpus
    return 0.0


def update_proc_cpu_snapshots(procs: List[GpuProc]):
    global prev_proc_snaps, prev_total_cpu_ticks
    prev_proc_snaps       = {p.pid: _read_proc_stat_ticks(p.pid) for p in procs}
    prev_total_cpu_ticks  = _read_total_cpu_ticks()


# ── Memory info ───────────────────────────────────────────────────────

def read_meminfo() -> MemInfo:
    m = MemInfo()
    huge_total, huge_free, huge_size = -1, -1, -1
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                k, _, v = line.partition(":")
                k = k.strip()
                tok = v.strip().split()
                if not tok:
                    continue
                try:
                    val = int(tok[0])
                except ValueError:
                    continue
                if   k == "MemTotal":       m.total_kb      = val
                elif k == "MemFree":        m.free_kb       = val
                elif k == "MemAvailable":   m.avail_kb      = val
                elif k == "Buffers":        m.buffers_kb    = val
                elif k == "Cached":         m.cached_kb     = val
                elif k == "SwapTotal":      m.swap_total_kb = val
                elif k == "SwapFree":       m.swap_free_kb  = val
                elif k == "HugePages_Total":huge_total       = val
                elif k == "HugePages_Free": huge_free        = val
                elif k == "Hugepagesize":   huge_size        = val
    except Exception:
        pass

    # DGX Spark: when HugePages are active, MemAvailable is inaccurate.
    # Use HugePages_Free * Hugepagesize instead; report swap as 0
    # (HugeTLB pages are not swappable).
    # See: docs.nvidia.com/dgx/dgx-spark/known-issues.html
    if huge_total > 0 and huge_free >= 0 and huge_size > 0:
        m.avail_kb    = huge_free * huge_size
        m.swap_free_kb = m.swap_total_kb     # effective 0 swap used

    m.calc()
    return m


# ── CPU temperature and frequency ────────────────────────────────────

def read_cpu_temp() -> int:
    """Returns highest thermal zone temp in °C."""
    max_temp = 0
    for i in range(20):
        val = _read_int(f"/sys/class/thermal/thermal_zone{i}/temp")
        if val is None:
            break
        max_temp = max(max_temp, val)
    return max_temp // 1000


def read_cpu_freq_mhz() -> int:
    val = _read_int("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    return (val // 1000) if val else 0


# ── Uptime / load average ─────────────────────────────────────────────

def fmt_uptime() -> str:
    try:
        with open("/proc/uptime", "r") as f:
            s = int(float(f.read().split()[0]))
        days = s // 86400; s %= 86400
        hrs  = s // 3600;  s %= 3600
        mins = s // 60
        return f"{days}d {hrs}h {mins}m" if days else f"{hrs}h {mins}m"
    except Exception:
        return "?"


def get_loadavg() -> Tuple[float, float, float]:
    try:
        with open("/proc/loadavg", "r") as f:
            p = f.read().split()
        return float(p[0]), float(p[1]), float(p[2])
    except Exception:
        return 0.0, 0.0, 0.0


# ── Process info helpers ──────────────────────────────────────────────

def get_proc_name(pid: int) -> str:
    return _read_str(f"/proc/{pid}/comm") or f"[pid {pid}]"


def get_proc_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            data = f.read(1024)
        if not data:
            return get_proc_name(pid)
        # NUL-separated args → space-separated
        cmd = data.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
        # Shorten first arg to its basename, keep rest
        space = cmd.find(" ")
        if space >= 0:
            return os.path.basename(cmd[:space]) + cmd[space:]
        return os.path.basename(cmd)
    except Exception:
        return get_proc_name(pid)


def get_proc_user(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("Uid:"):
                    uid = int(line.split()[1])
                    try:
                        return pwd.getpwuid(uid).pw_name
                    except KeyError:
                        return str(uid)
    except Exception:
        pass
    return "?"


# ── Tegra GPU sysfs fallback ──────────────────────────────────────────

def detect_tegra_gpu():
    global tegra_gpu_available, tegra_gpu_load_path, tegra_gpu_therm_zone
    gpu_paths = [
        "/sys/devices/gpu.0/load",
        "/sys/devices/platform/bus@0/17000000.gpu/load",
        "/sys/devices/platform/17000000.gpu/load",
    ]
    for p in gpu_paths:
        if os.path.exists(p):
            tegra_gpu_available = True
            tegra_gpu_load_path = p
            break

    for i in range(20):
        t = _read_str(f"/sys/class/thermal/thermal_zone{i}/type")
        if t is None:
            break
        if t.lower() in ("gpu-therm", "gpu-thermal"):
            tegra_gpu_therm_zone = i
            break


def detect_tegra_gpu_dev():
    global tegra_gpu_dev
    dev_paths = ["/dev/nvhost-gpu", "/dev/dri/card0", "/dev/dri/renderD128"]
    for p in dev_paths:
        try:
            st = os.stat(p)
            if stat_mod.S_ISCHR(st.st_mode):
                tegra_gpu_dev = st.st_rdev
                return
        except Exception:
            pass


def read_tegra_gpu_util() -> int:
    val = _read_int(tegra_gpu_load_path)
    return -1 if val is None else val // 10  # 0-1000 → 0-100


def read_tegra_gpu_temp() -> int:
    if tegra_gpu_therm_zone < 0:
        return -1
    val = _read_int(f"/sys/class/thermal/thermal_zone{tegra_gpu_therm_zone}/temp")
    return -1 if val is None else val // 1000


def scan_tegra_gpu_procs() -> List[GpuProc]:
    """Scan /proc for processes with open fds to GPU device nodes.
    Used on Jetson where NVML process listing returns garbage."""
    if not tegra_gpu_dev and not use_tegra_gpu:
        return []
    procs: List[GpuProc] = []
    my_pid   = os.getpid()
    seen_pids = set()
    try:
        for entry in os.scandir("/proc"):
            if len(procs) >= MAX_GPU_PROCS:
                break
            try:
                pid = int(entry.name)
            except ValueError:
                continue
            if pid == my_pid or pid in seen_pids:
                continue
            try:
                found = False
                for fd_entry in os.scandir(f"/proc/{pid}/fd"):
                    fd_path = fd_entry.path
                    # Check 1: device file rdev matches GPU
                    if tegra_gpu_dev:
                        try:
                            fd_st = os.stat(fd_path)
                            if stat_mod.S_ISCHR(fd_st.st_mode) and fd_st.st_rdev == tegra_gpu_dev:
                                found = True
                                break
                        except Exception:
                            pass
                    # Check 2: symlink target has nvhost/gpu or /dev/dri/render
                    try:
                        target = os.readlink(fd_path)
                        if ("nvhost" in target and "gpu" in target) or "/dev/dri/render" in target:
                            found = True
                            break
                    except Exception:
                        pass
                if found:
                    seen_pids.add(pid)
                    procs.append(GpuProc(
                        pid=pid, mem_bytes=0, type="C",
                        cpu_pct=calc_proc_cpu_pct(pid),
                        name=get_proc_cmdline(pid),
                        user=get_proc_user(pid),
                    ))
            except Exception:
                pass
    except Exception:
        pass
    return procs


# ── RDMA / InfiniBand monitoring ──────────────────────────────────────

_RDMA_ERR_COUNTERS = [
    "symbol_error_counter", "port_rcv_errors",
    "port_rcv_constraint_errors", "port_xmit_constraint_errors",
    "link_error_recovery_counter", "link_downed_counter",
]


def read_rdma_ports():
    global rdma_ports, rdma_available, _rdma_prev_xmit, _rdma_prev_recv, _rdma_prev_time
    ib_base = "/sys/class/infiniband"
    if not os.path.isdir(ib_base):
        rdma_available = False
        rdma_ports = []
        return

    now = time.monotonic()
    dt  = max(1.0, now - _rdma_prev_time) if _rdma_prev_time is not None else 0.0

    rdma_available = True
    new_ports: List[RdmaPort] = []
    try:
        for dev_entry in os.scandir(ib_base):
            dev_name = dev_entry.name
            for p in range(1, 3):
                state_path = f"{ib_base}/{dev_name}/ports/{p}/state"
                if not os.path.exists(state_path):
                    continue
                r = RdmaPort(device=dev_name, port=p)

                state = _read_str(state_path) or ""
                if ":" in state:
                    state = state.split(":", 1)[1].strip()
                r.state = state
                r.rate  = _read_str(f"{ib_base}/{dev_name}/ports/{p}/rate") or ""

                base = f"{ib_base}/{dev_name}/ports/{p}/counters"
                r.xmit_bytes = (_read_int(f"{base}/port_xmit_data") or 0) * 4
                r.recv_bytes = (_read_int(f"{base}/port_rcv_data")  or 0) * 4
                r.xmit_pkts  =  _read_int(f"{base}/port_xmit_packets") or 0
                r.recv_pkts  =  _read_int(f"{base}/port_rcv_packets")  or 0
                r.errors     = sum(_read_int(f"{base}/{e}") or 0 for e in _RDMA_ERR_COUNTERS)

                key = (dev_name, p)
                if _rdma_prev_time is not None and key in _rdma_prev_xmit and dt > 0:
                    r.xmit_bytes_sec = max(0.0, (r.xmit_bytes - _rdma_prev_xmit[key]) / dt)
                    r.recv_bytes_sec = max(0.0, (r.recv_bytes - _rdma_prev_recv[key]) / dt)
                _rdma_prev_xmit[key] = r.xmit_bytes
                _rdma_prev_recv[key] = r.recv_bytes
                new_ports.append(r)
    except Exception:
        pass

    rdma_ports     = new_ports
    _rdma_prev_time = now


# ── History ring buffer ───────────────────────────────────────────────

def record_history(cpu: float, gpu: float):
    global history_pos, history_count
    cpu_history[history_pos] = cpu
    gpu_history[history_pos] = gpu
    history_pos   = (history_pos + 1) % HISTORY_LEN
    if history_count < HISTORY_LEN:
        history_count += 1


# ── Formatting helpers ────────────────────────────────────────────────

def fmt_bytes(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.1f}G"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f}M"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f}K"
    return f"{n}B"


# ── TUI drawing helpers ────────────────────────────────────────────────

def _bar_color(pct: float) -> int:
    if pct > 90:
        return CP_RED
    if pct > 60:
        return CP_YELLOW
    return CP_GREEN


def _safe_addstr(win, y: int, x: int, s: str, attr: int = 0):
    """Write string at (y, x), clipping to terminal bounds, eating curses errors."""
    rows, cols = win.getmaxyx()
    if y < 0 or y >= rows or x < 0 or x >= cols:
        return
    s = s[:max(0, cols - x - 1)]  # leave 1 cell margin to avoid LINES/COLS wrap errors
    if not s:
        return
    try:
        win.addstr(y, x, s, attr) if attr else win.addstr(y, x, s)
    except curses.error:
        pass


def draw_bar(win, y: int, x: int, width: int, pct: float, color: int):
    """Filled block bar (ACS_BLOCK) with dim bullet background."""
    rows, cols = win.getmaxyx()
    if y < 0 or y >= rows or x < 0:
        return
    width = min(width, cols - x - 1)
    if width <= 0:
        return
    filled = min(width, int(pct / 100.0 * width + 0.5))
    try:
        win.move(y, x)
        win.attron(curses.color_pair(color))
        for _ in range(filled):
            win.addch(curses.ACS_BLOCK)
        win.attroff(curses.color_pair(color))
        win.attron(curses.color_pair(CP_DIM))
        for _ in range(width - filled):
            win.addch(curses.ACS_BULLET)
        win.attroff(curses.color_pair(CP_DIM))
    except curses.error:
        pass


def draw_bar_segmented(win, y: int, x: int, width: int,
                       pct_used: float, pct_bufcache: float,
                       color_used: int, color_cache: int):
    """Two-segment bar (app memory + buf/cache)."""
    rows, cols = win.getmaxyx()
    if y < 0 or y >= rows or x < 0:
        return
    width = min(width, cols - x - 1)
    if width <= 0:
        return
    fu = min(width, int(pct_used     / 100.0 * width + 0.5))
    fc = min(width - fu, int(pct_bufcache / 100.0 * width + 0.5))
    try:
        win.move(y, x)
        win.attron(curses.color_pair(color_used))
        for _ in range(fu):
            win.addch(curses.ACS_BLOCK)
        win.attroff(curses.color_pair(color_used))
        win.attron(curses.color_pair(color_cache))
        for _ in range(fc):
            win.addch(curses.ACS_BLOCK)
        win.attroff(curses.color_pair(color_cache))
        win.attron(curses.color_pair(CP_DIM))
        for _ in range(width - fu - fc):
            win.addch(curses.ACS_BULLET)
        win.attroff(curses.color_pair(CP_DIM))
    except curses.error:
        pass


def draw_history_chart(win, top_y: int, total_w: int, chart_h: int):
    """Full-width rolling CPU/GPU history chart using Unicode block elements."""
    n = min(history_count, HISTORY_LEN)
    if n == 0:
        return

    rows, _ = win.getmaxyx()
    margin  = 2
    label_w = 5   # "100% "
    left_x  = margin + label_w
    right_x = total_w - margin
    avail_w = right_x - left_x
    if avail_w < 10:
        return

    # Title
    try:
        win.addstr(top_y - 1, left_x, "CPU",
                   curses.A_BOLD | curses.color_pair(CP_WHITE))
        win.addstr("/",  curses.color_pair(CP_DIM))
        win.addstr("GPU", curses.A_BOLD | curses.color_pair(CP_CYAN))
        win.addstr(" history", curses.color_pair(CP_DIM))
    except curses.error:
        pass

    # Fixed column width — prevents rescaling as history fills
    col_w   = max(3, avail_w // HISTORY_LEN)
    bar_w   = col_w - 1   # 1-char gap between samples
    cpu_w   = bar_w // 2
    gpu_w   = bar_w - cpu_w

    # Right-align: newest sample on the right
    chart_total = HISTORY_LEN * col_w
    x_start     = left_x + (avail_w - chart_total) + (HISTORY_LEN - n) * col_w

    for s in range(n):
        idx = (history_pos - n + s + HISTORY_LEN) % HISTORY_LEN
        cpu_blocks = int(cpu_history[idx] / 100.0 * chart_h * 8 + 0.5)
        gpu_blocks = int(gpu_history[idx] / 100.0 * chart_h * 8 + 0.5)
        x = x_start + s * col_w
        for row in range(chart_h):
            ry       = top_y + chart_h - 1 - row
            if ry < 0 or ry >= rows:
                continue
            row_base = row * 8
            cpu_fill = max(0, min(8, cpu_blocks - row_base))
            gpu_fill = max(0, min(8, gpu_blocks - row_base))
            try:
                win.addstr(ry, x,
                           BLOCK_CHARS[cpu_fill] * cpu_w,
                           curses.color_pair(CP_GREEN))
                win.addstr(BLOCK_CHARS[gpu_fill] * gpu_w,
                           curses.color_pair(CP_CYAN))
                win.addstr(" ")
            except curses.error:
                pass

    # Y-axis labels
    _safe_addstr(win, top_y,             margin, "100%", curses.color_pair(CP_DIM))
    _safe_addstr(win, top_y + chart_h - 1, margin, "  0%", curses.color_pair(CP_DIM))

    # X-axis time labels
    x_row = top_y + chart_h
    for t in range(0, HISTORY_LEN, 5):
        sx = HISTORY_LEN - 1 - t
        lx = x_start + sx * col_w
        if left_x <= lx < right_x - 2 and x_row < rows:
            _safe_addstr(win, x_row, lx, f"{t:<3}", curses.color_pair(CP_DIM))
    _safe_addstr(win, x_row, margin, "  t=", curses.color_pair(CP_DIM))


# ── Main screen draw ──────────────────────────────────────────────────

def draw_screen(win):
    global last_gpu_util
    rows, cols = win.getmaxyx()
    win.erase()
    y = 0

    # ── Header ────────────────────────────────────────────────────────
    _safe_addstr(win, y, 0, " nv-monitor",
                 curses.A_BOLD | curses.color_pair(CP_CYAN))
    model = cpu_model_name or "Unknown CPU"
    _safe_addstr(win, y, 12, model, curses.color_pair(CP_WHITE))

    uptime = fmt_uptime()
    l1, l5, l15 = get_loadavg()
    info = f"up {uptime}  load {l1:.2f} {l5:.2f} {l15:.2f}"
    _safe_addstr(win, y, cols - len(info) - 1, info)
    y += 1

    try:
        win.attron(curses.color_pair(CP_DIM))
        win.hline(y, 0, curses.ACS_HLINE, cols)
        win.attroff(curses.color_pair(CP_DIM))
    except curses.error:
        pass
    y += 1

    # ── CPU section ───────────────────────────────────────────────────
    cpu_temp = read_cpu_temp()
    cpu_freq = read_cpu_freq_mhz()

    _safe_addstr(win, y, 1, "CPU", curses.A_BOLD | curses.color_pair(CP_YELLOW))
    cpu_hdr = f"  {num_cpus} cores"
    if cpu_freq > 0:
        cpu_hdr += f"  {cpu_freq} MHz"
    if cpu_temp > 0:
        cpu_hdr += f"  {cpu_temp} C"
    _safe_addstr(win, y, 4, cpu_hdr)

    # Overall CPU bar (right half of header row)
    _safe_addstr(win, y, cols // 2 + 1, "Overall: ", curses.A_BOLD)
    bx = cols // 2 + 10
    bw = max(10, cols // 2 - 17)
    draw_bar(win, y, bx, bw, cpu_pct[0], _bar_color(cpu_pct[0]))
    _safe_addstr(win, y, bx + bw, f" {cpu_pct[0]:5.1f}%")
    y += 1

    # Per-core bars — dual-column layout
    half  = (num_cpus + 1) // 2
    lbl_w = 9    # "XX YYYY " (core num + type label)
    bar_w = max(5, cols // 2 - lbl_w - 7)

    for i in range(half):
        if y >= rows - 2:
            break
        for side, cpu_idx, col_off in [
            (0, i,          1),
            (1, i + half,   cols // 2 + 1),
        ]:
            if side == 1 and cpu_idx >= num_cpus:
                continue
            pct   = cpu_pct[cpu_idx + 1]
            color = _bar_color(pct)
            lbl   = cpu_part_label(cpu_idx)
            bx    = col_off + lbl_w - (0 if side == 0 else 1)
            _safe_addstr(win, y, col_off, f"{cpu_idx:2d} ")
            _safe_addstr(win, y, col_off + 3, f"{lbl:<4} ", curses.color_pair(CP_DIM))
            draw_bar(win, y, bx, bar_w, pct, color)
            _safe_addstr(win, y, bx + bar_w, f" {pct:5.1f}%")
        y += 1

    y += 1

    # ── Memory section ────────────────────────────────────────────────
    mi          = read_meminfo()
    pct_app     = (mi.app_kb     / mi.total_kb * 100.0) if mi.total_kb else 0.0
    pct_bc      = (mi.bufcache_kb / mi.total_kb * 100.0) if mi.total_kb else 0.0

    _safe_addstr(win, y, 1, "MEM", curses.A_BOLD | curses.color_pair(CP_BLUE))
    try:
        win.move(y, 5)
        win.addstr("  ")
        win.addstr(fmt_bytes(mi.app_kb     * 1024) + " used",
                   curses.color_pair(CP_GREEN))
        win.addstr(" + ")
        win.addstr(fmt_bytes(mi.bufcache_kb * 1024) + " buf/cache",
                   curses.color_pair(CP_BLUE))
        win.addstr(f" / {fmt_bytes(mi.total_kb * 1024)}")
    except curses.error:
        pass
    y += 1

    bw = max(10, cols - 13)
    draw_bar_segmented(win, y, 4, bw, pct_app, pct_bc, CP_GREEN, CP_BLUE)
    _safe_addstr(win, y, 4 + bw, f" {pct_app + pct_bc:.1f}%")
    y += 1

    if mi.swap_total_kb > 0:
        swap_pct  = mi.swap_used_kb / mi.swap_total_kb * 100.0
        swap_color = CP_RED if swap_pct > 80 else (CP_YELLOW if swap_pct > 40 else CP_MAGENTA)
        _safe_addstr(win, y, 1, "SWP", curses.A_BOLD | curses.color_pair(CP_BLUE))
        _safe_addstr(win, y, 5,
                     f"  {fmt_bytes(mi.swap_used_kb * 1024)} / {fmt_bytes(mi.swap_total_kb * 1024)}")
        y += 1
        bw = max(10, cols - 13)
        draw_bar(win, y, 4, bw, swap_pct, swap_color)
        _safe_addstr(win, y, 4 + bw, f" {swap_pct:.1f}%")
        y += 1

    y += 1
    try:
        win.attron(curses.color_pair(CP_DIM))
        win.hline(y, 0, curses.ACS_HLINE, cols)
        win.attroff(curses.color_pair(CP_DIM))
    except curses.error:
        pass
    y += 1

    # ── GPU section ───────────────────────────────────────────────────
    if not nvml_ok:
        _safe_addstr(win, y, 1, "GPU: NVML not available",
                     curses.color_pair(CP_RED))
        y += 2
    else:
        gpu_util_sum = 0.0
        gpu_util_n   = 0

        for d in range(gpu_count):
            if y >= rows - 4:
                break
            handle = nvml.device_handle(d)
            if handle is None:
                continue

            name = nvml.device_name(handle)

            # Utilization
            util_gpu = 0
            if use_tegra_gpu:
                t = read_tegra_gpu_util()
                if t >= 0:
                    util_gpu = t
            else:
                u = nvml.utilization(handle)
                if u:
                    util_gpu = u[0]
            gpu_util_sum += util_gpu
            gpu_util_n   += 1

            # Temperature
            temp = 0
            if use_tegra_gpu and tegra_gpu_therm_zone >= 0:
                t = read_tegra_gpu_temp()
                if t > 0:
                    temp = t
            else:
                t = nvml.temperature(handle)
                if t is not None:
                    temp = t

            power_mw = nvml.power_usage_mw(handle)
            clk_gfx  = nvml.clock_mhz(handle, NVML_CLOCK_GRAPHICS)
            clk_mem  = nvml.clock_mhz(handle, NVML_CLOCK_MEM)
            fan      = nvml.fan_speed(handle)

            # GPU header line
            _safe_addstr(win, y, 1, f"GPU {d}",
                         curses.A_BOLD | curses.color_pair(CP_CYAN))
            hdr = f"  {name}  {temp} C"
            if power_mw is not None:
                hdr += f"  {power_mw / 1000:.1f}W"
            if clk_gfx:
                hdr += f"  {clk_gfx} MHz"
            if fan is not None:
                hdr += f"  Fan {fan}%"
            _safe_addstr(win, y, 7, hdr)
            y += 1

            # GPU utilization bar
            _safe_addstr(win, y, 1, "  GPU ")
            bx = 7
            bw = max(10, cols - bx - 7)
            gpu_bar_color = CP_RED if util_gpu > 90 else (CP_YELLOW if util_gpu > 60 else CP_CYAN)
            draw_bar(win, y, bx, bw, float(util_gpu), gpu_bar_color)
            _safe_addstr(win, y, bx + bw + 1, f"{util_gpu:3d}%")
            y += 1

            # VRAM
            mem_info = nvml.memory_info(handle)
            if mem_info and mem_info[0] > 0:
                total_b, _, used_b = mem_info
                mem_pct   = used_b / total_b * 100.0
                mem_color = CP_RED if mem_pct > 90 else (CP_YELLOW if mem_pct > 60 else CP_MAGENTA)
                _safe_addstr(win, y, 1, "  VRAM")
                bx = 7
                bw = max(10, cols - bx - 18)
                draw_bar(win, y, bx, bw, mem_pct, mem_color)
                _safe_addstr(win, y, bx + bw + 1,
                             f"{fmt_bytes(used_b)}/{fmt_bytes(total_b)}")
            else:
                _safe_addstr(win, y, 1, "  VRAM")
                _safe_addstr(win, y, 7, "  unified memory (shared with CPU)",
                             curses.color_pair(CP_WHITE))
            y += 1

            # ENC/DEC encoder/decoder utilization
            enc = nvml.encoder_utilization(handle)
            dec = nvml.decoder_utilization(handle)
            if enc is not None or dec is not None:
                enc_dec = ""
                if enc is not None:
                    enc_dec += f"ENC {enc}%  "
                if dec is not None:
                    enc_dec += f"DEC {dec}%"
                _safe_addstr(win, y, 3, enc_dec)
                y += 1

            y += 1

            # ── GPU processes ──────────────────────────────────────────
            all_procs: List[GpuProc] = []
            seen_pids: set = set()

            for pid, mem in nvml.compute_processes(handle):
                all_procs.append(GpuProc(
                    pid=pid, mem_bytes=mem, type="C",
                    cpu_pct=calc_proc_cpu_pct(pid),
                    name=get_proc_cmdline(pid),
                    user=get_proc_user(pid),
                ))
                seen_pids.add(pid)

            for pid, mem in nvml.graphics_processes(handle):
                if pid not in seen_pids:
                    all_procs.append(GpuProc(
                        pid=pid, mem_bytes=mem, type="G",
                        cpu_pct=calc_proc_cpu_pct(pid),
                        name=get_proc_cmdline(pid),
                        user=get_proc_user(pid),
                    ))
                    seen_pids.add(pid)

            # Tegra fallback: scan /proc for GPU fd holders
            if not all_procs and use_tegra_gpu:
                all_procs = scan_tegra_gpu_procs()

            update_proc_cpu_snapshots(all_procs)

            # Sort
            if sort_mode == 0:
                all_procs.sort(key=lambda p: p.mem_bytes, reverse=True)
            else:
                all_procs.sort(key=lambda p: p.pid)

            if all_procs and y < rows - 2:
                hdr_str = (f"  {'PID':<8} {'USER':<12} {'TYPE':<4}"
                           f" {'CPU%':>7} {'GPU MEM':<12} COMMAND")
                _safe_addstr(win, y, 1, hdr_str,
                             curses.A_BOLD | curses.color_pair(CP_WHITE))
                y += 1

                for p in all_procs:
                    if y >= rows - 2:
                        break
                    mb = fmt_bytes(p.mem_bytes) if p.mem_bytes > 0 else "N/A"
                    name_max  = max(10, cols - 54)
                    truncname = p.name[:name_max]
                    type_color = CP_MAGENTA if p.type == "C" else CP_WHITE
                    try:
                        win.addstr(y, 1, f"  {p.pid:<8} {p.user:<12} ")
                        win.addstr(f"{p.type:<4}", curses.color_pair(type_color))
                        win.addstr(f" {p.cpu_pct:6.1f}% {mb:<12} {truncname}")
                    except curses.error:
                        pass
                    y += 1

                # Non-GPU processes summary row
                gpu_proc_cpu = sum(p.cpu_pct for p in all_procs)
                other_cpu    = max(0.0, cpu_pct[0] * num_cpus - gpu_proc_cpu)
                _safe_addstr(win, y, 1,
                             f"  {'':8} {'':12} {'':4} {other_cpu:6.1f}%"
                             f" {'':12} (other processes)",
                             curses.color_pair(CP_DIM))
                y += 1

        last_gpu_util = (gpu_util_sum / gpu_util_n) if gpu_util_n > 0 else 0.0

    # ── History chart (full width, pinned to bottom) ────────────────
    record_history(cpu_pct[0], last_gpu_util)
    chart_h   = 5
    chart_top = rows - 3 - chart_h   # -3: footer line + x-axis row + gap
    if chart_top > y + 1 and cols > 20:
        draw_history_chart(win, chart_top, cols, chart_h)

    # ── Footer ────────────────────────────────────────────────────────
    try:
        win.hline(rows - 1, 0, curses.ACS_HLINE, cols)
        win.move(rows - 1, 1)
        win.addstr(" q",  curses.A_BOLD | curses.color_pair(CP_WHITE))
        win.addstr(":quit ")
        win.addstr("s",   curses.A_BOLD | curses.color_pair(CP_WHITE))
        win.addstr(":sort ")
        win.addstr("+/-", curses.A_BOLD | curses.color_pair(CP_WHITE))
        win.addstr(":speed  ")
        win.addstr(f"{delay_ms / 1000:.1f}s", curses.color_pair(CP_DIM))
        _safe_addstr(win, rows - 1, cols - len(VERSION) - 2,
                     VERSION, curses.color_pair(CP_DIM))
    except curses.error:
        pass

    win.refresh()


# ── CSV logging ───────────────────────────────────────────────────────

def log_csv_header(f):
    cols = ["timestamp", "cpu_avg_pct"]
    for i in range(num_cpus):
        cols.append(f"cpu{i}_pct")
    cols += ["cpu_temp_c", "cpu_freq_mhz",
             "mem_used_kb", "mem_total_kb", "mem_bufcache_kb",
             "swap_used_kb", "swap_total_kb"]
    for g in range(gpu_count):
        cols += [f"gpu{g}_util_pct", f"gpu{g}_temp_c",
                 f"gpu{g}_power_mw", f"gpu{g}_clock_mhz"]
    for r in rdma_ports:
        cols += [f"rdma_{r.device}_p{r.port}_xmit_Bps",
                 f"rdma_{r.device}_p{r.port}_recv_Bps"]
    f.write(",".join(cols) + "\n")
    f.flush()


def log_csv_row(f):
    # Timestamp with milliseconds
    t   = time.time()
    ms  = int(t * 1000) % 1000
    now = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t)) + f".{ms:03d}"

    row = [now, f"{cpu_pct[0]:.1f}"]
    for i in range(1, num_cpus + 1):
        row.append(f"{cpu_pct[i]:.1f}")
    row += [str(read_cpu_temp()), str(read_cpu_freq_mhz())]

    mi = read_meminfo()
    row += [str(mi.app_kb), str(mi.total_kb), str(mi.bufcache_kb),
            str(mi.swap_used_kb), str(mi.swap_total_kb)]

    for d in range(gpu_count):
        handle = nvml.device_handle(d) if nvml_ok else None
        if handle is not None:
            util_gpu = 0
            if use_tegra_gpu:
                t2 = read_tegra_gpu_util()
                if t2 >= 0:
                    util_gpu = t2
            else:
                u = nvml.utilization(handle)
                if u:
                    util_gpu = u[0]

            temp = 0
            if use_tegra_gpu and tegra_gpu_therm_zone >= 0:
                t2 = read_tegra_gpu_temp()
                if t2 > 0:
                    temp = t2
            else:
                t2 = nvml.temperature(handle)
                if t2 is not None:
                    temp = t2

            power_mw = nvml.power_usage_mw(handle) or 0
            clk      = nvml.clock_mhz(handle, NVML_CLOCK_GRAPHICS) or 0
            row += [str(util_gpu), str(temp), str(power_mw), str(clk)]
        else:
            row += ["", "", "", ""]

    for r in rdma_ports:
        row += [f"{r.xmit_bytes_sec:.0f}", f"{r.recv_bytes_sec:.0f}"]

    f.write(",".join(row) + "\n")
    f.flush()


# ── Prometheus / OpenMetrics exporter ─────────────────────────────────

def format_metrics() -> str:
    """Build an OpenMetrics-formatted response body.

    Python f-strings always use '.' as the decimal separator regardless of
    locale, so no LC_NUMERIC workaround is needed (unlike the C version's
    setlocale(LC_NUMERIC, "C") call).
    """
    lines: List[str] = []

    def pm(s: str):
        lines.append(s)

    pm(f'# HELP nv_build_info nv-monitor version\n'
       f'# TYPE nv_build_info gauge\n'
       f'nv_build_info{{version="{VERSION}"}} 1')

    # Uptime
    try:
        with open("/proc/uptime") as f:
            uptime_s = int(float(f.read().split()[0]))
        pm(f'# HELP nv_uptime_seconds System uptime\n'
           f'# TYPE nv_uptime_seconds gauge\n'
           f'nv_uptime_seconds {uptime_s}')
    except Exception:
        pass

    # Load average
    l1, l5, l15 = get_loadavg()
    pm(f'# HELP nv_load_average System load average\n'
       f'# TYPE nv_load_average gauge\n'
       f'nv_load_average{{interval="1m"}} {l1:.2f}\n'
       f'nv_load_average{{interval="5m"}} {l5:.2f}\n'
       f'nv_load_average{{interval="15m"}} {l15:.2f}')

    # CPU usage
    pm(f'# HELP nv_cpu_usage_percent CPU utilization\n'
       f'# TYPE nv_cpu_usage_percent gauge\n'
       f'nv_cpu_usage_percent{{cpu="overall"}} {cpu_pct[0]:.1f}')
    for i in range(num_cpus):
        lbl = cpu_part_label(i)
        if lbl:
            pm(f'nv_cpu_usage_percent{{cpu="{i}",type="{lbl}"}} {cpu_pct[i + 1]:.1f}')
        else:
            pm(f'nv_cpu_usage_percent{{cpu="{i}"}} {cpu_pct[i + 1]:.1f}')

    pm(f'# HELP nv_cpu_temperature_celsius CPU temperature\n'
       f'# TYPE nv_cpu_temperature_celsius gauge\n'
       f'nv_cpu_temperature_celsius {read_cpu_temp()}')

    pm(f'# HELP nv_cpu_frequency_mhz CPU frequency\n'
       f'# TYPE nv_cpu_frequency_mhz gauge\n'
       f'nv_cpu_frequency_mhz {read_cpu_freq_mhz()}')

    # Memory
    mi = read_meminfo()
    pm(f'# HELP nv_memory_total_bytes Total system memory\n'
       f'# TYPE nv_memory_total_bytes gauge\n'
       f'nv_memory_total_bytes {mi.total_kb * 1024}\n'
       f'# HELP nv_memory_used_bytes Application memory used\n'
       f'# TYPE nv_memory_used_bytes gauge\n'
       f'nv_memory_used_bytes {mi.app_kb * 1024}\n'
       f'# HELP nv_memory_bufcache_bytes Buffer and cache memory\n'
       f'# TYPE nv_memory_bufcache_bytes gauge\n'
       f'nv_memory_bufcache_bytes {mi.bufcache_kb * 1024}')
    if mi.swap_total_kb > 0:
        pm(f'# HELP nv_swap_total_bytes Total swap\n'
           f'# TYPE nv_swap_total_bytes gauge\n'
           f'nv_swap_total_bytes {mi.swap_total_kb * 1024}\n'
           f'# HELP nv_swap_used_bytes Swap used\n'
           f'# TYPE nv_swap_used_bytes gauge\n'
           f'nv_swap_used_bytes {mi.swap_used_kb * 1024}')

    # GPU metrics
    if nvml_ok:
        count    = nvml.device_count()
        gpu_data = []
        for d in range(min(count, 8)):
            handle = nvml.device_handle(d)
            if handle is None:
                continue
            gd: Dict = {
                "name":     nvml.device_name(handle),
                "util":     0,
                "temp":     0,
                "power_mw": None,
                "clk_gfx":  None,
                "clk_mem":  None,
                "mem_total": None,
                "mem_used":  None,
                "fan":       None,
                "enc":       None,
                "dec":       None,
            }
            if use_tegra_gpu:
                t2 = read_tegra_gpu_util()
                if t2 >= 0:
                    gd["util"] = t2
            else:
                u = nvml.utilization(handle)
                if u:
                    gd["util"] = u[0]

            if use_tegra_gpu and tegra_gpu_therm_zone >= 0:
                t2 = read_tegra_gpu_temp()
                if t2 > 0:
                    gd["temp"] = t2
            else:
                t2 = nvml.temperature(handle)
                if t2 is not None:
                    gd["temp"] = t2

            gd["power_mw"] = nvml.power_usage_mw(handle)
            gd["clk_gfx"]  = nvml.clock_mhz(handle, NVML_CLOCK_GRAPHICS)
            gd["clk_mem"]  = nvml.clock_mhz(handle, NVML_CLOCK_MEM)
            mem_info = nvml.memory_info(handle)
            if mem_info and mem_info[0] > 0:
                gd["mem_total"] = mem_info[0]
                gd["mem_used"]  = mem_info[2]
            gd["fan"] = nvml.fan_speed(handle)
            gd["enc"] = nvml.encoder_utilization(handle)
            gd["dec"] = nvml.decoder_utilization(handle)
            gpu_data.append((d, gd))

        if gpu_data:
            pm('# HELP nv_gpu_info GPU device information\n# TYPE nv_gpu_info gauge')
            for d, gd in gpu_data:
                pm(f'nv_gpu_info{{gpu="{d}",name="{gd["name"]}"}} 1')

            pm('# HELP nv_gpu_utilization_percent GPU compute utilization\n'
               '# TYPE nv_gpu_utilization_percent gauge')
            for d, gd in gpu_data:
                pm(f'nv_gpu_utilization_percent{{gpu="{d}"}} {gd["util"]}')

            pm('# HELP nv_gpu_temperature_celsius GPU temperature\n'
               '# TYPE nv_gpu_temperature_celsius gauge')
            for d, gd in gpu_data:
                pm(f'nv_gpu_temperature_celsius{{gpu="{d}"}} {gd["temp"]}')

            pm('# HELP nv_gpu_power_watts GPU power draw\n'
               '# TYPE nv_gpu_power_watts gauge')
            for d, gd in gpu_data:
                if gd["power_mw"] is not None:
                    pm(f'nv_gpu_power_watts{{gpu="{d}"}} {gd["power_mw"] / 1000:.1f}')

            pm('# HELP nv_gpu_clock_mhz GPU clock speed\n'
               '# TYPE nv_gpu_clock_mhz gauge')
            for d, gd in gpu_data:
                if gd["clk_gfx"]:
                    pm(f'nv_gpu_clock_mhz{{gpu="{d}",type="graphics"}} {gd["clk_gfx"]}')
                if gd["clk_mem"]:
                    pm(f'nv_gpu_clock_mhz{{gpu="{d}",type="memory"}} {gd["clk_mem"]}')

            pm('# HELP nv_gpu_memory_total_bytes GPU memory total\n'
               '# TYPE nv_gpu_memory_total_bytes gauge')
            for d, gd in gpu_data:
                if gd["mem_total"] is not None:
                    pm(f'nv_gpu_memory_total_bytes{{gpu="{d}"}} {gd["mem_total"]}')

            pm('# HELP nv_gpu_memory_used_bytes GPU memory used\n'
               '# TYPE nv_gpu_memory_used_bytes gauge')
            for d, gd in gpu_data:
                if gd["mem_used"] is not None:
                    pm(f'nv_gpu_memory_used_bytes{{gpu="{d}"}} {gd["mem_used"]}')

            pm('# HELP nv_gpu_fan_speed_percent GPU fan speed\n'
               '# TYPE nv_gpu_fan_speed_percent gauge')
            for d, gd in gpu_data:
                if gd["fan"] is not None:
                    pm(f'nv_gpu_fan_speed_percent{{gpu="{d}"}} {gd["fan"]}')

            pm('# HELP nv_gpu_encoder_utilization_percent GPU encoder utilization\n'
               '# TYPE nv_gpu_encoder_utilization_percent gauge')
            for d, gd in gpu_data:
                if gd["enc"] is not None:
                    pm(f'nv_gpu_encoder_utilization_percent{{gpu="{d}"}} {gd["enc"]}')

            pm('# HELP nv_gpu_decoder_utilization_percent GPU decoder utilization\n'
               '# TYPE nv_gpu_decoder_utilization_percent gauge')
            for d, gd in gpu_data:
                if gd["dec"] is not None:
                    pm(f'nv_gpu_decoder_utilization_percent{{gpu="{d}"}} {gd["dec"]}')

    # RDMA / InfiniBand
    if rdma_available and rdma_ports:
        pm('# HELP nv_rdma_info RDMA port information\n# TYPE nv_rdma_info gauge')
        for r in rdma_ports:
            pm(f'nv_rdma_info{{device="{r.device}",port="{r.port}",'
               f'state="{r.state}",rate="{r.rate}"}} 1')

        for metric, attr, hlp, typ in [
            ("nv_rdma_xmit_bytes_total",   "xmit_bytes", "Total bytes transmitted",   "counter"),
            ("nv_rdma_recv_bytes_total",   "recv_bytes", "Total bytes received",       "counter"),
            ("nv_rdma_xmit_packets_total", "xmit_pkts",  "Total packets transmitted", "counter"),
            ("nv_rdma_recv_packets_total", "recv_pkts",   "Total packets received",   "counter"),
            ("nv_rdma_errors_total",       "errors",      "Total RDMA errors",        "counter"),
        ]:
            pm(f'# HELP {metric} {hlp}\n# TYPE {metric} {typ}')
            for r in rdma_ports:
                pm(f'{metric}{{device="{r.device}",port="{r.port}"}} {getattr(r, attr)}')

        for metric, attr, hlp in [
            ("nv_rdma_xmit_bytes_per_second", "xmit_bytes_sec", "Transmit throughput"),
            ("nv_rdma_recv_bytes_per_second", "recv_bytes_sec", "Receive throughput"),
        ]:
            pm(f'# HELP {metric} {hlp}\n# TYPE {metric} gauge')
            for r in rdma_ports:
                pm(f'{metric}{{device="{r.device}",port="{r.port}"}} {getattr(r, attr):.0f}')

    return "\n".join(lines) + "\n"


class PrometheusServer(threading.Thread):
    """Minimal HTTP server thread serving /metrics in OpenMetrics format.
    Mirrors the C version's prom_server pthread + prom_handle function."""

    def __init__(self, port: int, token: Optional[str]):
        super().__init__(daemon=True, name="prom-server")
        self.port  = port
        self.token = token
        self._sock: Optional[socket.socket] = None

    def run(self):
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind(("0.0.0.0", self.port))
            self._sock.listen(4)
            self._sock.settimeout(1.0)   # 1-s poll for clean shutdown (mirrors poll() in C)
            print(f"Prometheus metrics at http://0.0.0.0:{self.port}/metrics",
                  file=sys.stderr)
            while not g_quit:
                try:
                    conn, _ = self._sock.accept()
                except socket.timeout:
                    continue
                try:
                    self._handle(conn)
                finally:
                    conn.close()
        except Exception as e:
            print(f"Prometheus server error: {e}", file=sys.stderr)
        finally:
            if self._sock:
                self._sock.close()

    def _handle(self, conn: socket.socket):
        conn.settimeout(2.0)
        try:
            req = conn.recv(512).decode("utf-8", errors="replace")
        except Exception:
            return

        # Bearer token auth
        if self.token:
            if f"Authorization: Bearer {self.token}" not in req:
                conn.sendall(
                    b"HTTP/1.1 401 Unauthorized\r\n"
                    b"Content-Type: text/plain\r\n"
                    b"Connection: close\r\n\r\n"
                    b"Unauthorized\n"
                )
                return

        if "GET /metrics" in req:
            body = format_metrics().encode("utf-8")
            hdr  = (
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/plain; version=0.0.4; charset=utf-8\r\n"
                f"Content-Length: {len(body)}\r\n"
                "Connection: close\r\n\r\n"
            ).encode("utf-8")
            try:
                conn.sendall(hdr + body)
            except Exception:
                pass
        else:
            try:
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n"
                    b"Connection: close\r\n\r\n"
                    b"<html><body><h1>nv-monitor</h1>"
                    b"<p><a href=\"/metrics\">Metrics</a></p>"
                    b"</body></html>\n"
                )
            except Exception:
                pass


# ── TUI entry point (called by curses.wrapper) ────────────────────────

def _tui_main(win):
    global g_quit, sort_mode, delay_ms

    curses.curs_set(0)
    try:
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(CP_RED,     curses.COLOR_RED,     -1)
        curses.init_pair(CP_GREEN,   curses.COLOR_GREEN,   -1)
        curses.init_pair(CP_YELLOW,  curses.COLOR_YELLOW,  -1)
        curses.init_pair(CP_BLUE,    curses.COLOR_BLUE,    -1)
        curses.init_pair(CP_MAGENTA, curses.COLOR_MAGENTA, -1)
        curses.init_pair(CP_CYAN,    curses.COLOR_CYAN,    -1)
        curses.init_pair(CP_WHITE,   curses.COLOR_WHITE,   -1)
        # Pair 8 = dim gray. 244 is a 256-color index; fall back to default fg.
        try:
            curses.init_pair(CP_DIM, 244, -1)
        except curses.error:
            curses.init_pair(CP_DIM, -1, -1)
    except curses.error:
        pass

    win.nodelay(True)
    win.keypad(True)

    log_elapsed = 0

    while not g_quit:
        compute_cpu_usage()
        read_rdma_ports()
        win.clearok(True)       # force full repaint (mirrors clearok(stdscr,TRUE) in C)
        draw_screen(win)

        if log_fp:
            log_elapsed += delay_ms
            if log_elapsed >= log_interval_ms:
                log_csv_row(log_fp)
                log_elapsed = 0

        # Poll for input within the refresh interval
        elapsed = 0
        while elapsed < delay_ms and not g_quit:
            ch = win.getch()
            if ch in (ord('q'), ord('Q'), 27):        # q, Q, Esc
                g_quit = True
                break
            elif ch in (ord('s'), ord('S')):           # toggle sort
                sort_mode = (sort_mode + 1) % 2
                break
            elif ch in (ord('+'), ord('=')):           # faster refresh
                if delay_ms > 250:
                    delay_ms -= 250
            elif ch in (ord('-'), ord('_')):           # slower refresh
                if delay_ms < 5000:
                    delay_ms += 250
            elif ch == curses.KEY_RESIZE:              # terminal resize
                break
            time.sleep(0.05)
            elapsed += 50


# ── main ──────────────────────────────────────────────────────────────

def main() -> int:
    global nvml_ok, gpu_count, use_tegra_gpu, cpu_model_name
    global cpu_parts, prev_ticks, num_cpus
    global g_quit, delay_ms, log_fp, log_interval_ms, no_ui, prom_port, prom_token

    # setlocale for ncurses Unicode support.  Python f-strings/format() always
    # produce "." as the decimal separator (no locale dependency), so the
    # C version's setlocale(LC_NUMERIC,"C") fix is implicit and not needed.
    locale.setlocale(locale.LC_ALL, "")

    parser = argparse.ArgumentParser(
        prog="nv-monitor",
        description="System monitor for NVIDIA GPU systems",
        add_help=False,
    )
    parser.add_argument("-l", metavar="FILE",  dest="log_path",
                        help="Log statistics to CSV file")
    parser.add_argument("-i", metavar="MS",    dest="log_interval", type=int, default=1000,
                        help="Log interval in milliseconds (default: 1000)")
    parser.add_argument("-n", dest="no_ui",    action="store_true",
                        help="No UI (headless mode, requires -l or -p)")
    parser.add_argument("-p", metavar="PORT",  dest="prom_port", type=int, default=0,
                        help="Expose Prometheus metrics on PORT")
    parser.add_argument("-t", metavar="TOKEN", dest="prom_token",
                        help="Require Bearer token for /metrics")
    parser.add_argument("-r", metavar="MS",    dest="refresh", type=int, default=1000,
                        help="UI refresh interval in milliseconds (default: 1000)")
    parser.add_argument("-v", dest="version",  action="store_true",
                        help="Show version")
    parser.add_argument("-h", dest="help",     action="store_true",
                        help="Show this help")
    args = parser.parse_args()

    if args.help:
        parser.print_help()
        print("\nExamples:")
        print(f"  nv-monitor.py -l stats.csv              TUI + logging every 1s")
        print(f"  nv-monitor.py -l stats.csv -i 5000      TUI + logging every 5s")
        print(f"  nv-monitor.py -n -l stats.csv -i 500    Headless, log every 500ms")
        print(f"  nv-monitor.py -r 2000                   TUI refreshing every 2s")
        return 0

    if args.version:
        print(f"nv-monitor {VERSION}")
        return 0

    log_interval_ms = max(100, args.log_interval)
    delay_ms        = max(250, args.refresh)
    no_ui           = args.no_ui
    prom_port       = args.prom_port
    # CLI flag takes precedence, then env var (same precedence as C version)
    prom_token = args.prom_token or os.environ.get("NV_MONITOR_TOKEN")

    if no_ui and not args.log_path and not prom_port:
        print("Error: -n (no UI) requires -l <file> or -p <port>", file=sys.stderr)
        return 1

    # Open log file
    if args.log_path:
        try:
            log_fp = open(args.log_path, "w")
        except OSError as e:
            print(f"{args.log_path}: {e}", file=sys.stderr)
            return 1

    # Signal handlers
    def _on_signal(sig, frame):
        global g_quit
        g_quit = True
    signal.signal(signal.SIGINT,  _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # Load NVML
    nvml_ok   = nvml.load()
    if nvml_ok:
        gpu_count = nvml.device_count()

    # CPU info
    cpu_model_name = read_cpu_model_name()
    cpu_parts      = read_cpu_part_ids()

    # Tegra GPU detection (Jetson fallback)
    detect_tegra_gpu()
    detect_tegra_gpu_dev()
    if tegra_gpu_available:
        use_tegra_gpu = True

    # Initial CPU tick read — brief pause so first delta is meaningful
    ticks, n = read_cpu_ticks()
    for i, t in enumerate(ticks):
        prev_ticks[i] = t
    num_cpus = n
    time.sleep(0.1)
    compute_cpu_usage()

    # RDMA initial scan
    read_rdma_ports()

    # CSV header (written after first sample so we know num_cpus)
    if log_fp:
        log_csv_header(log_fp)

    # Start Prometheus exporter
    prom_server: Optional[PrometheusServer] = None
    if prom_port:
        prom_server = PrometheusServer(prom_port, prom_token)
        prom_server.start()

    if no_ui:
        headless_interval = log_interval_ms if log_fp else delay_ms
        if log_fp:
            print(f"Logging to {args.log_path} every {headless_interval}ms (Ctrl+C to stop)",
                  file=sys.stderr)
        else:
            print("Running headless (Ctrl+C to stop)", file=sys.stderr)
        while not g_quit:
            compute_cpu_usage()
            read_rdma_ports()
            if log_fp:
                log_csv_row(log_fp)
            time.sleep(headless_interval / 1000.0)
        print("\nStopped.", file=sys.stderr)
    else:
        try:
            curses.wrapper(_tui_main)
        except KeyboardInterrupt:
            pass

    # Cleanup
    if log_fp:
        log_fp.close()
    nvml.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
