from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import psutil

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_GB = 1_073_741_824   # bytes → GiB
_UJ_TO_J = 1e-6       # microjoules → joules

# RAPL sysfs
_RAPL_BASE = "/sys/class/powercap/intel-rapl"
_RAPL_PKG_PATTERN    = "{base}/intel-rapl:{p}/energy_uj"
_RAPL_DRAM_PATTERN   = "{base}/intel-rapl:{p}/intel-rapl:{p}:{sub}/energy_uj"
_RAPL_DRAM_NAME      = "{base}/intel-rapl:{p}/intel-rapl:{p}:{sub}/name"

# cgroup memory
_CG2_USAGE = "/sys/fs/cgroup/memory.current"
_CG2_LIMIT = "/sys/fs/cgroup/memory.max"
_CG1_USAGE = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
_CG1_LIMIT = "/sys/fs/cgroup/memory/memory.limit_in_bytes"

# ---------------------------------------------------------------------------
# RAPL helpers
# ---------------------------------------------------------------------------

def _find_rapl_pkg_paths() -> List[str]:
    paths: List[str] = []
    if not os.path.isdir(_RAPL_BASE):
        return paths
    p = 0
    while True:
        path = _RAPL_PKG_PATTERN.format(base=_RAPL_BASE, p=p)
        if not os.path.exists(path):
            break
        if os.access(path, os.R_OK):
            paths.append(path)
        p += 1
    return paths


def _find_rapl_dram_paths() -> List[str]:
    """Find RAPL DRAM sub-domain paths (Intel only, varies by CPU model)."""
    paths: List[str] = []
    if not os.path.isdir(_RAPL_BASE):
        return paths
    p = 0
    while True:
        base_path = _RAPL_PKG_PATTERN.format(base=_RAPL_BASE, p=p)
        if not os.path.exists(base_path):
            break
        sub = 0
        while True:
            name_path = _RAPL_DRAM_NAME.format(base=_RAPL_BASE, p=p, sub=sub)
            energy_path = _RAPL_DRAM_PATTERN.format(base=_RAPL_BASE, p=p, sub=sub)
            if not os.path.exists(name_path):
                break
            try:
                with open(name_path) as f:
                    domain_name = f.read().strip()
                if domain_name == "dram" and os.access(energy_path, os.R_OK):
                    paths.append(energy_path)
            except OSError:
                pass
            sub += 1
        p += 1
    return paths


def _sum_rapl(paths: List[str]) -> float:
    total = 0.0
    for path in paths:
        with open(path, "r", buffering=1) as f:
            total += float(f.read())
    return total


# ---------------------------------------------------------------------------
# cgroup memory
# ---------------------------------------------------------------------------

def _cgroup_memory() -> Optional[Tuple[float, float]]:
    # v2
    if os.path.exists(_CG2_USAGE) and os.path.exists(_CG2_LIMIT):
        try:
            usage = int(open(_CG2_USAGE).read().strip())
            raw = open(_CG2_LIMIT).read().strip()
            limit = int(raw) if raw != "max" else None
            if limit and limit < (1 << 62):
                return usage / _GB, usage / limit * 100.0
        except (OSError, ValueError):
            pass
    # v1
    if os.path.exists(_CG1_USAGE) and os.path.exists(_CG1_LIMIT):
        try:
            usage = int(open(_CG1_USAGE).read().strip())
            limit = int(open(_CG1_LIMIT).read().strip())
            if limit and limit < (1 << 62):
                return usage / _GB, usage / limit * 100.0
        except (OSError, ValueError):
            pass
    return None


# ---------------------------------------------------------------------------
# NVML (NVIDIA GPU — optional)
# ---------------------------------------------------------------------------

def _init_nvml():
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        return pynvml
    except Exception:
        return None


def _nvml_handle(pynvml, index: int = 0):
    try:
        return pynvml.nvmlDeviceGetHandleByIndex(index)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Apple GPU via psutil (macOS only)
# ---------------------------------------------------------------------------

_IS_MACOS = sys.platform == "darwin"


def _apple_gpu_stats() -> Optional[Dict[str, Optional[float]]]:
    """
    Read Apple GPU metrics via psutil if available (psutil >= 5.9.4 on macOS).
    Returns dict with usage_pct, memory_used_gb, memory_used_pct, or None.
    """
    if not _IS_MACOS:
        return None
    try:
        # psutil exposes gpu_usage via cpu_stats on some builds; check sensors path
        if not hasattr(psutil, "sensors_gpu"):
            return None
        gpus = psutil.sensors_gpu()
        if not gpus:
            return None
        g = gpus[0]
        mem_used_gb = getattr(g, "memory_used", None)
        mem_total = getattr(g, "memory_total", None)
        if mem_used_gb is not None:
            mem_used_gb = mem_used_gb / _GB
        mem_pct = (
            (mem_used_gb * _GB / mem_total * 100.0)
            if (mem_used_gb is not None and mem_total)
            else None
        )
        return {
            "usage_pct": getattr(g, "load", None),
            "memory_used_gb": mem_used_gb,
            "memory_used_pct": mem_pct,
            "power_watts": getattr(g, "power", None),
            "temperature_c": getattr(g, "temperature", None),
            "memory_power_watts": None,  # not exposed by psutil
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Disk helpers
# ---------------------------------------------------------------------------

def _disk_usage(path: str = "/") -> Tuple[float, float]:
    """Return (used_gb, used_pct) for the given mount point."""
    try:
        d = psutil.disk_usage(path)
        return d.used / _GB, d.percent
    except Exception:
        return 0.0, 0.0


# ---------------------------------------------------------------------------
# RAM power estimate (no RAPL DRAM available)
# ---------------------------------------------------------------------------
# Rough industry estimate: ~3 W per 8 GB of RAM in active use.
_RAM_W_PER_GB = 0.375


def _estimate_ram_power(ram_used_gb: float) -> float:
    return round(ram_used_gb * _RAM_W_PER_GB, 4)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class Tracker:
    __slots__ = (
        # RAPL
        "_rapl_pkg_paths", "_rapl_pkg_ok",
        "_rapl_dram_paths", "_rapl_dram_ok",
        "_last_pkg_uj", "_last_dram_uj",
        # NVML
        "_nvml", "_nvml_handle",
        # state
        "_last_ts",
        "_hw_mode",
        # config
        "_disk_path",
    )

    def __init__(self, disk_path: str = "/") -> None:
        self._disk_path = disk_path

        # ---- RAPL package ----
        self._rapl_pkg_ok = False
        self._rapl_pkg_paths: List[str] = []
        try:
            paths = _find_rapl_pkg_paths()
            if paths:
                self._rapl_pkg_paths = paths
                self._rapl_pkg_ok = True
        except (OSError, PermissionError):
            pass

        # ---- RAPL DRAM ----
        self._rapl_dram_ok = False
        self._rapl_dram_paths: List[str] = []
        try:
            paths = _find_rapl_dram_paths()
            if paths:
                self._rapl_dram_paths = paths
                self._rapl_dram_ok = True
        except (OSError, PermissionError):
            pass

        # ---- NVML ----
        self._nvml = _init_nvml()
        self._nvml_handle = None
        if self._nvml is not None:
            self._nvml_handle = _nvml_handle(self._nvml)
            if self._nvml_handle is None:
                self._nvml = None

        # ---- initial energy baseline ----
        self._last_ts: float = time.perf_counter()
        self._last_pkg_uj: float = (
            _sum_rapl(self._rapl_pkg_paths) if self._rapl_pkg_ok else 0.0
        )
        self._last_dram_uj: float = (
            _sum_rapl(self._rapl_dram_paths) if self._rapl_dram_ok else 0.0
        )

        # ---- hardware mode label ----
        parts: List[str] = []
        if self._rapl_pkg_ok:
            parts.append("RAPL")
        if self._nvml is not None:
            parts.append("NVML")
        elif _IS_MACOS:
            parts.append("AppleGPU")
        if not parts:
            parts.append("psutil")
        self._hw_mode: str = "+".join(parts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def checkpoint(self) -> Dict[str, object]:
        now_ts = time.perf_counter()
        elapsed = now_ts - self._last_ts

        # ── CPU energy (RAPL package) ─────────────────────────────────
        cpu_energy_j: Optional[float] = None
        cpu_power_w: Optional[float] = None
        if self._rapl_pkg_ok:
            try:
                now_uj = _sum_rapl(self._rapl_pkg_paths)
                delta_uj = max(0.0, now_uj - self._last_pkg_uj)
                cpu_energy_j = delta_uj * _UJ_TO_J
                cpu_power_w = cpu_energy_j / elapsed if elapsed > 0 else 0.0
                self._last_pkg_uj = now_uj
            except (OSError, PermissionError):
                self._rapl_pkg_ok = False

        # ── CPU utilisation ───────────────────────────────────────────
        cpu_pct: float = psutil.cpu_percent(interval=None)

        # ── RAM ───────────────────────────────────────────────────────
        cgroup = _cgroup_memory()
        if cgroup is not None:
            ram_used_gb, ram_used_pct = cgroup
        else:
            vm = psutil.virtual_memory()
            ram_used_gb = vm.used / _GB
            ram_used_pct = vm.percent

        # ── RAM energy (RAPL DRAM or estimate) ────────────────────────
        ram_energy_j: Optional[float] = None
        ram_power_w: Optional[float] = None
        if self._rapl_dram_ok:
            try:
                now_uj = _sum_rapl(self._rapl_dram_paths)
                delta_uj = max(0.0, now_uj - self._last_dram_uj)
                ram_energy_j = delta_uj * _UJ_TO_J
                ram_power_w = ram_energy_j / elapsed if elapsed > 0 else 0.0
                self._last_dram_uj = now_uj
            except (OSError, PermissionError):
                self._rapl_dram_ok = False

        if ram_power_w is None:
            # Estimation fallback: industry ~0.375 W per GiB in use
            ram_power_w = _estimate_ram_power(ram_used_gb)
            ram_energy_j = ram_power_w * elapsed if elapsed > 0 else None

        # ── Disk ──────────────────────────────────────────────────────
        disk_used_gb, disk_used_pct = _disk_usage(self._disk_path)

        # ── GPU ───────────────────────────────────────────────────────
        gpu_util: Optional[float] = None
        gpu_power_w: Optional[float] = None
        gpu_energy_j: Optional[float] = None
        gpu_mem_used_gb: Optional[float] = None
        gpu_mem_used_pct: Optional[float] = None
        gpu_mem_power_w: Optional[float] = None
        gpu_temp_c: Optional[float] = None

        if self._nvml is not None:
            try:
                nv = self._nvml
                h = self._nvml_handle
                util = nv.nvmlDeviceGetUtilizationRates(h)
                gpu_util = float(util.gpu)

                mw = nv.nvmlDeviceGetPowerUsage(h)
                gpu_power_w = mw / 1000.0
                gpu_energy_j = gpu_power_w * elapsed if elapsed > 0 else None

                mem = nv.nvmlDeviceGetMemoryInfo(h)
                gpu_mem_used_gb = mem.used / _GB
                gpu_mem_used_pct = mem.used / mem.total * 100.0 if mem.total else None

                # Memory power: NVML doesn't expose it directly;
                # use the memory-controller utilisation as a proxy fraction
                # of total board power when available, else None.
                try:
                    mem_util = float(util.memory)  # % memory controller busy
                    gpu_mem_power_w = round(gpu_power_w * mem_util / 100.0, 4)
                except Exception:
                    gpu_mem_power_w = None

                gpu_temp_c = float(
                    nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)
                )
            except Exception:
                pass  # partial read failure — leave whatever was set

        elif _IS_MACOS:
            apple = _apple_gpu_stats()
            if apple:
                gpu_util         = apple["usage_pct"]
                gpu_power_w      = apple["power_watts"]
                gpu_energy_j     = (
                    gpu_power_w * elapsed
                    if (gpu_power_w is not None and elapsed > 0)
                    else None
                )
                gpu_mem_used_gb  = apple["memory_used_gb"]
                gpu_mem_used_pct = apple["memory_used_pct"]
                gpu_mem_power_w  = apple["memory_power_watts"]
                gpu_temp_c       = apple["temperature_c"]

        # ── Total energy & emissions ──────────────────────────────────
        # Sum all measured/estimated energy sources for this interval.
        cpu_e = cpu_energy_j or 0.0
        ram_e = ram_energy_j or 0.0
        gpu_e = gpu_energy_j or 0.0
        total_energy_j = cpu_e + ram_e + gpu_e

        # ── Rotate state ─────────────────────────────────────────────
        self._last_ts = now_ts

        return {
            "elapsed_seconds":            elapsed,
            # CPU
            "cpu_utilization_pct":        cpu_pct,
            "cpu_energy_delta_joules":    cpu_energy_j,
            "cpu_power_watts":            cpu_power_w,
            # RAM
            "ram_used_gb":                ram_used_gb,
            "ram_used_pct":               ram_used_pct,
            "ram_energy_delta_joules":    ram_energy_j,
            "ram_power_watts":            ram_power_w,
            # Disk
            "disk_used_gb":               disk_used_gb,
            "disk_used_pct":              disk_used_pct,
            # Emissions & total energy
            "energy_consumed_delta_joules": total_energy_j,
            # GPU
            "gpu_utilization_pct":        gpu_util,
            "gpu_power_watts":            gpu_power_w,
            "gpu_energy_delta_joules":    gpu_energy_j,
            "gpu_memory_used_gb":         gpu_mem_used_gb,
            "gpu_memory_used_pct":        gpu_mem_used_pct,
            "gpu_memory_power_watts":     gpu_mem_power_w,
            "gpu_temperature_c":          gpu_temp_c,
            # Meta
            "hardware_access_mode":       self._hw_mode,
        }

    def close(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml = None
            self._nvml_handle = None

    def __enter__(self) -> "Tracker":
        return self

    def __exit__(self, *_) -> None:
        self.close()
