"""Python client for rocprobe CLI."""
import json
import subprocess
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class KernelMetric:
    name: str
    time_us: float
    mem_bw_util: float
    compute_occ: float
    l2_cache_hit: float
    vgpr_usage: int
    wavefront_active: int


@dataclass
class MemTransfer:
    direction: str
    size_bytes: int
    time_us: float
    bandwidth_gbs: float


@dataclass
class Report:
    gpu_name: str
    gfx_arch: str
    rocm_version: str
    kernels: List[KernelMetric]
    mem_transfers: List[MemTransfer]
    total_time_us: float
    avg_bw_util: float
    avg_occupancy: float
    power_draw_w: float
    temp_edge_c: float


class Profiler:
    """Python wrapper for rocprobe CLI."""

    def __init__(self, gpu_id: int = 0, binary: str = "rocprobe"):
        self.gpu_id = gpu_id
        self.binary = binary

    def profile(self, target: str, capture_mem: bool = False) -> Report:
        """Profile a target application."""
        cmd = [
            self.binary, "profile",
            "--target", target,
            "--gpu", str(self.gpu_id),
            "--format", "json",
        ]
        if capture_mem:
            cmd.append("--mem-transfers")

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        kernels = [
            KernelMetric(**k) for k in data.get("kernels", [])
        ]
        transfers = [
            MemTransfer(**t) for t in data.get("mem_transfers", [])
        ]

        return Report(
            gpu_name=data["gpu_name"],
            gfx_arch=data["gfx_arch"],
            rocm_version=data["rocm_version"],
            kernels=kernels,
            mem_transfers=transfers,
            total_time_us=data["total_time_us"],
            avg_bw_util=data["avg_bw_util"],
            avg_occupancy=data["avg_occupancy"],
            power_draw_w=data["power_draw_w"],
            temp_edge_c=data["temp_edge_c"],
        )

    def monitor(self, interval_ms: int = 1000, duration_secs: int = 0):
        """Stream GPU metrics (returns iterator)."""
        cmd = [
            self.binary, "monitor",
            "--interval", str(interval_ms),
            "--duration", str(duration_secs),
            "--gpu", str(self.gpu_id),
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
        for line in proc.stdout:
            yield line.strip()
