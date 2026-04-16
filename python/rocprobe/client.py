"""
ROCProbe Client — Python interface for GPU profiling.

This module provides a high-level Python API for profiling ROCm GPU
applications. It communicates with the rocprobe CLI binary or uses
PyO3 bindings when available.

Architecture:
    ┌─────────────────────────────────────────────┐
    │              Python Application              │
    ├─────────────────────────────────────────────┤
    │           rocprobe.Profiler (this module)    │
    ├─────────────────────────────────────────────┤
    │    PyO3 Bindings  │   CLI Subprocess Mode    │
    ├───────────────────┴─────────────────────────┤
    │              ROCm Runtime (HIP/HSA)          │
    └─────────────────────────────────────────────┘
"""

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class KernelResult:
    """Profiling result for a single kernel execution."""

    name: str
    grid_dims: Tuple[int, int, int]
    block_dims: Tuple[int, int, int]
    execution_time_us: int
    occupancy_pct: float
    vgpr_count: int
    sgpr_count: int
    l2_hit_rate: float
    ipc: float
    dispatch_count: int = 1

    @property
    def execution_time_ms(self) -> float:
        """Kernel execution time in milliseconds."""
        return self.execution_time_us / 1_000.0

    @property
    def total_time_ms(self) -> float:
        """Total time including all dispatches."""
        return self.execution_time_ms * self.dispatch_count


@dataclass
class MemoryResult:
    """Memory subsystem profiling results."""

    hbm_read_gbs: float
    hbm_write_gbs: float
    hbm_peak_gbs: float
    l2_hit_rate: float

    @property
    def total_bandwidth_gbs(self) -> float:
        return self.hbm_read_gbs + self.hbm_write_gbs

    @property
    def hbm_utilization(self) -> float:
        return self.total_bandwidth_gbs / self.hbm_peak_gbs


@dataclass
class ProfileResult:
    """Complete profiling result for a session."""

    kernels: List[KernelResult]
    memory: MemoryResult
    total_duration_ms: float
    device_name: str
    gfx_target: str
    compute_units: int

    def summary(self) -> str:
        """Return a human-readable summary string."""
        top_kernels = sorted(self.kernels, key=lambda k: k.total_time_ms, reverse=True)
        top = top_kernels[0] if top_kernels else None

        parts = []
        parts.append(f"Profiled {len(self.kernels)} kernels in {self.total_duration_ms:.1f}ms")
        parts.append(f"Device: {self.device_name} ({self.gfx_target})")

        if top:
            parts.append(
                f"Top kernel: {top.name} ({top.execution_time_ms:.1f}ms, "
                f"{top.occupancy_pct:.0f}% occupancy)"
            )

        parts.append(
            f"HBM bandwidth: {self.memory.total_bandwidth_gbs:.1f} GB/s "
            f"({self.memory.hbm_utilization:.0%} of peak)"
        )

        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a dictionary for serialization."""
        return {
            "kernels": [
                {
                    "name": k.name,
                    "grid": list(k.grid_dims),
                    "block": list(k.block_dims),
                    "time_us": k.execution_time_us,
                    "occupancy": k.occupancy_pct,
                    "vgpr": k.vgpr_count,
                    "sgpr": k.sgpr_count,
                    "ipc": k.ipc,
                    "dispatches": k.dispatch_count,
                }
                for k in self.kernels
            ],
            "memory": {
                "hbm_read_gbs": self.memory.hbm_read_gbs,
                "hbm_write_gbs": self.memory.hbm_write_gbs,
                "l2_hit_rate": self.memory.l2_hit_rate,
            },
            "device": self.device_name,
            "gfx": self.gfx_target,
            "total_duration_ms": self.total_duration_ms,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize results to JSON."""
        return json.dumps(self.to_dict(), indent=indent)


class Profiler:
    """
    ROCProbe GPU profiler for ROCm applications.

    Usage:
        profiler = Profiler(device=0)

        # Profile a function
        @profiler.profile
        def my_kernel(x):
            ...

        # Or use as context manager
        with profiler.session() as sess:
            for batch in dataloader:
                train_step(batch)
            result = sess.results()

    Args:
        device: ROCm device index (default: 0)
        metrics: List of metric names to collect
        stream: Enable real-time streaming output
        interval_ms: Streaming interval in milliseconds
    """

    def __init__(
        self,
        device: int = 0,
        metrics: Optional[List[str]] = None,
        stream: bool = False,
        interval_ms: int = 100,
    ):
        self.device = device
        self.metrics = metrics or [
            "kernel_time",
            "occupancy",
            "bandwidth",
            "ipc",
        ]
        self.stream = stream
        self.interval_ms = interval_ms
        self._bin_path = self._find_rocprobe_bin()
        self._use_native = self._check_native_bindings()

    def _find_rocprobe_bin(self) -> Optional[str]:
        """Locate the rocprobe CLI binary."""
        # Check common locations
        candidates = [
            Path("/usr/local/bin/rocprobe"),
            Path.home() / ".cargo" / "bin" / "rocprobe",
            Path(os.environ.get("ROCprobe_BIN", "")),
        ]

        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)

        # Try PATH
        import shutil
        return shutil.which("rocprobe")

    def _check_native_bindings(self) -> bool:
        """Check if PyO3 native bindings are available."""
        try:
            import _rocprobe  # noqa: F401
            return True
        except ImportError:
            return False

    def profile(self, func: Callable) -> Callable:
        """
        Decorator to profile a function's GPU execution.

        Usage:
            @profiler.profile
            def my_function(x):
                return model(x)
        """
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = self._profile_call(func, *args, **kwargs)
            return result

        wrapper._rocprobe_profiled = True
        return wrapper

    def _profile_call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function under profiling."""
        if self._use_native:
            return self._profile_native(func, *args, **kwargs)
        elif self._bin_path:
            return self._profile_cli(func, *args, **kwargs)
        else:
            # Fallback: use HIP events for basic timing
            return self._profile_hip_events(func, *args, **kwargs)

    def _profile_native(self, func: Callable, *args, **kwargs) -> Any:
        """Profile using native PyO3 bindings."""
        import _rocprobe

        session = _rocprobe.ProfilerSession(self.device)
        session.start()

        result = func(*args, **kwargs)

        session.stop()
        self._last_result = session.get_results()

        return result

    def _profile_cli(self, func: Callable, *args, **kwargs) -> Any:
        """Profile by wrapping the call in a rocprobe CLI session."""
        # For CLI mode, we just time the function call locally
        start = time.perf_counter_ns()
        result = func(*args, **kwargs)
        end = time.perf_counter_ns()

        elapsed_us = (end - start) / 1_000

        # Store a minimal result
        self._last_result = ProfileResult(
            kernels=[
                KernelResult(
                    name=func.__name__,
                    grid_dims=(1, 1, 1),
                    block_dims=(256, 1, 1),
                    execution_time_us=int(elapsed_us),
                    occupancy_pct=0.0,
                    vgpr_count=0,
                    sgpr_count=0,
                    l2_hit_rate=0.0,
                    ipc=0.0,
                )
            ],
            memory=MemoryResult(
                hbm_read_gbs=0.0,
                hbm_write_gbs=0.0,
                hbm_peak_gbs=5300.0,
                l2_hit_rate=0.0,
            ),
            total_duration_ms=elapsed_us / 1_000.0,
            device_name="Unknown",
            gfx_target="unknown",
            compute_units=0,
        )

        return result

    def _profile_hip_events(self, func: Callable, *args, **kwargs) -> Any:
        """Fallback: use HIP events for basic kernel timing."""
        try:
            import torch

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            result = func(*args, **kwargs)
            end_event.record()

            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)

            self._last_result = ProfileResult(
                kernels=[
                    KernelResult(
                        name=func.__name__,
                        grid_dims=(0, 0, 0),
                        block_dims=(0, 0, 0),
                        execution_time_us=int(elapsed_ms * 1_000),
                        occupancy_pct=0.0,
                        vgpr_count=0,
                        sgpr_count=0,
                        l2_hit_rate=0.0,
                        ipc=0.0,
                    )
                ],
                memory=MemoryResult(
                    hbm_read_gbs=0.0,
                    hbm_write_gbs=0.0,
                    hbm_peak_gbs=5300.0,
                    l2_hit_rate=0.0,
                ),
                total_duration_ms=elapsed_ms,
                device_name="HIP Device",
                gfx_target="unknown",
                compute_units=0,
            )

            return result

        except ImportError:
            raise RuntimeError(
                "ROCProbe requires either native bindings, CLI binary, or PyTorch with ROCm"
            )

    def session(self) -> "Session":
        """Create a profiling session context manager."""
        return Session(self)

    def last_result(self) -> Optional[ProfileResult]:
        """Get the result of the last profiled call."""
        return getattr(self, "_last_result", None)

    @staticmethod
    def list_devices() -> List[Dict[str, Any]]:
        """
        List available ROCm GPU devices.

        Returns:
            List of device info dictionaries.
        """
        devices = []

        try:
            output = subprocess.check_output(
                ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            data = json.loads(output)
            # Parse rocm-smi JSON output
            for card_id, card_info in data.items():
                if card_id.startswith("card"):
                    devices.append({
                        "id": int(card_id.replace("card", "")),
                        "name": card_info.get("Card Series", "Unknown"),
                        "memory_mb": int(card_info.get("VRAM Total Memory (B)", 0)) // (1024 * 1024),
                    })
        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
            # Fallback: try HIP runtime detection
            try:
                output = subprocess.check_output(
                    ["rocminfo"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                # Parse rocminfo output
                current_device = None
                for line in output.splitlines():
                    if "Marketing Name" in line:
                        name = line.split(":")[-1].strip()
                        if current_device is not None:
                            current_device["name"] = name
                    elif "Device Type" in line and "GPU" in line:
                        current_device = {"id": len(devices), "name": "Unknown", "memory_mb": 0}
                    elif current_device is not None and "Device Type" in line:
                        devices.append(current_device)
                        current_device = None
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

        return devices


class Session:
    """
    Context manager for profiling multiple operations.

    Usage:
        with profiler.session() as sess:
            for batch in dataloader:
                train_step(batch)
            result = sess.results()
    """

    def __init__(self, profiler: Profiler):
        self.profiler = profiler
        self._start_time: Optional[float] = None

    def __enter__(self) -> "Session":
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def results(self) -> ProfileResult:
        """Get aggregated profiling results for the session."""
        result = self.profiler.last_result()
        if result is None:
            raise RuntimeError("No profiling data collected. Execute GPU operations within the session.")
        return result
