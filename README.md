# 🔍 rocprobe

**Lightweight ROCm GPU profiler CLI for MI300X**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![ROCm](https://img.shields.io/badge/ROCm-7.x-blue.svg)](https://rocm.docs.amd.com/)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)

`rocprobe` is a lightweight command-line GPU profiler built for AMD Instinct MI300X accelerators. It provides real-time kernel execution metrics, memory bandwidth utilization, and compute occupancy analysis — without the overhead of full profiling suites like rocprof.

## Why?

- **Fast**: Sub-second profiling with near-zero overhead
- **Focused**: Only the metrics that matter for MI300X optimization
- **Scriptable**: JSON output for CI/CD pipelines and automated tuning
- **ROCm-native**: Built directly on HIP/ROCm APIs, no abstraction layers

## Quick Start

```bash
# Install from source
cargo install --path .

# Profile a single kernel launch
rocprobe profile --target ./my_app --format table

# Stream metrics in real-time
rocprobe monitor --interval 500ms --gpu 0

# Export for analysis
rocprobe profile --target ./my_app --format json > report.json
```

## Architecture

```
┌─────────────────────────────────────────────┐
│              rocprobe CLI                    │
├──────────┬──────────┬───────────────────────┤
│ profile  │ monitor  │ analyze               │
├──────────┴──────────┴───────────────────────┤
│          ROCm Metrics Collector             │
│  ┌─────────┐ ┌──────────┐ ┌──────────────┐ │
│  │ HIP API │ │ rocprof  │ │ sysfs reader │ │
│  │ hooks   │ │ events   │ │ (power/temp) │ │
│  └─────────┘ └──────────┘ └──────────────┘ │
├─────────────────────────────────────────────┤
│         AMD GPU Driver (amdgpu)             │
│            MI300X VF (gfx942)               │
└─────────────────────────────────────────────┘
```

## Metrics Captured

| Metric | Description | Unit |
|--------|-------------|------|
| `kernel_time` | Kernel execution duration | μs |
| `mem_bw_util` | Memory bandwidth utilization | % |
| `compute_occ` | Compute unit occupancy | % |
| `l2_cache_hit` | L2 cache hit rate | % |
| `vgpr_usage` | VGPR allocation per CU | count |
| `wavefront_active` | Active wavefronts per SIMD | count |
| `power_draw` | GPU power consumption | W |
| `temp_edge` | Edge temperature | °C |

## Example Output

```
╔═══════════════════════════════════════════════════════════╗
║                    rocprobe v0.3.1                        ║
║          GPU 0: AMD Instinct MI300X VF (gfx942)          ║
╠═══════════════════╦═══════════╦═══════════╦═══════════════╣
║ Kernel            ║ Time (μs) ║ Mem BW %  ║ Occupancy %   ║
╠═══════════════════╬═══════════╬═══════════╬═══════════════╣
║ gemm_16384x16384  ║    1,247  ║     87.3  ║         94.1  ║
║ softmax_fwd       ║       89  ║     62.1  ║         78.5  ║
║ reduce_sum        ║       12  ║     45.8  ║         56.2  ║
║ conv2d_nhwc       ║      341  ║     91.7  ║         88.9  ║
╚═══════════════════╩═══════════╩═══════════╩═══════════════╝

Summary: 4 kernels | 1,689 μs total | 71.7% avg bandwidth util
```

## Python Bindings

```python
from rocprobe import Profiler

p = Profiler(gpu_id=0)
report = p.profile("./my_app")

for kernel in report.kernels:
    print(f"{kernel.name}: {kernel.time_us}μs, {kernel.mem_bw_util}% bw")

print(f"Total: {report.total_time_us}μs")
```

## Building

```bash
# Prerequisites: ROCm 7.x, Rust 1.75+
git clone https://github.com/huang-poi/rocprobe.git
cd rocprobe
cargo build --release

# Run tests (requires MI300X or compatible GPU)
cargo test --features gpu-tests
```

## Roadmap

- [x] Basic kernel profiling (HIP API)
- [x] Real-time monitoring mode
- [x] JSON export
- [ ] rocprofiler-SDK integration (ROCm 7.2+)
- [ ] Multi-GPU support
- [ ] Flame graph generation
- [ ] CI/CD regression detection

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT — see [LICENSE](LICENSE).
