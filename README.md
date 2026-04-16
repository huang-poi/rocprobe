# 🔍 ROCProbe

> A lightweight, high-performance GPU profiler CLI for AMD ROCm on MI300X accelerators

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![ROCm](https://img.shields.io/badge/ROCm-6.0+-red.svg)](https://rocm.docs.amd.com/)
[![MI300X](https://img.shields.io/badge/MI300X-Supported-green.svg)](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
[![CI](https://img.shields.io/github/actions/workflow/status/huang-poi/rocprobe/ci.yml?branch=main)](https://github.com/huang-poi/rocprobe/actions)
[![Crates.io](https://img.shields.io/crates/v/rocprobe.svg)](https://crates.io/crates/rocprobe)

---

**ROCProbe** is a zero-overhead command-line profiler for AMD Instinct MI300X GPUs, built in Rust for speed and safety. It hooks into the ROCm runtime to capture kernel execution times, memory bandwidth utilization, occupancy metrics, and wavefront activity — all from your terminal.

```
┌─────────────────────────────────────────────────────────┐
│                      ROCProbe                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐   │
│  │ CLI      │──▶│ Profiler │──▶│ Metric Collector │   │
│  │ (clap)   │   │ Core     │   │ (rocprofiler)    │   │
│  └──────────┘   └──────────┘   └──────────────────┘   │
│       │              │                  │                │
│       ▼              ▼                  ▼                │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐   │
│  │ Output   │◀──│ HSA      │◀──│ HIP Runtime      │   │
│  │ Formatter│   │ Runtime  │   │ Interceptor      │   │
│  └──────────┘   └──────────┘   └──────────────────┘   │
│                       │                                 │
│                       ▼                                 │
│              ┌────────────────┐                        │
│              │ MI300X GPU     │                        │
│              │ (gfx942)       │                        │
│              └────────────────┘                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## ✨ Features

- **Kernel Profiling** — Per-kernel execution time, occupancy, and wavefront counts
- **Memory Analysis** — HBM bandwidth utilization, copy engine throughput, memory pressure detection
- **Occupancy Metrics** — Wavefront occupancy, SIMD utilization, register pressure analysis
- **Real-time Streaming** — Live profiling output as kernels execute
- **Python Bindings** — Import `rocprobe` in Python for integration into ML training loops
- **JSON/CSV Export** — Machine-readable output for CI/CD pipelines and dashboards
- **Low Overhead** — <1% performance impact on profiled workloads

## 📦 Installation

### From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/huang-poi/rocprobe.git
cd rocprobe

# Build with release optimizations
cargo build --release

# Install to PATH
cargo install --path .
```

### Pre-built Binary

```bash
# Download the latest release
curl -sSL https://github.com/huang-poi/rocprobe/releases/latest/download/rocprobe-linux-x64 -o /usr/local/bin/rocprobe
chmod +x /usr/local/bin/rocprobe
```

### Python Package

```bash
pip install rocprobe
```

### Prerequisites

- **ROCm 6.0+** installed and configured
- **Rust 1.75+** toolchain (for building from source)
- **AMD Instinct MI300X** or compatible accelerator (gfx90a/gfx942)

## 🚀 Usage

### Basic Profiling

```bash
# Profile a HIP application
rocprofile ./my_hip_app

# Profile with specific metrics
rocprofile --metrics kernel_time,occupancy,bandwidth ./my_hip_app

# Stream real-time metrics
rocprofile --stream --interval 100 ./my_hip_app
```

### Example Output

```
╔══════════════════════════════════════════════════════════════════╗
║                    ROCProbe v0.3.1 — Profiling Results          ║
╠══════════════════════════════════════════════════════════════════╣
║ Device: AMD Instinct MI300X | GFX: gfx942 | CU: 304            ║
║ Session: 12.847s | Kernels: 1,247 | Total Time: 11.203s         ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  KERNEL EXECUTION TIMELINE                                       ║
║  ─────────────────────────────────────────────────────────────── ║
║  matmul_forward    ████████████████████████░░░░  72.3%  8.102s  ║
║  layernorm_fwd     ████████░░░░░░░░░░░░░░░░░░  14.2%  1.591s  ║
║  attention_fwd     ████░░░░░░░░░░░░░░░░░░░░░░   8.1%  0.907s  ║
║  residual_add      █░░░░░░░░░░░░░░░░░░░░░░░░░   3.2%  0.358s  ║
║  other (98)        █░░░░░░░░░░░░░░░░░░░░░░░░░   2.2%  0.245s  ║
║                                                                  ║
║  MEMORY BANDWIDTH                                                ║
║  ─────────────────────────────────────────────────────────────── ║
║  HBM Read:     2,847.3 GB/s  (92.1% of peak)                   ║
║  HBM Write:    1,203.8 GB/s  (78.4% of peak)                   ║
║  Total:        4,051.1 GB/s                                      ║
║                                                                  ║
║  OCCUPANCY                                                       ║
║  ─────────────────────────────────────────────────────────────── ║
║  Active Waves: 1,824 / 2,432  (75.0%)                          ║
║  Avg Occupancy: 68.4%                                            ║
║  Register Pressure: Moderate (48 avg VGPRs)                      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

### Python Integration

```python
from rocprobe import Profiler

# Profile a function
profiler = Profiler(device=0)

@profiler.profile
def train_step(batch):
    logits = model(batch)
    loss = loss_fn(logits, targets)
    loss.backward()
    return loss

# Run profiling
result = train_step(batch)
print(result.summary())
# => "train_step: 12.4ms avg, 84% occupancy, 2.1TB/s HBM bandwidth"
```

### JSON Export

```bash
# Export to JSON for CI/CD
rocprofile --output results.json --format json ./my_app

# Export to CSV
rocprofile --output results.csv --format csv ./my_app
```

## 🏗️ Architecture

ROCProbe uses a layered architecture for minimal overhead and maximum flexibility:

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Space                              │
├─────────────┬───────────────────┬───────────────────────────────┤
│  rocprofile │  rocprobe (lib)   │  Python bindings              │
│  (CLI)      │                   │  (PyO3)                       │
├─────────────┴───────────────────┴───────────────────────────────┤
│                     Profiler Core (Rust)                        │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────────────────┐  │
│  │ Metric       │ │ Session      │ │ Output                  │  │
│  │ Collector    │ │ Manager      │ │ Formatter               │  │
│  └──────┬───────┘ └──────┬───────┘ └────────────┬───────────┘  │
├─────────┼────────────────┼──────────────────────┼───────────────┤
│                     ROCm Runtime Layer                          │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────────────────┐  │
│  │ rocprofiler   │ │ HIP Runtime  │ │ HSA Runtime            │  │
│  │ (counters)    │ │ (intercept)  │ │ (memory/disp)          │  │
│  └──────┬───────┘ └──────┬───────┘ └────────────┬───────────┘  │
├─────────┴────────────────┴──────────────────────┴───────────────┤
│                      Hardware Layer                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  AMD Instinct MI300X (gfx942)                           │   │
│  │  • 304 Compute Units  • 192 GB HBM3                     │   │
│  │  • 5.3 TB/s Bandwidth • 1,530 TFLOPS FP16              │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Supported Metrics

| Category | Metrics | Source |
|----------|---------|--------|
| **Kernel** | Execution time, launch params, grid/block dims | HIP Runtime |
| **Occupancy** | Wavefront count, SIMD utilization, occupancy % | rocprofiler |
| **Memory** | HBM read/write BW, L2 cache hit rate, DRAM utilization | HSA/counters |
| **Compute** | FP16/FP32/FP64 TFLOPS, INT8 TOPS, FMA utilization | PM4/counters |
| **Power** | GPU power, junction temp, memory temp | SMI integration |

## 🗺️ Roadmap

- [x] v0.1.0 — Basic kernel profiling and CLI
- [x] v0.2.0 — Memory bandwidth metrics
- [x] v0.3.0 — Python bindings and JSON export
- [ ] v0.4.0 — Multi-GPU profiling support
- [ ] v0.5.0 — Web dashboard for visualization
- [ ] v0.6.0 — MI300A (APU) support

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [AMD ROCm](https://rocm.docs.amd.com/) — Runtime and profiling infrastructure
- [rocprofiler](https://github.com/ROCm/rocprofiler) — Hardware counter access
- [HSA-Runtime](https://github.com/ROCm/hsa-runtime) — HSA agent management
- Built with ❤️ for the MI300X community

---

<p align="center">
  <i>Built for AMD Instinct MI300X • Powered by Rust • Open Source</i>
</p>
