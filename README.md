# rocprobe

A lightweight GPU profiler for AMD ROCm platforms. Designed for fast kernel-level profiling without full ROCm Profiler overhead.

## Features
- Per-kernel timing, occupancy, memory bandwidth
- CLI-first: pipe to jq, csvkit, or any tool
- MI300X (CDNA3) optimized
- JSON, CSV, table output formats

## Quick Start
```bash
cargo install rocprobe
rocprobe profile --app ./my_hip_app --format table
rocprobe status --device 0 --format json
rocprobe occupancy --trace kernels.json
rocprobe memband --device 0 --interval 200
```

## Related Projects
- [hip-graph-capture](https://github.com/huang-poi/hip-graph-capture)
- [rocblas-lite](https://github.com/huang-poi/rocblas-lite)
- [mi300x-bench](https://github.com/huang-poi/mi300x-bench)
- [hip-kernel-lab](https://github.com/huang-poi/hip-kernel-lab)
- [rocm-devbox](https://github.com/huang-poi/rocm-devbox)

## License
MIT

// docs: update README with v0.2.0 features
