# Contributing to rocprobe

Thanks for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/huang-poi/rocprobe.git
cd rocprobe
cargo build
cargo test
```

## Running Tests

Unit tests (no GPU required):
```bash
cargo test
```

Integration tests (requires MI300X):
```bash
cargo test --features gpu-tests
```

## Code Style

- Rust: `cargo fmt` + `cargo clippy`
- Python: `black` + `ruff`

## Pull Requests

1. Fork the repo
2. Create a feature branch
3. Add tests for new functionality
4. Ensure CI passes
5. Submit PR with clear description

## Reporting Issues

Please include:
- ROCm version (`cat /opt/rocm/.info/version`)
- GPU model (`rocm-smi --showproductname`)
- Minimal reproduction steps
