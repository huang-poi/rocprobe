# Contributing to ROCProbe

Thank you for your interest in contributing to ROCProbe! This document provides guidelines for contributing to this project.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/huang-poi/rocprobe.git
cd rocprobe

# Build
cargo build

# Run tests
cargo test

# Run with verbose logging
RUST_LOG=debug cargo run -- devices
```

## Code Style

- Follow standard Rust conventions (rustfmt + clippy)
- Use descriptive variable names
- Add doc comments for public APIs
- Keep functions focused and small

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Reporting Issues

- Use the GitHub issue tracker
- Include ROCm version, GPU model, and OS
- Provide minimal reproduction steps

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
