//! Basic profiling example
//!
//! Run with: cargo run --example basic

use std::process::Command;

fn main() {
    println!("rocprobe basic example");
    println!("=====================");
    println!();

    // Check ROCm availability
    let rocm_version = std::fs::read_to_string("/opt/rocm/.info/version")
        .unwrap_or_else(|_| "not found".to_string());
    println!("ROCm version: {}", rocm_version.trim());

    // Check GPU
    let output = Command::new("rocm-smi")
        .args(["--showproductname"])
        .output()
        .expect("rocm-smi not found");

    println!("GPU info:");
    println!("{}", String::from_utf8_lossy(&output.stdout));
}
