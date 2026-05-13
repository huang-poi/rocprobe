use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::process::Command;
use std::time::{Duration, Instant};

#[derive(Debug, Serialize, Deserialize)]
pub struct KernelTrace {
    pub kernel_name: String,
    pub grid_size: [u64; 3],
    pub block_size: [u64; 3],
    pub registers_per_thread: u32,
    pub shared_memory_bytes: u64,
    pub duration_us: f64,
    pub occupancy: f64,
    pub memory_read_bytes: u64,
    pub memory_write_bytes: u64,
    pub gpu_id: u32,
    pub stream_id: u32,
    pub timestamp_ns: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProfileResult {
    pub traces: Vec<KernelTrace>,
    pub total_duration_s: f64,
    pub avg_kernel_time_us: f64,
    pub peak_memory_bandwidth_gbs: f64,
    pub gpu_utilization_pct: f64,
}

pub fn run(app: &str, format: &str, duration: u64) -> Result<()> {
    println!("Profiling {} for {}s...", app, duration);
    std::env::set_var("HIP_PROFILE_API", "1");
    let start = Instant::now();
    let output = Command::new(app)
        .env("HIP_PROFILE_API", "1")
        .output()
        .context("Failed to launch")?;
    let elapsed = start.elapsed();
    println!("Finished in {:.2}s", elapsed.as_secs_f64());
    Ok(())
}

// refactor(profiler): extract kernel parsing

// refactor(profiler): add async profiling with tokio
