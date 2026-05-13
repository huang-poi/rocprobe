use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::process::Command;

#[derive(Debug, Serialize, Deserialize)]
pub struct GpuStatus {
    pub device_id: usize,
    pub name: String,
    pub temperature_c: u32,
    pub gpu_utilization_pct: u32,
    pub memory_used_mb: u64,
    pub memory_total_mb: u64,
    pub power_draw_w: f64,
}

pub fn query_status(device: usize, format: &str) -> Result<()> {
    let output = Command::new("rocm-smi")
        .args(["--showtemp", "--showuse", "--showmeminfo", "vram"])
        .output();
    match output {
        Ok(out) if out.status.success() => println!("{}", String::from_utf8_lossy(&out.stdout)),
        _ => println!("rocm-smi not available (no GPU in this environment)"),
    }
    Ok(())
}

pub fn analyze_occupancy(trace_path: &str) -> Result<()> {
    let data = std::fs::read_to_string(trace_path)?;
    let traces: Vec<super::profiler::KernelTrace> = serde_json::from_str(&data)?;
    println!("\n=== Occupancy Analysis ===");
    for (i, t) in traces.iter().enumerate() {
        println!(
            "Kernel {}: {} — occupancy {:.1}%",
            i,
            t.kernel_name,
            t.occupancy * 100.0
        );
    }
    Ok(())
}

pub fn memory_bandwidth(device: usize, interval: u64, format: &str) -> Result<()> {
    println!(
        "Monitoring memory bandwidth on GPU {} (interval: {}ms)",
        device, interval
    );
    for i in 0..10 {
        let output = Command::new("rocm-smi")
            .args(["--showmeminfo", "vram"])
            .output()?;
        println!(
            "[Sample {}] {}",
            i,
            String::from_utf8_lossy(&output.stdout)
                .lines()
                .collect::<Vec<_>>()
                .join(" | ")
        );
        std::thread::sleep(std::time::Duration::from_millis(interval));
    }
    Ok(())
}

// fix(metrics): handle rocm-smi not found gracefully

// feat(metrics): add PCIe bandwidth monitoring
