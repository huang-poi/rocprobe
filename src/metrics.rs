use anyhow::Result;
use std::time::{Duration, Instant};
use std::thread;
use crate::profiler;

pub struct MonitorConfig {
    pub interval_ms: u64,
    pub duration_secs: u64,
    pub gpu_id: u32,
}

/// Real-time GPU metrics snapshot
#[derive(Debug)]
struct MetricSnapshot {
    timestamp_ms: u128,
    power_w: f64,
    temp_c: f64,
    gpu_busy_pct: f64,
    mem_used_gb: f64,
    mem_total_gb: f64,
}

/// Read GPU busy% from rocm-smi or sysfs
fn read_gpu_busy(gpu_id: u32) -> f64 {
    let path = format!(
        "/sys/class/drm/card{}/device/gpu_busy_percent",
        gpu_id
    );
    std::fs::read_to_string(&path)
        .unwrap_or_default()
        .trim()
        .parse::<f64>()
        .unwrap_or(0.0)
}

/// Read VRAM usage from sysfs
fn read_vram_usage(gpu_id: u32) -> (f64, f64) {
    let base = format!("/sys/class/drm/card{}/device", gpu_id);

    let used = std::fs::read_to_string(format!("{}/mem_info_vram_used", base))
        .unwrap_or_default()
        .trim()
        .parse::<f64>()
        .unwrap_or(0.0) / 1_073_741_824.0;

    let total = std::fs::read_to_string(format!("{}/mem_info_vram_total", base))
        .unwrap_or_default()
        .trim()
        .parse::<f64>()
        .unwrap_or(0.0) / 1_073_741_824.0;

    (used, total)
}

pub fn run_monitor(config: &MonitorConfig) -> Result<()> {
    let start = Instant::now();
    let interval = Duration::from_millis(config.interval_ms);
    let max_duration = if config.duration_secs > 0 {
        Some(Duration::from_secs(config.duration_secs))
    } else {
        None
    };

    eprintln!("rocprobe: monitoring GPU {} (Ctrl+C to stop)", config.gpu_id);
    eprintln!("{:<12} {:<10} {:<8} {:<10} {:<14} {:<10}",
              "TIME", "POWER(W)", "TEMP(C)", "GPU(%)", "VRAM(GB)", "BW_UTIL(%)");
    eprintln!("{}", "-".repeat(70));

    loop {
        if let Some(max) = max_duration {
            if start.elapsed() > max {
                break;
            }
        }

        let (power, temp) = profiler::read_gpu_metrics(config.gpu_id)?;
        let gpu_busy = read_gpu_busy(config.gpu_id);
        let (vram_used, vram_total) = read_vram_usage(config.gpu_id);

        let elapsed_ms = start.elapsed().as_millis();
        let ts = format!("{:.1}s", elapsed_ms as f64 / 1000.0);

        println!("{:<12} {:<10.1} {:<8.1} {:<10.1} {:<6.1}/{:<6.1} {:<10.1}",
                 ts, power, temp, gpu_busy, vram_used, vram_total, 0.0);

        thread::sleep(interval);
    }

    Ok(())
}
