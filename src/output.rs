//! Output Formatting — Terminal tables, JSON, and CSV export
//!
//! Handles all presentation logic for profiling results. Supports three
//! output formats:
//!
//! - **Table**: Rich terminal output with Unicode box-drawing characters
//! - **JSON**: Machine-readable JSON for CI/CD pipelines
//! - **CSV**: Spreadsheet-compatible tabular export
//!
//! # Color Coding
//!
//! Table output uses ANSI colors to indicate performance characteristics:
//! - 🟢 Green: Good (>80% utilization / low pressure)
//! - 🟡 Yellow: Moderate (50-80% utilization)
//! - 🔴 Red: Poor (<50% utilization / high pressure)

use anyhow::Result;
use colored::*;
use serde::Serialize;
use std::time::Duration;

use crate::metrics::{KernelMetrics, MemoryMetrics, OccupancyMetrics, ProfilingResults};

/// Print profiling results as a formatted terminal table
pub fn print_table(results: &ProfilingResults) {
    let info = &results.session_info;

    // Header
    println!();
    println!("{}", "╔══════════════════════════════════════════════════════════════════╗".cyan());
    println!("{}", format!(
        "║  {:<62} ║",
        format!("ROCProbe v0.3.1 — Profiling Results")
    ).cyan());
    println!("{}", "╠══════════════════════════════════════════════════════════════════╣".cyan());
    println!("{}", format!(
        "║  {:<62} ║",
        format!(
            "Device: {} | GFX: {} | CU: {} | HBM: {} GB",
            info.device_name, info.gfx_target, info.compute_units, info.memory_gb
        )
    ).cyan());
    println!("{}", format!(
        "║  {:<62} ║",
        format!(
            "Session: {:.3}s | Kernels: {} | ROCm: {}",
            info.total_duration.as_secs_f64(),
            info.total_kernels,
            info.rocm_version,
        )
    ).cyan());
    println!("{}", "╠══════════════════════════════════════════════════════════════════╣".cyan());

    // Kernel Timeline
    print_kernel_timeline(&results.kernels, info.total_duration);

    // Memory Bandwidth
    print_memory_bandwidth(&results.memory);

    // Occupancy
    print_occupancy(&results.occupancy);

    // Per-kernel details table
    print_kernel_details(&results.kernels);

    // Footer
    println!("{}", "╚══════════════════════════════════════════════════════════════════╝".cyan());
    println!();
}

/// Print kernel execution timeline as a bar chart
fn print_kernel_timeline(kernels: &[KernelMetrics], total_duration: Duration) {
    let total_us: u64 = kernels.iter().map(|k| k.execution_time_us * k.dispatch_count as u64).sum();
    let total_secs = total_duration.as_secs_f64();

    println!("{}", format!("║  {:<62} ║", "KERNEL EXECUTION TIMELINE").bold());
    println!("{}", format!("║  {:<62} ║", "─".repeat(62)).dimmed());

    for kernel in kernels {
        let kernel_total = kernel.execution_time_us * kernel.dispatch_count as u64;
        let pct = (kernel_total as f64 / total_us as f64) * 100.0;
        let bar_len = (pct / 2.5) as usize;  // Scale to ~40 chars
        let bar: String = "█".repeat(bar_len);
        let pad: String = "░".repeat(40 - bar_len);
        let time_str = format_duration_us(kernel_total);

        let color = if pct > 50.0 {
            "red"
        } else if pct > 20.0 {
            "yellow"
        } else {
            "green"
        };

        let bar_colored = match color {
            "red" => format!("{}{}", bar.red(), pad.dimmed()),
            "yellow" => format!("{}{}", bar.yellow(), pad.dimmed()),
            _ => format!("{}{}", bar.green(), pad.dimmed()),
        };

        let line = format!(
            "  {:<20} {} {:>5.1}%  {}",
            kernel.name, bar_colored, pct, time_str
        );

        println!("{}", format!("║  {:<62} ║", line).cyan());
    }

    println!("{}", format!("║  {:<62} ║", "").cyan());
}

/// Print memory bandwidth section
fn print_memory_bandwidth(memory: &MemoryMetrics) {
    let total_bw = memory.hbm_read_gbs + memory.hbm_write_gbs;

    println!("{}", format!("║  {:<62} ║", "MEMORY BANDWIDTH").bold());
    println!("{}", format!("║  {:<62} ║", "─".repeat(62)).dimmed());

    let read_util = (memory.hbm_read_gbs / memory.hbm_peak_gbs) * 100.0;
    let write_util = (memory.hbm_write_gbs / memory.hbm_peak_gbs) * 100.0;

    let read_color = util_color(read_util);
    let write_color = util_color(write_util);

    println!("{}", format!("║  {:<62} ║", format!(
        "  HBM Read:   {:>8.1} GB/s  ({})",
        memory.hbm_read_gbs,
        format!("{:.1}% of peak", read_util).color(read_color)
    )).cyan());

    println!("{}", format!("║  {:<62} ║", format!(
        "  HBM Write:  {:>8.1} GB/s  ({})",
        memory.hbm_write_gbs,
        format!("{:.1}% of peak", write_util).color(write_color)
    )).cyan());

    println!("{}", format!("║  {:<62} ║", format!(
        "  Total:      {:>8.1} GB/s",
        total_bw
    )).cyan());

    println!("{}", format!("║  {:<62} ║", format!(
        "  L2 Hit Rate: {:.1}%",
        memory.l2_hit_rate * 100.0
    )).cyan());

    println!("{}", format!("║  {:<62} ║", "").cyan());
}

/// Print occupancy metrics section
fn print_occupancy(occupancy: &OccupancyMetrics) {
    println!("{}", format!("║  {:<62} ║", "OCCUPANCY").bold());
    println!("{}", format!("║  {:<62} ║", "─".repeat(62)).dimmed());

    let occ_color = util_color(occupancy.occupancy_pct);
    let simd_color = util_color(occupancy.simd_utilization * 100.0);

    println!("{}", format!("║  {:<62} ║", format!(
        "  Active Waves: {} / {}  ({})",
        occupancy.active_waves,
        occupancy.max_waves,
        format!("{:.1}%", occupancy.occupancy_pct).color(occ_color)
    )).cyan());

    println!("{}", format!("║  {:<62} ║", format!(
        "  Avg Occupancy: {:.1}%",
        occupancy.avg_occupancy_pct
    )).cyan());

    let pressure_color = match occupancy.register_pressure.as_str() {
        "High" => "red",
        "Moderate" => "yellow",
        _ => "green",
    };

    println!("{}", format!("║  {:<62} ║", format!(
        "  Register Pressure: {}",
        occupancy.register_pressure.color(pressure_color)
    )).cyan());

    println!("{}", format!("║  {:<62} ║", format!(
        "  SIMD Utilization: {:.1}%",
        occupancy.simd_utilization * 100.0
    ).color(simd_color)).cyan());

    println!("{}", format!("║  {:<62} ║", "").cyan());
}

/// Print detailed per-kernel metrics table
fn print_kernel_details(kernels: &[KernelMetrics]) {
    println!("{}", format!("║  {:<62} ║", "KERNEL DETAILS").bold());
    println!("{}", format!("║  {:<62} ║", "─".repeat(62)).dimmed());

    let header = format!(
        "  {:<18} {:>8} {:>8} {:>6} {:>6} {:>6}",
        "Kernel", "Time", "IPC", "VGPR", "SGPR", "Occ%"
    );
    println!("{}", format!("║  {:<62} ║", header).cyan());

    for k in kernels {
        let line = format!(
            "  {:<18} {:>8} {:>8.1} {:>6} {:>6} {:>5.1}%",
            truncate(&k.name, 18),
            format_duration_us(k.execution_time_us),
            k.ipc,
            k.vgpr_count,
            k.sgpr_count,
            k.occupancy_pct,
        );
        println!("{}", format!("║  {:<62} ║", line).cyan());
    }

    println!("{}", format!("║  {:<62} ║", "").cyan());
}

/// Convert profiling results to CSV format
pub fn to_csv(results: &ProfilingResults) -> Result<String> {
    let mut wtr = csv::Writer::from_writer(vec![]);

    // Write header
    wtr.write_record([
        "kernel", "grid_x", "grid_y", "grid_z", "block_x", "block_y", "block_z",
        "time_us", "total_cycles", "instructions", "ipc", "occupancy_pct",
        "vgpr", "sgpr", "lds_bytes", "l2_hit_rate", "dispatch_count",
    ])?;

    // Write kernel data
    for k in &results.kernels {
        wtr.write_record([
            &k.name,
            &k.grid_dims.0.to_string(),
            &k.grid_dims.1.to_string(),
            &k.grid_dims.2.to_string(),
            &k.block_dims.0.to_string(),
            &k.block_dims.1.to_string(),
            &k.block_dims.2.to_string(),
            &k.execution_time_us.to_string(),
            &k.total_cycles.to_string(),
            &k.instructions.to_string(),
            &format!("{:.2}", k.ipc),
            &format!("{:.1}", k.occupancy_pct),
            &k.vgpr_count.to_string(),
            &k.sgpr_count.to_string(),
            &k.lds_bytes.to_string(),
            &format!("{:.3}", k.l2_hit_rate),
            &k.dispatch_count.to_string(),
        ])?;
    }

    let data = String::from_utf8(wtr.into_inner()?)?;
    Ok(data)
}

/// Format a duration in microseconds to a human-readable string
fn format_duration_us(us: u64) -> String {
    if us >= 1_000_000 {
        format!("{:.3}s", us as f64 / 1_000_000.0)
    } else if us >= 1_000 {
        format!("{:.1}ms", us as f64 / 1_000.0)
    } else {
        format!("{}µs", us)
    }
}

/// Get a color for a utilization percentage
fn util_color(pct: f64) -> &'static str {
    if pct >= 80.0 {
        "green"
    } else if pct >= 50.0 {
        "yellow"
    } else {
        "red"
    }
}

/// Truncate a string to a maximum length
fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max - 1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration_us(500), "500µs");
        assert_eq!(format_duration_us(1500), "1.5ms");
        assert_eq!(format_duration_us(2_500_000), "2.500s");
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello_world_extra", 10), "hello_worl…");
    }

    #[test]
    fn test_util_color() {
        assert_eq!(util_color(90.0), "green");
        assert_eq!(util_color(60.0), "yellow");
        assert_eq!(util_color(30.0), "red");
    }
}
