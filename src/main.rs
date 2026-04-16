//! ROCProbe — A lightweight ROCm GPU profiler for AMD Instinct MI300X
//!
//! This crate provides a high-performance, low-overhead profiling interface
//! for AMD ROCm applications. It hooks into the HIP runtime to capture kernel
//! execution metrics, memory bandwidth utilization, and hardware occupancy data.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────┐
//! │                    ROCProbe Core                          │
//! ├──────────────┬──────────────────┬───────────────────────┤
//! │ CLI Module   │ Profiler Core    │ Metric Collector      │
//! │ (main.rs)    │ (profiler.rs)    │ (metrics.rs)          │
//! └──────────────┴──────────────────┴───────────────────────┘
//! ```

mod metrics;
mod output;
mod profiler;

use anyhow::Result;
use clap::{Parser, Subcommand};

use crate::profiler::ProfilerSession;

/// ROCProbe — Lightweight ROCm GPU profiler for MI300X
#[derive(Parser, Debug)]
#[command(
    name = "rocprobe",
    version = "0.3.1",
    author = "huang-poi",
    about = "A lightweight ROCm GPU profiler for AMD MI300X accelerators"
)]
struct Cli {
    /// Path to the application binary to profile
    #[arg(value_name = "APP")]
    app: Option<String>,

    /// Enable real-time streaming output
    #[arg(long, short)]
    stream: bool,

    /// Streaming interval in milliseconds
    #[arg(long, default_value = "100")]
    interval: u64,

    /// Output format: table, json, csv
    #[arg(long, short, value_enum, default_value = "table")]
    format: OutputFormat,

    /// Output file path (stdout if not specified)
    #[arg(long, short)]
    output: Option<String>,

    /// ROCm device index to profile
    #[arg(long, short, default_value = "0")]
    device: u32,

    /// Comma-separated list of metrics to collect
    #[arg(long, value_delimiter = ',')]
    metrics: Option<Vec<String>>,

    /// Enable verbose logging
    #[arg(long, short)]
    verbose: bool,

    /// Subcommand
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// List available GPU devices and their capabilities
    Devices,

    /// Show supported profiling metrics
    Metrics,

    /// Run a quick hardware benchmark
    Benchmark {
        /// Duration in seconds
        #[arg(long, default_value = "10")]
        duration: u64,
    },
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum OutputFormat {
    Table,
    Json,
    Csv,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    if cli.verbose {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();
    } else {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    }

    log::info!("ROCProbe v{} starting", env!("CARGO_PKG_VERSION"));

    match cli.command {
        Some(Commands::Devices) => {
            list_devices()?;
        }
        Some(Commands::Metrics) => {
            list_metrics()?;
        }
        Some(Commands::Benchmark { duration }) => {
            run_benchmark(duration)?;
        }
        None => {
            // Profile the given application
            let app = cli.app.ok_or_else(|| {
                anyhow::anyhow!("No application specified. Use: rocprobe <app> or see --help")
            })?;

            let mut session = ProfilerSession::builder()
                .device(cli.device)
                .stream(cli.stream)
                .interval_ms(cli.interval)
                .app_path(&app)
                .build()?;

            // Configure metrics
            if let Some(ref metric_names) = cli.metrics {
                session.set_metrics(metric_names)?;
            }

            // Run profiling session
            let results = session.run()?;

            // Output results
            match cli.format {
                OutputFormat::Table => {
                    output::print_table(&results);
                }
                OutputFormat::Json => {
                    let json = serde_json::to_string_pretty(&results)?;
                    if let Some(path) = cli.output {
                        std::fs::write(&path, &json)?;
                        log::info!("Results written to {}", path);
                    } else {
                        println!("{}", json);
                    }
                }
                OutputFormat::Csv => {
                    let csv_str = output::to_csv(&results)?;
                    if let Some(path) = cli.output {
                        std::fs::write(&path, &csv_str)?;
                        log::info!("Results written to {}", path);
                    } else {
                        print!("{}", csv_str);
                    }
                }
            }
        }
    }

    Ok(())
}

/// List all ROCm-compatible GPU devices in the system
fn list_devices() -> Result<()> {
    use crate::metrics::DeviceManager;

    let devices = DeviceManager::query_devices()?;

    println!("\n  ROCProbe — Available Devices\n");
    println!("  {:<6} {:<28} {:<10} {:<12} {:<10}",
             "ID", "Name", "GFX", "Memory", "CUs");
    println!("  {}", "─".repeat(70));

    for dev in &devices {
        println!("  {:<6} {:<28} {:<10} {:<12} {:<10}",
                 dev.id, dev.name, dev.gfx_target,
                 format!("{} GB", dev.memory_gb), dev.compute_units);
    }

    println!();
    Ok(())
}

/// List all supported profiling metrics
fn list_metrics() -> Result<()> {
    use crate::metrics::MetricRegistry;

    let registry = MetricRegistry::new()?;
    let metrics = registry.available_metrics();

    println!("\n  ROCProbe — Supported Metrics\n");
    println!("  {:<24} {:<12} {:<40}",
             "Metric", "Category", "Description");
    println!("  {}", "─".repeat(80));

    for m in metrics {
        println!("  {:<24} {:<12} {:<40}",
                 m.name, m.category, m.description);
    }

    println!();
    Ok(())
}

/// Run a quick GPU benchmark
fn run_benchmark(duration: u64) -> Result<()> {
    use crate::metrics::BenchmarkRunner;

    println!("\n  ROCProbe — Hardware Benchmark ({}s)\n", duration);

    let runner = BenchmarkRunner::new(0)?;
    let results = runner.run(duration)?;

    println!("  Results:");
    println!("  • Peak FP16:    {:.1} TFLOPS", results.fp16_tflops);
    println!("  • Peak FP32:    {:.1} TFLOPS", results.fp32_tflops);
    println!("  • Peak HBM BW:  {:.1} GB/s", results.hbm_bandwidth_gbs);
    println!("  • Peak L2 BW:   {:.1} GB/s", results.l2_bandwidth_gbs);
    println!("  • Peak INT8:    {:.1} TOPS", results.int8_tops);
    println!();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parse() {
        let cli = Cli::try_parse_from(["rocprobe", "--device", "0", "--stream", "./my_app"]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        assert!(cli.stream);
        assert_eq!(cli.device, 0);
    }

    #[test]
    fn test_output_format_parse() {
        let cli = Cli::try_parse_from(["rocprobe", "--format", "json", "./my_app"]);
        assert!(cli.is_ok());
    }
}
