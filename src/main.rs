use clap::{Parser, Subcommand};
use anyhow::Result;

mod profiler;
mod metrics;
mod output;

#[derive(Parser)]
#[command(name = "rocprobe")]
#[command(version = "0.3.1")]
#[command(about = "Lightweight ROCm GPU profiler CLI for MI300X")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// GPU device ID
    #[arg(short, long, default_value_t = 0)]
    gpu: u32,

    /// Output format
    #[arg(short, long, default_value = "table")]
    format: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Profile kernel execution for a target application
    Profile {
        /// Target application or binary
        #[arg(short, long)]
        target: String,

        /// Capture memory transfers
        #[arg(long)]
        mem_transfers: bool,

        /// Max kernels to capture (0 = unlimited)
        #[arg(long, default_value_t = 0)]
        max_kernels: u32,
    },

    /// Stream GPU metrics in real-time
    Monitor {
        /// Polling interval (ms)
        #[arg(short, long, default_value_t = 1000)]
        interval: u64,

        /// Duration in seconds (0 = until Ctrl+C)
        #[arg(short, long, default_value_t = 0)]
        duration: u64,
    },

    /// Analyze a previous profiling session
    Analyze {
        /// Path to profiling JSON
        #[arg(short, long)]
        input: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Profile { target, mem_transfers, max_kernels } => {
            let mut config = profiler::ProfileConfig::default();
            config.capture_mem_transfers = mem_transfers;
            config.max_kernels = max_kernels;
            config.gpu_id = cli.gpu;

            let report = profiler::run_profile(&target, &config)?;
            output::print_report(&report, &cli.format)?;
        }
        Commands::Monitor { interval, duration } => {
            let config = metrics::MonitorConfig {
                interval_ms: interval,
                duration_secs: duration,
                gpu_id: cli.gpu,
            };
            metrics::run_monitor(&config)?;
        }
        Commands::Analyze { input } => {
            let report = profiler::load_report(&input)?;
            output::print_report(&report, &cli.format)?;
        }
    }

    Ok(())
}
