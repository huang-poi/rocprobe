use clap::{Parser, Subcommand};
mod metrics;
mod output;
mod profiler;

#[derive(Parser)]
#[command(name = "rocprobe")]
#[command(about = "Lightweight ROCm GPU profiler")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Profile {
        #[arg(short, long)]
        app: String,
        #[arg(short, long, default_value = "table")]
        format: String,
        #[arg(short, long, default_value = "10")]
        duration: u64,
    },
    Status {
        #[arg(short, long, default_value = "0")]
        device: usize,
        #[arg(short, long, default_value = "table")]
        format: String,
    },
    Occupancy {
        #[arg(short, long)]
        trace: String,
    },
    Memband {
        #[arg(short, long, default_value = "0")]
        device: usize,
        #[arg(long, default_value = "100")]
        interval: u64,
        #[arg(short, long, default_value = "table")]
        format: String,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Profile {
            app,
            format,
            duration,
        } => profiler::run(&app, &format, duration),
        Commands::Status { device, format } => metrics::query_status(device, &format),
        Commands::Occupancy { trace } => metrics::analyze_occupancy(&trace),
        Commands::Memband {
            device,
            interval,
            format,
        } => metrics::memory_bandwidth(device, interval, &format),
    }
}

// refactor: improve CLI error messages

// feat: add batch profiling mode
