use anyhow::Result;
use crate::profiler::ProfileResult;
use serde::Serialize;

pub fn display_profile(result: &ProfileResult, format: &str) -> Result<()> {
    match format {
        "json" => println!("{}", serde_json::to_string_pretty(result)?),
        "csv" => {
            let mut wtr = csv::Writer::from_writer(std::io::stdout());
            wtr.write_record(&["kernel", "duration_us", "occupancy"])?;
            for t in &result.traces {
                wtr.write_record(&[&t.kernel_name, &t.duration_us.to_string(), &format!("{:.4}", t.occupancy)])?;
            }
            wtr.flush()?;
        }
        _ => {
            println!("=== ROCprobe Profile Report ===");
            println!("Total Duration: {:.2}s", result.total_duration_s);
            println!("Avg Kernel Time: {:.1}us", result.avg_kernel_time_us);
            println!("Kernels: {}", result.traces.len());
            for t in &result.traces {
                println!("  {}: {:.1}us ({:.1}% occ)", t.kernel_name, t.duration_us, t.occupancy * 100.0);
            }
        }
    }
    Ok(())
}

// feat(output): add colored terminal output
