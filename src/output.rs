use anyhow::Result;
use crate::profiler::ProfileReport;
use colored::*;

pub fn print_report(report: &ProfileReport, format: &str) -> Result<()> {
    match format {
        "json" => {
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        "table" | _ => {
            print_table(report);
        }
    }
    Ok(())
}

fn print_table(report: &ProfileReport) {
    let width = 70;

    println!("{}", "=".repeat(width));
    println!("  {} v0.3.1", "rocprobe".bold());
    println!("  GPU 0: {} ({})", report.gpu_name.cyan(), report.gfx_arch);
    println!("  ROCm {}", report.rocm_version);
    println!("{}", "=".repeat(width));

    if report.kernels.is_empty() {
        println!("  No kernels captured.");
        return;
    }

    println!();
    println!("  {:<25} {:>10} {:>10} {:>12}",
             "Kernel", "Time(μs)", "Mem BW %", "Occupancy %");
    println!("  {}", "-".repeat(60));

    for k in &report.kernels {
        let bw_color = if k.mem_bw_util > 80.0 {
            k.mem_bw_util.to_string().green()
        } else if k.mem_bw_util > 50.0 {
            k.mem_bw_util.to_string().yellow()
        } else {
            k.mem_bw_util.to_string().red()
        };

        println!("  {:<25} {:>10} {:>10} {:>12}",
                 k.name.dimmed(),
                 format!("{:>8}", format_num(k.time_us)),
                 bw_color,
                 format!("{:.1}", k.compute_occ));
    }

    println!();
    println!("  Summary: {} kernels | {} μs total | {:.1}% avg BW | {:.1}% avg occupancy",
             report.kernels.len().to_string().bold(),
             format_num(report.total_time_us),
             report.avg_bw_util,
             report.avg_occupancy);

    if report.power_draw_w > 0.0 {
        println!("  Power: {:.1}W | Temp: {:.1}°C", report.power_draw_w, report.temp_edge_c);
    }
    println!();
}

fn format_num(n: f64) -> String {
    if n >= 1_000_000.0 {
        format!("{:.1}M", n / 1_000_000.0)
    } else if n >= 1_000.0 {
        format!("{:.1}K", n / 1_000.0)
    } else {
        format!("{:.1}", n)
    }
}
