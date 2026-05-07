use crate::metrics::hardware::GpuArch;

pub struct RooflineAnalysis {
    pub kernel_name: String,
    pub arithmetic_intensity: f64,
    pub achieved_gflops: f64,
    pub bottleneck: String,
    pub efficiency_pct: f64,
}

pub fn analyze_roofline(name: &str, flops: u64, bytes: u64, duration_ns: u64, arch: &GpuArch) -> RooflineAnalysis {
    let ai = flops as f64 / bytes as f64;
    let gflops = (flops as f64 / (duration_ns as f64 / 1e9)) / 1e9;
    let ridge = arch.ridge_point();
    let bottleneck = if ai < ridge { "Memory" } else { "Compute" };
    RooflineAnalysis { kernel_name: name.to_string(), arithmetic_intensity: ai, achieved_gflops: gflops, bottleneck: bottleneck.to_string(), efficiency_pct: gflops / (arch.peak_fp32_tflops * 1000.0) * 100.0 }
}

// feat(profiler): generate ASCII roofline plot

// feat(profiler): add SVG roofline export
