#[derive(Debug)]
pub struct GpuArch {
    pub name: String,
    pub compute_units: u32,
    pub peak_fp32_tflops: f64,
    pub peak_memory_bw_gbs: f64,
    pub memory_size_gb: u64,
}

impl GpuArch {
    pub fn mi300x() -> Self {
        Self { name: "MI300X (gfx942)".into(), compute_units: 304, peak_fp32_tflops: 163.4, peak_memory_bw_gbs: 5300.0, memory_size_gb: 192 }
    }
    pub fn mi250x() -> Self {
        Self { name: "MI250X (gfx90a)".into(), compute_units: 220, peak_fp32_tflops: 47.9, peak_memory_bw_gbs: 3200.0, memory_size_gb: 128 }
    }
    pub fn ridge_point(&self) -> f64 { self.peak_fp32_tflops * 1e12 / (self.peak_memory_bw_gbs * 1e9) }
}

// feat(metrics): add MI300A APU support
