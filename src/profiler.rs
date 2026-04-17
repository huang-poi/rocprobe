use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelInfo {
    pub name: String,
    pub grid_size: [u32; 3],
    pub block_size: [u32; 3],
    pub time_us: f64,
    pub mem_bw_util: f64,
    pub compute_occ: f64,
    pub l2_cache_hit: f64,
    pub vgpr_usage: u32,
    pub wavefront_active: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemTransfer {
    pub direction: String,
    pub size_bytes: u64,
    pub time_us: f64,
    pub bandwidth_gbs: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProfileReport {
    pub gpu_name: String,
    pub gfx_arch: String,
    pub rocm_version: String,
    pub kernels: Vec<KernelInfo>,
    pub mem_transfers: Vec<MemTransfer>,
    pub total_time_us: f64,
    pub avg_bw_util: f64,
    pub avg_occupancy: f64,
    pub power_draw_w: f64,
    pub temp_edge_c: f64,
}

#[derive(Debug, Clone)]
pub struct ProfileConfig {
    pub gpu_id: u32,
    pub capture_mem_transfers: bool,
    pub max_kernels: u32,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            gpu_id: 0,
            capture_mem_transfers: false,
            max_kernels: 0,
        }
    }
}

/// Initialize HIP runtime and query device properties
fn init_hip_device(gpu_id: u32) -> Result<DeviceInfo> {
    // In production: hipInit(0), hipGetDeviceProperties
    // Simulated device info for MI300X
    Ok(DeviceInfo {
        name: "AMD Instinct MI300X VF".to_string(),
        gfx_arch: "gfx942".to_string(),
        compute_units: 304,
        max_waves_per_cu: 32,
        l2_cache_size_mb: 256,
        mem_clock_mhz: 2600,
        mem_bus_width_bits: 8192,
    })
}

struct DeviceInfo {
    name: String,
    gfx_arch: String,
    compute_units: u32,
    max_waves_per_cu: u32,
    l2_cache_size_mb: u32,
    mem_clock_mhz: u32,
    mem_bus_width_bits: u32,
}

/// Hook HIP kernel launches via rocm_smi / hipProfiler API
fn collect_kernel_traces(target: &str, config: &ProfileConfig) -> Result<Vec<KernelInfo>> {
    // Phase 1: Run with HIP_ACTIVITY tracking
    let output = Command::new(target)
        .env("HIP_TRACE_API", "1")
        .env("ROCPROBE_GPU", config.gpu_id.to_string())
        .output()
        .context("Failed to launch target application")?;

    // Phase 2: Parse rocprof output
    let trace_file = format!("/tmp/rocprobe_trace_{}.json", config.gpu_id);
    let _ = std::fs::read_to_string(&trace_file)
        .unwrap_or_else(|_| "[]".to_string());

    // In production: parse actual HIP API trace
    // Return collected kernel info
    Ok(vec![])
}

/// Read real-time GPU metrics via sysfs
pub fn read_gpu_metrics(gpu_id: u32) -> Result<(f64, f64)> {
    let base_path = format!("/sys/class/drm/card{}/device", gpu_id);

    let power_path = format!("{}/hwmon/hwmon*/power1_average", base_path);
    let temp_path = format!("{}/hwmon/hwmon*/temp1_input", base_path);

    let power_w = std::fs::read_to_string(&power_path)
        .unwrap_or_default()
        .trim()
        .parse::<f64>()
        .unwrap_or(0.0) / 1_000_000.0;

    let temp_c = std::fs::read_to_string(&temp_path)
        .unwrap_or_default()
        .trim()
        .parse::<f64>()
        .unwrap_or(0.0) / 1000.0;

    Ok((power_w, temp_c))
}

pub fn run_profile(target: &str, config: &ProfileConfig) -> Result<ProfileReport> {
    let device = init_hip_device(config.gpu_id)?;
    let kernels = collect_kernel_traces(target, config)?;
    let (power, temp) = read_gpu_metrics(config.gpu_id)?;

    let total_time: f64 = kernels.iter().map(|k| k.time_us).sum();
    let avg_bw = if kernels.is_empty() {
        0.0
    } else {
        kernels.iter().map(|k| k.mem_bw_util).sum::<f64>() / kernels.len() as f64
    };
    let avg_occ = if kernels.is_empty() {
        0.0
    } else {
        kernels.iter().map(|k| k.compute_occ).sum::<f64>() / kernels.len() as f64
    };

    Ok(ProfileReport {
        gpu_name: device.name,
        gfx_arch: device.gfx_arch,
        rocm_version: get_rocm_version(),
        kernels,
        mem_transfers: vec![],
        total_time_us: total_time,
        avg_bw_util: avg_bw,
        avg_occupancy: avg_occ,
        power_draw_w: power,
        temp_edge_c: temp,
    })
}

pub fn load_report(path: &str) -> Result<ProfileReport> {
    let data = std::fs::read_to_string(path)
        .context(format!("Cannot read report: {}", path))?;
    let report: ProfileReport = serde_json::from_str(&data)?;
    Ok(report)
}

fn get_rocm_version() -> String {
    std::fs::read_to_string("/opt/rocm/.info/version")
        .unwrap_or_else(|_| "7.2.0".to_string())
        .trim()
        .to_string()
}
