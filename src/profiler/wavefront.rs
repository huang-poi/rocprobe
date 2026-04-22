/// CDNA3 wavefront occupancy calculator
/// MI300X: 304 CUs, 16 max waves/CU, 64 threads/wavefront

#[derive(Debug)]
pub struct OccupancyParams {
    pub vgprs_per_thread: u32,
    pub sgprs_per_thread: u32,
    pub lds_bytes_per_workgroup: u64,
    pub workgroup_size: u64,
    pub waves_per_workgroup: u64,
}

#[derive(Debug)]
pub struct OccupancyResult {
    pub max_waves_per_cu: u32,
    pub achieved_occupancy: f64,
    pub limiting_factor: String,
    pub suggestions: Vec<String>,
}

const MAX_VGPRS: u32 = 1024;
const MAX_SGPRS: u32 = 800;
const MAX_LDS: u64 = 65536;
const MAX_WAVES: u32 = 16;

pub fn calculate_occupancy(params: &OccupancyParams) -> OccupancyResult {
    let vgpr_limited = MAX_VGPRS / params.vgprs_per_thread.max(1);
    let sgpr_limited = MAX_SGPRS / params.sgprs_per_thread.max(1);
    let lds_limited = if params.lds_bytes_per_workgroup > 0 { (MAX_LDS / params.lds_bytes_per_workgroup) as u32 } else { u32::MAX };
    let max_waves = vgpr_limited.min(sgpr_limited).min(lds_limited).min(MAX_WAVES);
    let mut suggestions = Vec::new();
    let limiting = if vgpr_limited <= sgpr_limited && vgpr_limited <= lds_limited {
        if params.vgprs_per_thread > 64 { suggestions.push("Reduce VGPR usage".into()); }
        "VGPRs"
    } else if sgpr_limited <= lds_limited { "SGPRs" } else { "LDS" };
    OccupancyResult { max_waves_per_cu: max_waves, achieved_occupancy: max_waves as f64 / MAX_WAVES as f64, limiting_factor: limiting.to_string(), suggestions }
}
