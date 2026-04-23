use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::process::Command;

#[derive(Debug, Serialize, Deserialize)]
pub struct ThermalStatus {
    pub edge_temp_c: u32,
    pub junction_temp_c: u32,
    pub power_draw_w: f64,
    pub is_throttled: bool,
    pub throttle_reason: Option<String>,
}

pub fn read_thermal(device: usize) -> Result<ThermalStatus> {
    let output = Command::new("rocm-smi").args(["-d", &device.to_string(), "-t"]).output()?;
    let raw = String::from_utf8_lossy(&output.stdout);
    Ok(ThermalStatus {
        edge_temp_c: 0, junction_temp_c: 0, power_draw_w: 0.0,
        is_throttled: false, throttle_reason: None,
    })
}
