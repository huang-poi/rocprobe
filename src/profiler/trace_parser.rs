use anyhow::Result;
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct RocprofEntry { op: String, kernel_name: Option<String>, start_timestamp: Option<u64>, end_timestamp: Option<u64> }

pub fn parse_rocprof_v2(path: &Path) -> Result<Vec<super::KernelTrace>> {
    let data = std::fs::read_to_string(path)?;
    let entries: Vec<RocprofEntry> = serde_json::from_str(&data)?;
    let mut traces = Vec::new();
    for e in entries {
        if e.op == "kernel_dispatch" {
            let start = e.start_timestamp.unwrap_or(0);
            let end = e.end_timestamp.unwrap_or(0);
            traces.push(super::KernelTrace {
                kernel_name: e.kernel_name.unwrap_or_default(), grid_size: [1,1,1], block_size: [64,1,1],
                registers_per_thread: 0, shared_memory_bytes: 0, duration_us: (end - start) as f64 / 1000.0,
                occupancy: 0.0, memory_read_bytes: 0, memory_write_bytes: 0, gpu_id: 0, stream_id: 0, timestamp_ns: start,
            });
        }
    }
    Ok(traces)
}

// feat(profiler): support perfetto export
