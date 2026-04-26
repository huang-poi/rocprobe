use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BenchmarkResult { pub name: String, pub value: f64, pub unit: String }

pub fn compare_benchmarks(baseline: &[BenchmarkResult], current: &[BenchmarkResult]) {
    let base_map: HashMap<&str, &BenchmarkResult> = baseline.iter().map(|b| (b.name.as_str(), b)).collect();
    println!("Benchmark Comparison:");
    for c in current {
        if let Some(b) = base_map.get(c.name.as_str()) {
            let change = ((c.value - b.value) / b.value) * 100.0;
            println!("  {}: {:.2} -> {:.2} ({:+.1}%)", c.name, b.value, c.value, change);
        }
    }
}
