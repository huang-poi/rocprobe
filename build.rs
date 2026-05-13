/// ROCProbe build script
///
/// Detects ROCm installation and sets up linking for native libraries.
use std::env;
use std::path::PathBuf;

fn main() {
    // Try to find ROCm installation
    let rocm_paths = [
        "/opt/rocm",
        "/opt/rocm-6.0",
        "/opt/rocm-5.7",
        "/usr/local/rocm",
    ];

    let rocm_path = rocm_paths.iter().find(|p| std::path::Path::new(p).exists());

    if let Some(path) = rocm_path {
        println!("cargo:rustc-link-search=native={}/lib", path);
        println!("cargo:rustc-link-search=native={}/lib64", path);

        // Link rocprofiler
        println!("cargo:rustc-link-lib=dylib=rocprofiler64");

        // Link HSA runtime
        println!("cargo:rustc-link-lib=dylib=hsa-runtime64");

        // Link HIP runtime
        println!("cargo:rustc-link-lib=dylib=amdhip64");

        println!("cargo:warning=Found ROCm at {}", path);
    } else {
        println!("cargo:warning=ROCm not found. Building in stub mode.");
    }

    // Re-run if ROCm paths change
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
}
