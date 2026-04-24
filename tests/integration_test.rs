use assert_cmd::Command;

#[test]
fn test_help() {
    let mut cmd = Command::cargo_bin("rocprobe").unwrap();
    cmd.arg("--help");
    cmd.assert().success();
}

#[test]
fn test_occupancy_with_fixture() {
    let mut cmd = Command::cargo_bin("rocprobe").unwrap();
    cmd.args(["occupancy", "--trace", "tests/fixtures/sample_trace.json"]);
    let _ = cmd.output();
}
