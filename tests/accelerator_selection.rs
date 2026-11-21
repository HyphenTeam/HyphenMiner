use std::{env, fs, process::Command};

#[test]
fn only_an_explicit_directory_enables_hardware() {
    let install = tempfile::tempdir().unwrap();
    let directory = install.path().join("accelerators");
    fs::create_dir(&directory).unwrap();
    let exe = install
        .path()
        .join(format!("hyphen-miner{}", env::consts::EXE_SUFFIX));
    fs::copy(env!("CARGO_BIN_EXE_hyphen-miner"), &exe).unwrap();
    let plugin = format!("hyphen_backend_cuda.{}", env::consts::DLL_EXTENSION);
    fs::write(directory.join(&plugin), b"invalid test plugin").unwrap();

    // An empty payout address stops startup after backend selection, before dialing a pool.
    let run = |args: &[&str]| {
        Command::new(&exe)
            .current_dir(install.path())
            .env("HYPHEN_ACCELERATOR_DIR", &directory)
            .env("RUST_LOG", "info")
            .args(args)
            .output()
            .unwrap()
    };
    let default = run(&[]);
    assert!(String::from_utf8_lossy(&default.stdout).contains("CPU nonce search"));
    assert!(String::from_utf8_lossy(&default.stderr).contains("requires --wallet-address"));

    let explicit = run(&["--accelerator-dir", "accelerators"]);
    assert!(!explicit.status.success());
    assert!(String::from_utf8_lossy(&explicit.stderr).contains(&plugin));
    assert!(!String::from_utf8_lossy(&explicit.stdout).contains("CPU nonce search"));

    let legacy = run(&["--network", "testnet", "--accelerator-dir", "accelerators"]);
    assert!(!legacy.status.success());
    assert!(String::from_utf8_lossy(&legacy.stderr).contains("full GPU nonce search requires"));

    let check = run(&["accelerator-check"]);
    assert!(!check.status.success());
    assert!(String::from_utf8_lossy(&check.stderr).contains("requires --accelerator-dir"));
}
