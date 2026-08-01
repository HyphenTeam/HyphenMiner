# Hyphen Miner

[中文](README_CN.md)

Hyphen Miner is the standalone scientific-computation miner for Hyphen block version 3. It speaks Pool Protocol v5 and rejects legacy pool, chain-identity, and block versions.

## Scientific PoUW

Mining executes the same deterministic 64x64 Q12 diffusion PDE as the node and pool. The block difficulty is the exact full-work iteration count. A lower iteration count is a pool accounting checkpoint, not a low hash.

For each job the miner:

1. Verifies network magic, consensus parameter hash, genesis hash, block version, height, parent, epoch seed, reward keys, transaction bytes, and transaction root.
2. Derives the scientific input from the complete header except pow_commitment.
3. Computes and submits a lower-iteration checkpoint when Pool v5 requests one.
4. Continues to the full block iteration count, commits the final field, and signs the complete header and reward keys.
5. Verifies the pool's signed receipt chain for every accepted submission.

BLAKE3 is used only for deterministic input derivation, commitments, message digests, and receipt integrity. The miner does not compare a digest to a difficulty target.

The protocol name is PoUW v1, but this block kernel is not backed by a user-published scientific task. It proves execution of the specified PDE and nothing more. It does not provide succinct verification or prove external scientific utility. Telemetry counts seven specified arithmetic operations per cell update; it is not a native instruction or energy count.

The fixed cross-implementation vector is:

```text
25078c250c5b44211bbf0fea60e90ac7024df6ff94d154161852fdd72684e524
```

## Native accelerators

Native ABI v2 defines the exact consensus 64x64, four-neighbour, periodic-boundary Q12 kernel for CUDA, HIP and OpenVINO providers without linking vendor SDKs into the Rust protocol process. Mining threads use a device only after its output passes an independent Rust cross-check. Old ABI v1 plugins are rejected. QNN remains fail-closed until a target-specific graph package implements ABI v2.

When no verified device is present, mining uses the Rust CPU kernel. `--require-accelerator` disables that fallback: startup and worker-side plugin loading must both succeed. A device execution failure then stops mining instead of submitting an unchecked result.

Unavailable SDKs or devices are reported as unavailable. A provider that initializes but cannot execute the exact deterministic kernel is not advertised as compute-capable. This verification is not a SNARK/STARK or an AetherCompute user-task settlement proof.

Build available backends:

```powershell
cmake -S .\native -B .\native\build -DHYPHEN_ENABLE_CUDA=ON
cmake --build .\native\build --config Release
```

Inspect devices:

```powershell
.\target\release\hyphen-miner.exe --accelerator-dir .\accelerators accelerators
```

## Build

```powershell
cargo build --release --locked
cargo test --locked
cargo clippy --all-targets --locked -- -D warnings
.\target\release\hyphen-miner.exe keygen --output .\miner.key
```

Connect to a pool:

```powershell
.\target\release\hyphen-miner.exe `
  --pool 127.0.0.1:3340 `
  --network devnet `
  --key-file .\miner.key `
  --wallet-address '<devnet-hy1-address>' `
  --threads 4 `
  --accelerator-dir .\accelerators
```

Use the current binary's --help output for authoritative arguments. miner.key is a protocol identity key, not a wallet recovery key.

## Compatibility and security

- Pool protocol: 5
- Block version: 3
- PoUW protocol: 1
- Devnet consensus parameters: 54bf97e4e28d4fcf963d884a555a8425bbfe7c84d2753001bcabbaf116232fda
- Devnet genesis: 47d530160cfef9141fe3b37b886e09b9f96ec4dc93d6c05005b9c6dbf35b1972

RustBinary uses the bounded fixed-width little-endian profile and rejects trailing bytes. Consensus-facing migrations must fail explicitly; format fallback is forbidden in commitment, signature and authorization paths.

This software has not received an independent consensus, cryptographic or accelerator audit. Use research networks only.

## License

Hyphen Miner is licensed under the PolyForm Strict License 1.0.0. See [LICENSE](LICENSE).
