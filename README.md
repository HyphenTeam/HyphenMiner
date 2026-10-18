# Hyphen Miner

Hyphen Miner searches the useful-work proof of Hyphen block version 5 (PoUW v3):
single-use Cunningham prime chains. It speaks Pool Protocol v6 and refuses any
pool, chain identity, or block version that does not match the compiled
consensus constants.

The witness search always has a portable CPU implementation and can optionally
use a native CUDA or HIP small-prime sieve. The accelerator never decides
whether a witness is valid: it returns sieve survivors, and the Rust side runs
deterministic Miller-Rabin plus the local single-use check on every survivor.

Consensus types, hashing, keys, and the wire profile are not reimplemented here.
The miner builds against the sibling `crates/` tree from `HyphenTeam/Hyphen`:
`../Cargo.toml` and `../crates` are materialized by CI and the release workflow
before the build. That keeps the miner and the consensus implementation in one
test surface instead of a copied snapshot; a consensus change is tested here
before either repository is released. The base tree is never vendored into this
repository.

## What the miner computes

A witness is an odd `n >= 3` whose Cunningham chain `n, 2n+1, 4n+3, ...` is
prime for `pouw_chain_length` elements. The chain length is fixed per network and
carried in the job, so a wrong network is rejected instead of producing work no
node can use.

Each block requires `difficulty` witnesses (1..=4096). For every round the miner:

1. Scans its own residue class (`stride = 2 * threads`, odd offset per thread),
   so workers never overlap and never need to coordinate.
2. Submits a credit share as soon as `share_difficulty` witnesses exist, then
   keeps scanning until the list is complete.
3. Sorts the list ascending, computes
   `pow_commitment = witness_commitment(parent_commitment, height, witnesses)`,
   signs the header with the miner key, and submits.

Witnesses are single use. The node stores every claimed witness in a sparse
Merkle set, so a precomputed pool can be spent exactly once and precomputation
degrades into mining early with no advantage. The miner also remembers what it
already sent: that memory is not consensus state, it only stops the miner from
paying twice for the same witness. The pool and the node reject duplicates on
their own.

## Files

- `src/main.rs` — connection, jobs, witness search, share submission
- `src/accelerator.rs` — runtime ABI loader and CPU fallback
- `src/protocol.rs` — Pool Protocol v6 envelope and messages
- `src/primitives.rs` — canonical re-exports; no consensus code lives here
- `native/` — CUDA and HIP Cunningham sieve plugins (ABI v2, kernel 3)

The native plugins implement only the small-prime sieve over a candidate window.
They are optional shared libraries, loaded from `accelerators/` or from
`HYPHEN_ACCELERATOR_DIR`. A missing plugin, an unsupported device, or a device
error switches the worker to the CPU path; it never weakens verification.

## Build and run

```powershell
cargo build --release --locked
cargo test --locked
cargo clippy --all-targets --locked -- -D warnings

.\target\release\hyphen-miner.exe --network devnet --pool 127.0.0.1:3340 `
    --wallet-address hy1...
```

Useful flags: `--threads`, `--batch-size` (candidates per search call),
`--allow-shared-reward-recipient` (only after reviewing the pool policy),
`--no-accelerator` (force the CPU path), and `--accelerator-dir` (plugin
directory).

## Native accelerator

Build the CUDA plugin on a machine with the CUDA toolkit:

```powershell
cmake -S native -B native/build -DHYPHEN_ENABLE_CUDA=ON
cmake --build native/build --config Release
```

Build the HIP plugin on a machine with ROCm/HIP:

```powershell
cmake -S native -B native/build -DHYPHEN_ENABLE_HIP=ON
cmake --build native/build --config Release
```

The plugins land in `accelerators/`. The loader also accepts an explicit
`--accelerator-dir` or `HYPHEN_ACCELERATOR_DIR`. `HYPHEN_CUDA_ARCHITECTURES`
defaults to `native`; `HYPHEN_HIP_ARCHITECTURES` is empty by default, so hipcc
chooses. Set an explicit list for a binary you intend to distribute to other
GPUs.

Verify the device sieve against the CPU predicate before mining:

```powershell
cargo run --release -- accelerator-check --batch 2000000 --offset 1000001 `
    --stride 32 --length 6
```

The command fails if the GPU survivor set differs from the CPU small-prime
sieve, or if the plugin returns values outside the requested window. It reports
device time, wall time, survivor count, and the number of fully valid chains.

Platform behavior:

| Platform | Backend | Fallback |
|---|---|---|
| Linux x86_64/aarch64 | CUDA or HIP plugin | CPU |
| Windows x86_64 | CUDA or HIP plugin | CPU |
| macOS | none built | CPU |
| Android arm64/armv7 | none built | CPU |
| RISC-V | none built | CPU |

The CPU search is the canonical path on every target. macOS Metal, Android
Adreno, and other NPU/DSP backends are not wired into this binary; no
acceleration claim is made for them.

## Mining through a proxy

A miner on a network that only permits traffic to a proxy endpoint can still
reach its pool:

```powershell
.\target\release\hyphen-miner.exe --network devnet `
    --via-proxy socks5://user:pass@proxy.internal:1080 `
    --pool pool.example:3340 --wallet-address hy1...
```

`--via-proxy` accepts `socks5://[user:pass@]host:port` and
`http://[user:pass@]host:port` (HTTP `CONNECT`). The miner performs that
handshake itself — RFC 1928 with RFC 1929 authentication, or RFC 9110 §9.3.6
with `Proxy-Authorization: Basic` — then speaks the pool protocol through the
resulting stream. Because the pool is carried as a name, the proxy resolves it
and this host's resolver never learns the pool address. Everything above the
socket is untouched: the share protocol, the keepalive and the framing are the
same bytes as a direct connection.

Notes an operator should know:

- The endpoint is the only thing that can reach the pool. A proxy that cannot be
  used is a startup error — the miner never falls back to a direct connection.
- `--via-proxy` is resolved once at startup, so a typo fails before any hashing
  begins rather than at the first reconnect after a share is lost.
- Whatever obfuscation or egress policy you run behind that endpoint is yours:
  [Corduit](https://crates.io/crates/corduit) exposes exactly these two inbounds
  (`socks5`, `http`), so it works as the service this flag dials without being
  linked into the miner — no proxy engine, no extra threads in the mining
  process, and no library version coupling between them.
- Only lawfully usable proxies: the endpoint is yours to configure and to be
  entitled to.
- Credentials belong in the spec, not in a shell history you share: prefer a
  wrapper script or your platform's secret store over a literal command line.

## Protocol

- Block version: 5
- PoUW protocol: 3
- Pool protocol: 6

## Limits

This software has not received an independent consensus or cryptographic audit.
Use research networks only.
