# Hyphen Miner

Hyphen Miner searches the useful-work proof of Hyphen block version 5 (PoUW v3):
single-use Cunningham prime chains. It speaks Pool Protocol v6 and refuses any
pool, chain identity, or block version that does not match the compiled
consensus constants.

Consensus types, hashing, keys, and the wire profile are not reimplemented here.
`hyphen-core`, `hyphen-crypto`, `hyphen-pow`, and `hyphen-transport` are path
dependencies, so a consensus change reaches the miner with the node instead of
being copied and forgotten.

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
- `src/protocol.rs` — Pool Protocol v6 envelope and messages
- `src/primitives.rs` — canonical re-exports; no consensus code lives here
- `native/` — retired v2 diffusion kernels, kept as C++/CUDA source only

No code path loads `native/`. The v2 diffusion field it implements no longer
exists in consensus, so the miner makes no device-acceleration claim; the
sources are kept for a future sieve-accelerated search.

## Build and run

```powershell
cargo build --release --locked
cargo test --locked
cargo clippy --all-targets --locked -- -D warnings

.\target\release\hyphen-miner.exe --network devnet --pool 127.0.0.1:3340 `
    --wallet-address hy1...
```

Useful flags: `--threads`, `--batch-size` (candidates per search call),
`--allow-shared-reward-recipient` (only after reviewing the pool policy).

## Mining through a proxy

A miner on a network that only permits traffic to a proxy endpoint can hand that
problem to [Corduit](https://crates.io/crates/corduit):

```powershell
.\target\release\hyphen-miner.exe --network devnet `
    --via-proxy socks5://user:pass@proxy.internal:1080 `
    --pool pool.example:3340 --wallet-address hy1...
```

`--via-proxy` accepts `socks5://`, `socks4://` and `http://` endpoints with
optional `user:password` credentials. The miner starts Corduit's engine in its
own process with a loopback `mixed` inbound, routes the pool link through the
operator's outbound, and — because the pool name is carried as a name — lets the
proxy resolve it instead of asking this host's resolver. Everything above the
socket is untouched: the share protocol, the keepalive and the framing are the
same bytes as a direct connection.

Notes an operator should know:

- A catch-all rule selects the operator's outbound, and the direct outbound that
  Corduit requires to exist for its own bookkeeping is never reached by a rule.
  A proxy that cannot be used is an error, never a silent direct connection.
- `--via-proxy-listen 127.0.0.1:PORT` pins the loopback port instead of letting
  the miner scan for a free one.
- Only lawfully usable proxies: the endpoint is yours to configure and to be
  entitled to.
- Corduit is a hard dependency, so `--via-proxy` is always available. It brings
  its own in-tree protocol stack; Hysteria and TUIC stay behind Corduit's opt-in
  features and are absent from this build.

## Protocol

- Block version: 5
- PoUW protocol: 3
- Pool protocol: 6

## Limits

This software has not received an independent consensus or cryptographic audit.
Use research networks only.