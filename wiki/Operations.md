# Miner operations · 矿工运行指南

## Identity and reward address

Create a persistent signing key once with `cargo run --release --locked -- keygen --output miner.key`. Pass `--key-file miner.key` on subsequent starts. Use `--wallet-address` for the reward address. A new ephemeral signing identity on each restart can split worker statistics. The miner checks each job's network parameters and authorizes the reward keys in the candidate header.

Each accepted SOLO nonce must satisfy both the fixed neural gate and the block’s final BLAKE3 scratchpad threshold. Reward authorization binds the miner’s receiving keys.

## CPU, plugins and connections

Use `--threads N --no-accelerator` to select CPU workers. To build the CUDA backend, run `cmake -S native -B native/build -DHYPHEN_ENABLE_CUDA=ON` followed by `cmake --build native/build --config Release`. Point `--accelerator-dir` at the directory containing the plugin. It computes batches of neural scores and the host checks qualifying nonces. To verify complete device windows against Rust, set `HYPHEN_ACCELERATOR_DIR` and run `cargo test --release --locked native_neural_windows_match_rust -- --ignored`.

Run `hyphen-miner --accelerator-dir DIRECTORY accelerator-check --offset 125 --batch 1` to reproduce the first-height HYN3 vector. The check compares every returned device score with Rust, applies the memory gate and prints `scores=1 accepted=1` for this nonce. [Reference vector](https://github.com/HyphenTeam/Hyphen/blob/main/test-vectors/neural-block-work-v3.json)

For an operator-provided SOCKS5 or HTTP CONNECT proxy use `--via-proxy socks5://HOST:PORT` or `--via-proxy http://HOST:PORT`. Set `--pool` to the destination pool address. Job changes invalidate outstanding work for that parent.

## 中文说明

首次运行使用 `keygen --output miner.key` 生成签名密钥；此后用 `--key-file miner.key` 复用该身份，用 `--wallet-address` 指定奖励地址。CPU 可通过 `--threads` 限制工作线程数，CUDA 通过 `--accelerator-dir` 加载插件。矿池按 SOLO 方式接收通过两道门槛的候选区块。
