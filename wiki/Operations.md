# Miner operations · 矿工运行指南

## Identity and reward address

Create a persistent signing key once with `cargo run --release --locked -- keygen --output miner.key`. Pass `--key-file miner.key` on subsequent starts. Use `--wallet-address` for the reward address. A new ephemeral signing identity on each restart can split worker statistics. The miner checks each job's network parameters and authorizes the reward keys in the candidate header.

On the standard devnet, a credit share contains new Cunningham witnesses. A miner can extend its own share to the full block difficulty. On `neural-devnet` each accepted solution is a full block candidate, and the pool must run in SOLO mode. To participate in shared settlement on the standard devnet, review the pool's payout wallet and explicitly set `--allow-shared-reward-recipient`.

## CPU, plugins and connections

Use `--threads` to limit CPU workers and `--no-accelerator` to select the portable CPU search. To build and check the CUDA sieve on a host with the CUDA toolkit and a supported device, run the following in the miner repository.

```sh
cmake -S native -B native/build -DHYPHEN_ENABLE_CUDA=ON
cmake --build native/build --config Release
cargo run --release --locked -- --accelerator-dir ./accelerators accelerator-check --batch 2000000 --offset 1000001 --stride 32 --length 6
```

For ROCm, select `-DHYPHEN_ENABLE_HIP=ON` and run the same comparison. The check compares device survivors with the CPU calculation. The miner falls back to CPU search on a device error. An operator-provided proxy can be selected with `--via-proxy socks5://HOST:PORT` or `--via-proxy http://HOST:PORT`. Configure `--pool` with the destination pool address in both cases.

The miner queues up to two candidates and tracks up to 16 submissions awaiting acknowledgment. When the connection is restored, the miner starts a new session and retains the witness claims recorded by the running process. It moves large claim sets to temporary storage. Restarting the process clears the local record. The pool and the node each check whether a witness has already been used.

For neural work use `--network neural-devnet --accelerator-dir ./accelerators`. The CUDA plugin derives candidates and scores them in batches of up to 32,768. The miner recomputes the winning score in Rust. Each solution log identifies the backend and search duration. To compare entire GPU score windows on your device, set `HYPHEN_ACCELERATOR_DIR` to the plugin directory and run `cargo test --release --locked native_neural_windows_match_rust -- --ignored`.

## 中文说明

首次启动前，运行 `cargo run --release --locked -- keygen --output miner.key` 生成身份密钥。此后通过 `--key-file miner.key` 使用同一密钥，通过 `--wallet-address` 指定奖励地址。在标准开发网上，矿池按新增的坎宁安见证计算份额。矿工可以继续搜索，直至其提交的见证达到候选区块所需的难度。在 `neural-devnet` 上，每个有效解均可作为候选区块提交，矿池须采用 SOLO 模式。在标准开发网上参与共享结算前，请核对矿池的收款地址，并通过 `--allow-shared-reward-recipient` 明确授权。

使用 `--threads` 限制 CPU 线程数。上面的 CMake 命令构建 CUDA 插件，ROCm 主机改用 `-DHYPHEN_ENABLE_HIP=ON`。`accelerator-check` 在本机对照 CPU 结果；`--no-accelerator` 强制使用 CPU。代理可选 SOCKS5 或 HTTP CONNECT，`--pool` 始终填写矿池地址。

矿工最多排队两个候选，同时等待最多 16 个提交回执。同一进程重连时保留见证去重记录，记录增大后转入临时文件；进程重启后由矿池及链上记录继续判重。

使用 `--network neural-devnet --accelerator-dir ./accelerators` 可启用 CUDA 神经工作量搜索。每批最多派生并计算 32,768 个候选，矿工用 Rust 复算达标候选的分数，日志记录实际后端和搜索耗时。将 `HYPHEN_ACCELERATOR_DIR` 指向插件目录，运行 `cargo test --release --locked native_neural_windows_match_rust -- --ignored` 可在本机逐项核对完整窗口。
