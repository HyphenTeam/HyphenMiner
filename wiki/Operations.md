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

The miner retains at most two queued candidates and 16 unacknowledged submissions. Reconnect starts a fresh session while preserving the running process's witness claims. Large claim sets spill to temporary storage. Restarting the process clears these local claims. The pool and chain independently check prior use.

## 中文说明

首次运行 `cargo run --release --locked -- keygen --output miner.key` 创建签名身份，之后使用 `--key-file miner.key` 复用。`--wallet-address` 指定收款地址。标准开发网按新增的坎宁安见证记份额，同一矿工可以把自己的份额扩展为完整区块候选。`neural-devnet` 的每个有效解都是完整候选，矿池须使用 SOLO。标准网共享结算需核对矿池钱包后显式传入 `--allow-shared-reward-recipient`。

使用 `--threads` 限制 CPU 线程数。上面的 CMake 命令构建 CUDA 插件，ROCm 主机改用 `-DHYPHEN_ENABLE_HIP=ON`。`accelerator-check` 在本机对照 CPU 结果；`--no-accelerator` 强制使用 CPU。代理可选 SOCKS5 或 HTTP CONNECT，`--pool` 始终填写矿池地址。

矿工最多排队两个候选，同时等待最多 16 个提交回执。同一进程重连时保留见证去重记录，记录增大后转入临时文件；进程重启后由矿池及链上记录继续判重。
