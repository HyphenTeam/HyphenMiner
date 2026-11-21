# Miner operations · 矿工运行指南

## Identity and reward address

Create a persistent signing key once with `cargo run --release --locked -- keygen --output miner.key`. Pass `--key-file miner.key` on subsequent starts. Use `--wallet-address` for the reward address. A new ephemeral signing identity on each restart can split worker statistics. The miner checks each job's network parameters and authorizes the reward keys in the candidate header.

Each accepted SOLO nonce must satisfy both the fixed neural gate and the block’s final BLAKE3 scratchpad threshold. Reward authorization binds the miner’s receiving keys.

## CPU, plugins and connections

CPU mining is the default, including when an `accelerators` directory exists next to the executable. Set `--threads N` to limit CPU workers. To build CUDA, run `cmake -S native -B native/build -DHYPHEN_ENABLE_CUDA=ON` followed by `cmake --build native/build --config Release`. Pass `--accelerator-dir DIRECTORY` to each GPU invocation. HYN3 uses a single host controller and runs the full search on the device. A missing, incompatible or failed plugin stops GPU mining with an error.

Each nonce passing the BLAKE3 gate uses 1 MiB of device scratchpad and runs 32,768 fill steps followed by 32,768 dependent reads. The plugin processes finalists in groups capped at 512 MiB of scratchpad and one quarter of reported free VRAM. `--batch-size` limits the number of candidate nonces per search window. Older network profiles support CPU mining. To run the hardware regression tests, set `HYPHEN_TEST_ACCELERATOR_DIR` to the plugin directory and run `cargo test --release --locked native_neural_ -- --ignored --test-threads=1`.

For GeForce GTX 1060 (Pascal, compute capability 6.1), configure a fresh build with CUDA 12.x. The default targets include native `sm_61`, `sm_75`, `sm_86` code and `compute_61` PTX. With Ninja, pass `-DCMAKE_BUILD_TYPE=Release` at configure time. CUDA 13 dropped offline compilation for Pascal. On the target machine, run `hyphen-miner --accelerator-dir DIRECTORY accelerator-check --batch 32768 --timings` to compare every score and winning nonce with Rust. Compare `gpu_search_ns` and `cpu_search_ns` for the complete searches over the same nonce window.

Run `hyphen-miner --accelerator-dir DIRECTORY accelerator-check --offset 125 --batch 1` to reproduce the first-height HYN3 vector. The check compares every returned device score with Rust, applies the memory gate and prints `scores=1 accepted=1` for this nonce. [Reference vector](https://github.com/HyphenTeam/Hyphen/blob/main/test-vectors/neural-block-work-v3.json)

For an operator-provided SOCKS5 or HTTP CONNECT proxy use `--via-proxy socks5://HOST:PORT` or `--via-proxy http://HOST:PORT`. Set `--pool` to the destination pool address. Job changes invalidate outstanding work for that parent.

## 中文说明

首次运行使用 `keygen --output miner.key` 生成签名密钥，此后用 `--key-file miner.key` 复用该身份，用 `--wallet-address` 指定奖励地址。默认使用 CPU，可通过 `--threads` 限制线程数。即使程序旁边已有 `accelerators` 文件夹，也要显式传入 `--accelerator-dir DIRECTORY` 才会启用 GPU。HYN3 的神经评分和每个 nonce 的全部 BLAKE3 计算均在显卡完成，每个通过初筛的 nonce 使用 1 MiB 显存。插件缺失、版本不兼容或运行失败时，矿工报错退出。
