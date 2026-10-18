# Hyphen Miner

Hyphen Miner 搜索 Hyphen 区块版本 5（PoUW v3）的有用工作量证明：单次使用的
Cunningham 素数链。它使用 Pool Protocol v6，并拒绝任何与已编译共识常量不一致的
矿池、链身份或区块版本。

见证搜索始终保留可移植的 CPU 实现，并可选使用原生 CUDA 或 HIP 小素数筛。加速器不决定
见证是否有效：它只返回通过筛法的候选，Rust 侧对每个候选执行确定性 Miller-Rabin 与本地
单次使用检查。

共识类型、哈希、密钥与 wire 编码在这里不再重复实现：矿工通过同级的 `crates/` 目录
构建 `HyphenTeam/Hyphen` 的共识实现。CI 与发布工作流会先把 `../Cargo.toml` 和
`../crates` 物化到同级目录，再开始构建。这样矿工和共识实现处于同一测试面，而不是各自
维护一份可能漂移的快照；基础链源码不会被复制进本仓库。

## 矿工在算什么

见证（witness）是一个奇数 n >= 3，其 Cunningham 链 n, 2n+1, 4n+3, ... 在
pouw_chain_length 个元素内全为素数。链长由网络固定并随 job 下发，网络不对会被直接
拒绝，而不是产出节点无法使用的工作量。

每个区块需要 difficulty 个见证（1..=4096）。每轮矿工：

1. 只扫描自己的剩余类（stride = 2 * 线程数，每线程一个奇数偏移），线程之间不重叠、
   不需要协调。
2. 凑够 share_difficulty 个见证后立即提交一个 credit 份额，然后继续扫描直到列表完整。
3. 将列表升序排序，计算
   pow_commitment = witness_commitment(parent_commitment, height, witnesses)，
   用矿工私钥签区块头并提交。

见证是一次性的。节点把所有已认领见证写入稀疏 Merkle 集合，因此预计算池只能花掉一次，
预计算退化为提前挖矿、零优势。矿工本地也记住自己已提交过的见证：这份记忆不是共识状态，
只是避免自己为同一个见证重复付出。矿池与节点各自独立拒绝重复。

## 文件

- src/main.rs — 连接、job 处理、见证搜索、份额提交
- src/accelerator.rs — 原生插件 ABI 加载器与 CPU 回退
- src/protocol.rs — Pool Protocol v6 信封与消息
- src/primitives.rs — canonical 再导出，这里没有任何共识实现
- native/ — CUDA 与 HIP Cunningham 筛插件（ABI v2，kernel 3）

原生插件只实现候选窗口上的小素数筛，是可选共享库，从 accelerators/ 或
HYPHEN_ACCELERATOR_DIR 加载。插件缺失、设备不受支持或设备报错都会把工作线程切回 CPU
路径；验证强度不会降低。

## 构建与运行

```powershell
cargo build --release --locked
cargo test --locked
cargo clippy --all-targets --locked -- -D warnings

.\target\release\hyphen-miner.exe --network devnet --pool 127.0.0.1:3340 `
    --wallet-address hy1...
```

常用参数：--threads、--batch-size（每次搜索的候选数）、
--allow-shared-reward-recipient（仅在审阅矿池策略后使用）、--no-accelerator（强制 CPU）、
--accelerator-dir（插件目录）。

## 原生加速

在装有 CUDA Toolkit 的机器上构建 CUDA 插件：

```powershell
cmake -S native -B native/build -DHYPHEN_ENABLE_CUDA=ON
cmake --build native/build --config Release
```

在装有 ROCm/HIP 的机器上构建 HIP 插件：

```powershell
cmake -S native -B native/build -DHYPHEN_ENABLE_HIP=ON
cmake --build native/build --config Release
```

插件输出到 accelerators/。也可以用 --accelerator-dir 或 HYPHEN_ACCELERATOR_DIR 指定
目录。HYPHEN_CUDA_ARCHITECTURES 默认是 native；HYPHEN_HIP_ARCHITECTURES 默认为空，由
hipcc 选择。如果要分发到其他 GPU，请显式设置架构列表。

挖矿前先用 CPU 谓词校验设备筛：

```powershell
cargo run --release -- accelerator-check --batch 2000000 --offset 1000001 `
    --stride 32 --length 6
```

GPU 幸存集合与 CPU 小素数筛不一致、或插件返回窗口外的值时，命令会失败。输出包含设备
时间、墙钟时间、幸存数与完整有效链数量。

平台行为：

| 平台 | 后端 | 回退 |
|---|---|---|
| Linux x86_64/aarch64 | CUDA 或 HIP 插件 | CPU |
| Windows x86_64 | CUDA 或 HIP 插件 | CPU |
| macOS | 不构建 | CPU |
| Android arm64/armv7 | 不构建 | CPU |
| RISC-V | 不构建 | CPU |

CPU 搜索是所有目标上的规范路径。macOS Metal、Android Adreno 与其他 NPU/DSP 后端尚未接入
本二进制；这里不对它们做加速承诺。

## 协议

- 区块版本：5
- PoUW 协议：3
- 矿池协议：6

## 局限

本软件尚未经过独立的共识与密码学审计。请仅在研究网络使用。
