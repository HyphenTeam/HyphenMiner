# Hyphen Miner

Hyphen Miner 搜索 Hyphen 区块版本 5（PoUW v3）的有用工作量证明：单次使用的
Cunningham 素数链。它使用 Pool Protocol v6，并拒绝任何与已编译共识常量不一致的
矿池、链身份或区块版本。

见证搜索始终保留可移植的 CPU 实现，并可选使用原生 CUDA 或 HIP 小素数筛。加速器不决定
见证是否有效：它只返回通过筛法的候选，Rust 侧对每个候选执行确定性 Miller-Rabin 与本地
单次使用检查。

共识类型、哈希、密钥与 wire 编码在这里不再重复实现：hyphen-core、hyphen-crypto、
hyphen-pow、hyphen-neural 通过 `Cargo.toml` 与 `Cargo.lock` 钉定到公开 Hyphen revision
`c57f908464df6634eba829e8267f9c5330b3a4b4`。这里没有指向主链本地检出的路径依赖，CI
从干净克隆构建本仓库。共识变更通过升级该 revision 到达矿工，而不是复制代码。

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

加速器是可选的共享库。Rust 矿工始终保留可移植 CPU 路径；没有插件加载成功或设备调用失败时
会自动回退，不会因为 GPU 故障停止挖矿。

### Linux + NVIDIA GPU

先确认驱动和工具链：

```bash
nvidia-smi
nvcc --version
```

`nvidia-smi` 能看到显卡，说明驱动和 GPU 已就绪；`nvcc` 可用，说明构建插件所需的 CUDA
Toolkit 已安装。没有 GPU 直通的 KVM 虚拟机无法使用 CUDA，矿工会打印
`Native accelerator unavailable (...); using CPU witness search` 并继续走 CPU。

针对本机显卡构建插件：

```bash
cmake -S native -B native/build -DHYPHEN_ENABLE_CUDA=ON
cmake --build native/build --config Release -j"$(nproc)"
```

`HYPHEN_CUDA_ARCHITECTURES` 默认为 `native`。如果 CMake 无法识别设备，用

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

读取计算能力，去掉小数点后传入，例如 `-DHYPHEN_CUDA_ARCHITECTURES=86`。如果要分发给
其他显卡，显式设置架构列表，如 `75;80;86;89`。

插件输出到 `accelerators/hyphen_backend_cuda.so`。挖矿前先与 CPU 谓词比对：

```bash
./target/release/hyphen-miner \
  --accelerator-dir "$PWD/accelerators" \
  accelerator-check --batch 2000000 --offset 1000001 --stride 32 --length 6
```

正常输出包含后端、设备、GPU/CPU 幸存数和完整有效链数量。幸存集合不一致、返回窗口外的值
或插件 ABI 不匹配都会以非零状态退出；这种插件不能用于挖矿。确认无误后用同一目录启动矿工：

```bash
./target/release/hyphen-miner \
  --accelerator-dir "$PWD/accelerators" \
  --network devnet \
  --pool 127.0.0.1:3340 \
  --wallet-address hy1...
```

`--no-accelerator` 强制 CPU 路径；`HYPHEN_ACCELERATOR_DIR` 与 `--accelerator-dir`
等价。两者任一存在时即为权威目录，不再搜索默认路径；只有都未设置时才依次查找可执行文件旁、
当前目录和构建树中的 `accelerators/`。

### Windows 与 HIP

```powershell
cmake -S native -B native/build -DHYPHEN_ENABLE_CUDA=ON
cmake --build native/build --config Release
```

HIP 插件采用同样的布局：在 ROCm 主机上使用 `-DHYPHEN_ENABLE_HIP=ON` 构建。
`HYPHEN_HIP_ARCHITECTURES` 默认为空，由 hipcc 选择本机 GPU 架构。

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
