# Hyphen Miner

[中文说明](README_CN.md)

Hyphen Miner is the standalone miner and native scientific accelerator host for
the Hyphen base chain. It speaks Pool Protocol v3, validates PoW jobs, and loads
versioned CUDA, HIP, OpenVINO and QNN plugins without linking vendor SDKs into
the consensus-facing Rust process.

This is an independent repository. It does not contain the node or pool and is
not built by the Hyphen base-chain workspace.

## Current boundary

- Implemented: multi-threaded CPU mining, shared epoch arena, reconnect with
  bounded backoff, VarDiff updates, signed envelopes, reward-address checks,
  transaction-root verification, full-block miner authorization, and signed
  hash-chained share receipt verification. The native accelerator ABI executes
  a bounded deterministic Q12 diffusion/PDE kernel and verifies every device
  result against an independent Rust implementation before returning it.
- NVIDIA CUDA is built and hardware-tested on this development host. AMD HIP
  and Intel OpenVINO NPU/GPU are conditional native builds and remain
  `unavailable` when their SDK or hardware is absent. The QNN plugin performs
  real HTP provider/device initialization but advertises no compute capability
  until a target-SoC deterministic graph package is installed.
- Not implemented: Pool Protocol scientific-job transport, miner submission of
  `hyphen-compute` result envelopes, production scientific circuits/verifier
  keys, direct node fallback, and complete censorship/MEV resistance.
- Pool v3 lets the miner detect mutation of the received transaction list or
  reward keys. The pool still selects the list in this version.

## Build

```bash
cargo build --release --locked
cargo test --locked
cargo clippy --all-targets --locked -- -D warnings
```

Generate a persistent miner identity:

```bash
./target/release/hyphen-miner keygen --output ./miner.key
```

Mine on devnet:

```bash
./target/release/hyphen-miner \
  --network devnet \
  --pool 127.0.0.1:3340 \
  --key-file ./miner.key \
  --wallet-address '<network-correct-hy1-address>' \
  --threads 0 \
  --batch-size 100000
```

`--threads 0` uses all logical CPUs. An ephemeral identity is allowed for local
devnet testing; mainnet research mode requires `--key-file`. A shared pool may
redirect coinbase to its settlement wallet only when the miner explicitly adds
`--allow-shared-reward-recipient`.

## Native scientific accelerators

Build only backends whose vendor SDK is installed. Plugins are emitted into
`accelerators/` and are loaded by canonical absolute path at runtime.

```powershell
cmake -S native -B native/build-cuda -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DHYPHEN_ENABLE_CUDA=ON
cmake --build native/build-cuda --config Release

cmake -S native -B native/build-openvino -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DHYPHEN_ENABLE_OPENVINO=ON `
  -DOpenVINO_DIR='<openvino-runtime>/cmake'

cmake -S native -B native/build-hip -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DHYPHEN_ENABLE_HIP=ON

cmake -S native -B native/build-qnn -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DHYPHEN_ENABLE_QNN=ON `
  -DQNN_SDK_ROOT='D:/Qualcomm AI Engine Direct SDK'
```

Enumerate devices and run device/CPU known-answer tests:

```powershell
./target/release/hyphen-miner --accelerator-dir ./accelerators accelerators
```

`available`, `unavailable`, and `self-test failed` are distinct states.
`--require-accelerator` makes startup fail closed unless at least one device
passes actual device execution, exact output comparison, and operation-count
validation.

Execute and fully cross-check a scientific kernel. The input and output are
little-endian `i32` arrays; v1 accepts at least three cells in `0..=262143`.

```powershell
./target/release/hyphen-miner `
  --accelerator-dir ./accelerators `
  scientific-run `
  --backend nvidia-cuda `
  --input ./cells.i32le `
  --output ./evolved.i32le `
  --alpha-q12 512 `
  --iterations 64
```

The command prints domain-separated input/output commitments, stable device ID,
operation count and device time only after the entire output matches Rust. This
is deterministic execution verification, not a SNARK and not a chain settlement
proof.

## Job verification

Before hashing, the miner checks protocol version, network magic, consensus
parameter hash, genesis hash, block version, height, difficulty, epoch seed,
miner public key, reward keys, arena limits, transaction list, and committed
transaction root. A mismatched field rejects the job.

For a full block, authorization signs a digest that binds the header and reward
keys. This prevents a pool from taking one solved header and silently replacing
the committed transaction root or destination. It does not prevent the pool
from omitting a transaction before constructing the job.

## Mathematics

Interpret a PoW hash as a uniformly distributed integer `X` in
`[0, 2^256-1]`. For difficulty `D >= 1`, the implementation uses

```text
T(D) = floor((2^256 - 1) / D)
```

and accepts `X <= T(D)`. Therefore

```text
p_D = (T(D) + 1) / 2^256 ~= 1 / D.
```

After `n` independent trials, the probability of at least one accepted share is

```text
Pr[success] = 1 - (1 - p_D)^n.
```

At hashrate `h`, the expected time is `E[t] = 1/(h p_D)`, approximately
`D/h`. These are probabilistic expectations; short runs can deviate greatly.

A pool share uses `D_share`, while a block uses `D_block`. Normally
`D_share <= D_block`. Every block solution is a share, but most shares are not
blocks. A share is accounting evidence and has no on-chain value by itself.

Accepted-share receipts form

```text
r_i = H_d(pool_pk, miner_pk, i, r_(i-1), H(submission_i), H(result_i)).
```

The pool signs the envelope containing `r_i`. Assuming hash collision
resistance and EUF-CMA signature security, changing a historical submission,
result, sequence, or predecessor either changes every later receipt or requires
a hash collision/signature forgery. This gives tamper evidence, not public data
availability or proof that the pool is solvent.

## Protocol compatibility

Consensus-facing local types must remain synchronized with the main chain.
Binary encoding uses RustBinary 0.1.2 with the same bounded, fixed-width,
little-endian, trailing-byte-rejecting profile as Hyphen. A protocol change
requires the fixed wire vector, all three chain-identity vectors, tests, strict
Clippy and a release build to pass. Do not add format fallback to hash,
signature or authorization paths.

## CI and releases

CI checks formatting, strict Clippy, tests, and a locked release build. A
successful CI run on `main` triggers a separate least-privilege release workflow
that builds Linux, Windows, and macOS executables, includes build/debug metadata,
publishes SHA-256 files, and creates a commit-bound prerelease.

Do not publish `miner.key`. It authenticates mining messages; it is not a wallet
recovery key.

## License

HyphenMiner is licensed under the GNU Affero General Public License v3.0. See
[LICENSE](LICENSE) for the complete terms.

---

<!-- hyphen-bilingual-chinese -->

# 中文版

# Hyphen Miner

[English](README.md)

Hyphen Miner 是独立矿工和原生科学加速器宿主。它连接 Hyphen Pool Protocol v3、验证 PoW 工作模板，并通过版本化 C ABI 加载 CUDA、HIP、OpenVINO 和 QNN 插件，厂商 SDK 不进入面向共识的 Rust 进程。

## 当前真实能力

已经实现多线程挖矿、epoch arena 共享、断线退避重连、VarDiff、签名消息、奖励地址约束、交易根复算、完整区块授权和签名 share 回执链。原生加速器 ABI 已实现有界、确定性的 Q12 扩散/PDE 算子；每次设备输出返回前都会由独立 Rust 实现完整复算。

NVIDIA CUDA 已在当前开发机实机验证。AMD HIP 与 Intel OpenVINO NPU/GPU 是条件构建后端；缺 SDK 或硬件时明确显示 `unavailable`。QNN 插件会真实初始化 HTP provider/device，但在目标 SoC 的确定性 graph package 未安装前不公布计算能力，不能进入调度。

Pool v3 仍未实现科研任务分发，矿工也尚未提交 `hyphen-compute` 结果 envelope；生产科研 circuit/verifier key 仍不存在。因此本地设备正确执行不是 SNARK，也不是已完成的链上 PoUW 结算。

因此当前准确结论是：矿工能检测池在发出 job 以后替换交易列表、奖励地址或 header，但 Pool v3 的交易集合仍由池选择，不能声称已经消灭交易审查和 MEV。

## 构建和运行

```powershell
cargo build --release --locked
cargo test --locked
cargo clippy --all-targets --locked -- -D warnings
.\target\release\hyphen-miner.exe keygen --output .\miner.key
```

```powershell
.\target\release\hyphen-miner.exe `
  --network devnet `
  --pool 127.0.0.1:3340 `
  --key-file .\miner.key `
  --wallet-address '<与网络匹配的 hy1 地址>' `
  --threads 0 `
  --batch-size 100000
```

`--threads 0` 使用全部逻辑 CPU。mainnet 研究配置拒绝临时身份。只有在你明确接受共享池把 coinbase 指向结算钱包时，才使用 `--allow-shared-reward-recipient`。

## 原生科学加速器

只构建本机已经安装 SDK 的后端，插件输出到 `accelerators/`：

```powershell
cmake -S native -B native/build-cuda -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DHYPHEN_ENABLE_CUDA=ON
cmake --build native/build-cuda --config Release

cmake -S native -B native/build-openvino -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DHYPHEN_ENABLE_OPENVINO=ON `
  -DOpenVINO_DIR='<openvino-runtime>/cmake'

cmake -S native -B native/build-hip -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DHYPHEN_ENABLE_HIP=ON

cmake -S native -B native/build-qnn -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DHYPHEN_ENABLE_QNN=ON `
  -DQNN_SDK_ROOT='D:/Qualcomm AI Engine Direct SDK'
```

枚举设备并执行设备端/CPU 已知答案交叉验证：

```powershell
.\target\release\hyphen-miner.exe `
  --accelerator-dir .\accelerators `
  accelerators
```

`available`、`unavailable`、`self-test failed` 是三个不同状态。加入 `--require-accelerator` 后，没有设备通过真实设备执行、逐字节复算和操作数检查时，Miner 会拒绝启动。

执行并完整复核科研算子。输入和输出均为小端 `i32` 数组，v1 至少需要三个 `0..=262143` 的单元：

```powershell
.\target\release\hyphen-miner.exe `
  --accelerator-dir .\accelerators `
  scientific-run `
  --backend nvidia-cuda `
  --input .\cells.i32le `
  --output .\evolved.i32le `
  --alpha-q12 512 `
  --iterations 64
```

只有完整输出与 Rust 一致后，命令才打印域分离的输入/输出 commitment、稳定设备 ID、操作数和设备耗时。这是确定性执行验证，不是 ZK/SNARK 证明。

## 模板验证

矿工开始计算前会检查协议版本、network magic、共识参数哈希、genesis hash、区块版本、高度、难度、epoch seed、矿工公钥、奖励公钥、arena 上限、交易字节和交易根。任一字段不一致都会拒绝 job。找到完整区块后，矿工签署的授权把 header 与奖励公钥绑定，池不能拿同一个解去替换交易根或收款人。

## PoW 与回执链的数学含义

将 256-bit PoW 输出看成在 `[0,2^256-1]` 上均匀分布的整数 `X`。难度 `D>=1` 对应：

```text
T(D)=floor((2^256-1)/D),    接受条件 X<=T(D).
```

单次成功概率为：

```text
p_D=(T(D)+1)/2^256 ~= 1/D.
```

独立尝试 `n` 次至少成功一次的概率为 `1-(1-p_D)^n`；哈希率为 `h` 时，期望等待时间为 `1/(h p_D)`，近似 `D/h`。这是期望，不是某段时间内必然找到。

Share 使用较低的 `D_share`，区块使用 `D_block`。满足区块难度的结果一定满足正常配置下的 share 难度，但普通 share 不是区块、不是链上余额，只是结算贡献证据。

第 `i` 个接受回执为：

```text
r_i=H_d(pool_pk,miner_pk,i,r_(i-1),H(submission_i),H(result_i)).
```

池对含 `r_i` 的 envelope 签名。若修改旧 submission、result、序号或前驱，后续回执都会改变；想保持链头不变需要哈希碰撞或签名伪造。这个性质是防篡改证据，不是公开账本可用性，也不能证明矿池有钱付款。

## 协议兼容性

面向共识的本地类型必须与 Hyphen 主链保持同步。二进制编码使用 RustBinary 0.1.2，
配置与主链相同：有界、固定宽度、小端并拒绝尾随字节。协议变更必须重新通过 wire
固定向量、三套链身份向量、测试、严格 Clippy 和 release 构建。不得在哈希、签名或
授权路径中加入格式回退解码。

## CI 和 Release

CI 执行格式、严格 Clippy、测试和锁定依赖的 release build。只有 `main` 上 CI 成功才触发自动 Release，分别构建 Linux、Windows、macOS 可执行文件，附提交号、工具链、调试信息和 SHA-256，并发布为 commit-bound prerelease。

`miner.key` 只能用于矿工协议身份，不能恢复钱包，也绝不能上传。

## 许可证

HyphenMiner 使用 PolyForm Strict License 1.0.0，完整条款见 [LICENSE](LICENSE)。
