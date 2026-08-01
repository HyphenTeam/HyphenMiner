# Hyphen Miner

[English](README.md)

Hyphen Miner 是 Hyphen 区块版本 3 的独立科学计算矿工。它使用 Pool Protocol v5，并拒绝旧协议、旧链身份和旧区块版本。

## 科学计算 PoUW

矿工、Pool 和节点执行相同的 64x64 Q12 二维扩散 PDE。区块难度是完整工作的精确迭代数；Pool 请求的较低迭代数只是贡献计量检查点，不是低哈希 share。

每个任务按以下顺序处理：

1. 校验 network magic、共识参数哈希、创世哈希、区块版本、高度、父块、epoch seed、奖励公钥、交易字节和交易根。
2. 从除 `pow_commitment` 外的完整区块头派生科学计算输入。
3. 按 Pool v5 要求提交较低迭代数的科学计算检查点。
4. 完成区块要求的全部迭代，承诺最终场，并签署完整区块头和奖励公钥。
5. 对每个接收结果校验 Pool 的签名回执链。

BLAKE3 只用于确定性输入派生、承诺、消息摘要和回执完整性。矿工不会把摘要与难度目标比较。

当前 PoUW v1 内核尚未绑定用户发布的 AetherCompute 科研任务，只能证明指定 PDE 已执行；它不能证明外部科学价值，也不提供简洁验证。遥测按每个单元每轮 7 次规范算术操作计数，不代表原生指令数或能耗。

固定跨实现向量：

```text
25078c250c5b44211bbf0fea60e90ac7024df6ff94d154161852fdd72684e524
```

## 原生加速器

原生 ABI v2 定义与共识一致的 64x64、四邻域、周期边界 Q12 内核。CUDA、HIP 和 OpenVINO 插件不把厂商 SDK 链接进 Rust 协议进程。设备结果只有通过独立 Rust 实现逐字节复核后才会用于挖矿；旧 ABI v1 插件会被拒绝。QNN 在目标设备图包实现 ABI v2 前保持 fail-closed。

没有通过自检的设备时默认使用 Rust CPU 内核。指定 `--require-accelerator` 后禁止回退：主线程和工作线程都必须成功加载插件，设备执行失败会停止挖矿，不会提交未经检查的结果。

构建已安装 SDK 的后端，例如 CUDA：

```powershell
cmake -S .\native -B .\native\build -DHYPHEN_ENABLE_CUDA=ON
cmake --build .\native\build --config Release
```

检查设备：

```powershell
.\target\release\hyphen-miner.exe --accelerator-dir .\accelerators accelerators
```

这项设备复核不是 SNARK/STARK，也不是用户 AetherCompute 任务的结算证明。缺少 SDK 或硬件时，后端只会报告 unavailable。

## 构建与运行

```powershell
cargo build --release --locked
cargo test --locked
cargo clippy --all-targets --locked -- -D warnings
.\target\release\hyphen-miner.exe keygen --output .\miner.key
```

连接 Pool：

```powershell
.\target\release\hyphen-miner.exe `
  --pool 127.0.0.1:3340 `
  --network devnet `
  --key-file .\miner.key `
  --wallet-address '<devnet-hy1-address>' `
  --threads 4 `
  --accelerator-dir .\accelerators
```

以当前二进制的 `--help` 为参数权威来源。`miner.key` 是矿工协议身份密钥，不是钱包恢复密钥。

## 兼容性与安全边界

- Pool protocol：5
- Accelerator ABI：2
- Block version：3
- PoUW protocol：1
- Devnet 共识参数：`54bf97e4e28d4fcf963d884a555a8425bbfe7c84d2753001bcabbaf116232fda`
- Devnet genesis：`47d530160cfef9141fe3b37b886e09b9f96ec4dc93d6c05005b9c6dbf35b1972`

RustBinary 使用有界字节数、有界集合长度、固定宽度小端编码，并拒绝尾随字节。共识迁移必须显式失败；承诺、签名和授权路径禁止格式回退。

本软件尚未经过独立共识、密码学或加速器审计，只应在研究网络使用。

## 许可证

Hyphen Miner 使用 PolyForm Strict License 1.0.0，完整条款见 [LICENSE](LICENSE)。
