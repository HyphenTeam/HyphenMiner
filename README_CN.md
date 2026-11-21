# Hyphen Miner

[English](README.md) · [矿工运行指南](https://github.com/HyphenTeam/HyphenMiner/wiki/Operations#中文说明)

Hyphen Miner 默认使用 CPU 搜索 HYN3 nonce。显式传入 `--accelerator-dir 插件目录` 后，CUDA 会生成任务种子与权重，完成神经评分、BLAKE3 初筛、工作内存填充与访问、最终阈值判断，并计算获胜 nonce 的工作承诺。主机负责签名和矿池通信。

## 在开发网挖矿

先启动开发网矿池，再从 Hyphen Wallet 获取开发网收款地址。将 `WALLET_ADDRESS` 替换为该地址。

```sh
cargo run --release --locked -- --network neural-devnet --pool 127.0.0.1:3340 --wallet-address WALLET_ADDRESS
```

矿池默认采用 SOLO 模式，将出块奖励直接支付到矿工授权的地址。[运行指南](wiki/Operations.md) · [Hyphen Pool](https://github.com/HyphenTeam/HyphenPool)

## 许可证

[PolyForm Strict License 1.0.0](LICENSE)
