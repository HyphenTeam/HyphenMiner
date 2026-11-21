# Hyphen Miner

[English](README.md) · [矿工运行指南](https://github.com/HyphenTeam/HyphenMiner/wiki/Operations#中文说明)

Hyphen Miner 在神经开发网上搜索 nonce。CPU 与 CUDA 插件计算同一定点神经评分，主机执行 BLAKE3 scratchpad，并向 SOLO 矿池提交签名候选区块。

## 在开发网挖矿

先启动开发网矿池，再从 Hyphen Wallet 获取开发网收款地址。将 `WALLET_ADDRESS` 替换为该地址。

```sh
cargo run --release --locked -- --network neural-devnet --pool 127.0.0.1:3340 --wallet-address WALLET_ADDRESS
```

矿池默认采用 SOLO 模式，将出块奖励直接支付到矿工授权的地址。[运行指南](wiki/Operations.md) · [Hyphen Pool](https://github.com/HyphenTeam/HyphenPool)

## 许可证

[PolyForm Strict License 1.0.0](LICENSE)
