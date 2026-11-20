# Hyphen Miner

[English](README.md)

Hyphen Miner 搜索可验证的工作量，并向 Hyphen 矿池提交签名份额。CPU 后端支持标准开发网和神经工作量开发网。在支持的硬件上，CUDA 或 HIP 插件可以加速坎宁安素数链搜索。

## 在开发网挖矿

先启动开发网矿池，再从 Hyphen Wallet 获取开发网收款地址。将 `WALLET_ADDRESS` 替换为该地址。

```sh
cargo run --release --locked -- --network devnet --pool 127.0.0.1:3340 --wallet-address WALLET_ADDRESS
```

矿池默认采用 SOLO 模式，将出块奖励直接支付到矿工授权的地址。[运行指南](wiki/Operations.md) · [Hyphen Pool](https://github.com/HyphenTeam/HyphenPool)

## 许可证

[PolyForm Strict License 1.0.0](LICENSE)
