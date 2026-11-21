# Hyphen Miner

[简体中文](README_CN.md) · [Miner documentation](https://github.com/HyphenTeam/HyphenMiner/wiki)

Hyphen Miner searches a nonce on the neural development network. CPUs and the CUDA plugin compute the same fixed-point neural score. The host applies BLAKE3 scratchpad and signs a successful block for the SOLO pool.

## Mine on devnet

Start a devnet pool and obtain a devnet receiving address from Hyphen Wallet. Replace `WALLET_ADDRESS` with that address.

```sh
cargo run --release --locked -- --network neural-devnet --pool 127.0.0.1:3340 --wallet-address WALLET_ADDRESS
```

The default SOLO mode at the pool pays the authorized address directly. [Operator guide](wiki/Operations.md) · [Hyphen Pool](https://github.com/HyphenTeam/HyphenPool)

## License

[PolyForm Strict License 1.0.0](LICENSE)
