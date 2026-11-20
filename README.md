# Hyphen Miner

[简体中文](README_CN.md)

Hyphen Miner searches for verifiable work and submits signed shares to a Hyphen pool. The portable CPU backend supports the standard and neural devnets. CUDA and HIP plugins accelerate the Cunningham search. The CUDA plugin also scores neural work on `neural-devnet`, with Rust verification of each winning score.

## Mine on devnet

Start a devnet pool and obtain a devnet receiving address from Hyphen Wallet. Replace `WALLET_ADDRESS` with that address.

```sh
cargo run --release --locked -- --network devnet --pool 127.0.0.1:3340 --wallet-address WALLET_ADDRESS
```

The default SOLO mode at the pool pays the authorized address directly. [Operator guide](wiki/Operations.md) · [Hyphen Pool](https://github.com/HyphenTeam/HyphenPool)

## License

[PolyForm Strict License 1.0.0](LICENSE)
