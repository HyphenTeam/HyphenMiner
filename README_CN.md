# Hyphen Miner

Hyphen Miner 搜索 Hyphen 区块版本 5（PoUW v3）的有用工作量证明：单次使用的
Cunningham 素数链。它使用 Pool Protocol v6，并拒绝任何与已编译共识常量不一致的
矿池、链身份或区块版本。

共识类型、哈希、密钥与 wire 编码在这里不再重复实现：hyphen-core、hyphen-crypto、
hyphen-pow 通过 `Cargo.toml` 与 `Cargo.lock` 钉定的公开 Hyphen revision
`576bc67d972d1156fbabb1ddf579104d5ff2ff88` 引入，共识改动通过升级该 revision 到达矿工，
而不是被抄一份然后被遗忘。这里没有指向主链本地检出的路径依赖：CI 从干净克隆构建本仓库。

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
- src/protocol.rs — Pool Protocol v6 信封与消息
- src/primitives.rs — canonical 再导出，这里没有任何共识实现
- native/ — 已退役的 v2 扩散场内核，仅保留 C++/CUDA 源码

没有任何代码路径会加载 native/。它实现的 v2 扩散场已不在共识中，因此矿工不做任何设备
加速声明；源码保留供未来的筛法加速搜索使用。

## 构建与运行

```powershell
cargo build --release --locked
cargo test --locked
cargo clippy --all-targets --locked -- -D warnings

.\target\release\hyphen-miner.exe --network devnet --pool 127.0.0.1:3340 `
    --wallet-address hy1...
```

常用参数：--threads、--batch-size（每次搜索的候选数）、
--allow-shared-reward-recipient（仅在审阅矿池策略后使用）。

## 协议

- 区块版本：5
- PoUW 协议：3
- 矿池协议：6

## 局限

本软件尚未经过独立的共识与密码学审计。请仅在研究网络使用。