# Hyphen Miner

[English](README.md)

Hyphen Miner 是独立的 CPU 矿工仓库。它连接 Hyphen Pool Protocol v3，验证链身份和工作模板，执行内存型 PoW，提交 share，并且只在结果同时满足区块难度时生成矿工出块授权。

## 当前真实能力

已经实现多线程挖矿、epoch arena 共享、断线退避重连、VarDiff、签名消息、奖励地址约束、交易根复算、完整区块授权和签名 share 回执链。Useful-Work 没有开启；没有 GPU 后端；矿工还不能自主从节点选交易并向池声明工作；池拒绝时也没有直接向节点提交的兜底。

因此当前准确结论是：矿工能检测池在发出 job 以后替换交易列表、奖励地址或 header，但 Pool v3 的交易集合仍由池选择，不能声称已经消灭交易审查和 MEV。

## 构建和运行

```powershell
cargo build --release --locked
cargo test --locked
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

## CI 和 Release

CI 执行格式、严格 Clippy、测试和锁定依赖的 release build。只有 `main` 上 CI 成功才触发自动 Release，分别构建 Linux、Windows、macOS 可执行文件，附提交号、工具链、调试信息和 SHA-256，并发布为 commit-bound prerelease。

`miner.key` 只能用于矿工协议身份，不能恢复钱包，也绝不能上传。

## 许可证

HyphenMiner 使用 GNU Affero General Public License v3.0，完整条款见 [LICENSE](LICENSE)。
