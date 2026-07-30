# Hyphen Miner

[中文说明](README_CN.md)

Hyphen Miner is the standalone CPU miner for the Hyphen base chain. It speaks
Pool Protocol v3, checks chain identity and job commitments, evaluates the
memory-hard PoW, submits shares, and signs a full-block authorization only when
the candidate also meets the network target.

This is an independent repository. It does not contain the node or pool and is
not built by the Hyphen base-chain workspace.

## Current boundary

- Implemented: multi-threaded CPU mining, shared epoch arena, reconnect with
  bounded backoff, VarDiff updates, signed envelopes, reward-address checks,
  transaction-root verification, full-block miner authorization, and signed
  hash-chained share receipt verification.
- Not implemented: Useful-Work, GPU mining, miner-originated transaction-set
  declaration, direct node submission when a pool refuses a job, and complete
  censorship/MEV resistance.
- Pool v3 lets the miner detect mutation of the received transaction list or
  reward keys. The pool still selects the list in this version.

## Build

```bash
cargo build --release --locked
cargo test --locked
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
