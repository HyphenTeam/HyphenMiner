# HyphenMiner

HyphenMiner connects CPU and GPU workers to a Hyphen mining pool. The miner signs in with a persistent identity, verifies the network parameters for each job and checks the reward address before submitting a block candidate.

- [Identity and reward address](Operations.md#identity-and-reward-address) covers signing keys, SOLO rewards and authorization for shared payouts.
- [CPU, plugins and connections](Operations.md#cpu-plugins-and-connections) covers worker limits, CUDA and HIP verification, proxy settings and reconnection.

## 中文指南

矿工使用签名密钥登录矿池，核验任务所属网络，并在提交候选区块前检查奖励地址。[矿工运行指南](Operations.md#中文说明)说明如何配置身份密钥与收款地址、验证 GPU 插件，以及在连接中断后恢复挖矿。
