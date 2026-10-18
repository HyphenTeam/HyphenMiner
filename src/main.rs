mod accelerator;
mod primitives;
mod protocol;
mod proxy;

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use accelerator::Accelerator;
use clap::{Parser, Subcommand};
use parking_lot::RwLock;
use prost::Message;
use tokio::net::TcpStream;
use tracing::{error, info, warn};

use primitives::{
    hex_encode, wire_config, BlockAuthorization, BlockHeader, ChainConfig, ChainIdentity, Hash256,
    SecretKey, FROZEN_BLOCK_VERSION,
};
use protocol::*;

#[derive(Parser, Debug)]
#[command(
    name = "hyphen-miner",
    about = "Hyphen standalone miner with verified native accelerators"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    #[arg(long, default_value = "127.0.0.1:3340")]
    pool: String,

    /// Mine through an operator-provided proxy.
    ///
    /// Format: `socks5://[user:pass@]host:port` or `http://[user:pass@]host:port`.
    /// The miner performs the SOCKS5 (RFC 1928) or HTTP `CONNECT` handshake with
    /// the operator's endpoint and then speaks the pool protocol through it
    /// unchanged. Whatever obfuscation or egress policy sits behind that
    /// endpoint is the operator's, not this binary's.
    #[arg(long = "via-proxy", value_name = "SPEC")]
    via_proxy: Option<String>,

    #[arg(long, default_value = "0")]
    threads: usize,

    #[arg(
        long,
        value_parser = ["mainnet", "testnet", "devnet", "neural-devnet"],
        default_value = "devnet"
    )]
    network: String,

    #[arg(long, default_value = "")]
    key_file: String,

    #[arg(long, default_value = "hyphen-miner/0.1")]
    user_agent: String,

    #[arg(long, default_value = "100000")]
    batch_size: u64,

    /// Wallet address for receiving mining rewards.
    /// Accepts a hy1... BIP44 address or a 64-char hex public key.
    /// If not specified, the miner's signing key is used as the payout key.
    #[arg(long, default_value = "")]
    wallet_address: String,

    /// Explicitly permits a shared pool to direct coinbase to its advertised
    /// settlement wallet. The exact reward keys remain miner-authorized.
    #[arg(long, default_value_t = false)]
    allow_shared_reward_recipient: bool,

    /// Disable native accelerator plugins and use the portable CPU search.
    #[arg(long, default_value_t = false)]
    no_accelerator: bool,

    /// Directory containing hyphen_backend_cuda / hyphen_backend_hip plugins.
    #[arg(long, default_value = "", global = true)]
    accelerator_dir: String,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Generate a new Ed25519 key pair and save to file
    Keygen {
        /// Output file path for the 32-byte secret key
        #[arg(long, default_value = "miner.key")]
        output: String,
    },
    /// Compare the native Cunningham sieve with the portable CPU sieve
    AcceleratorCheck {
        /// Candidates scanned in one call
        #[arg(long, default_value = "1000000")]
        batch: u32,
        /// First candidate (must be odd and above the small-prime bound)
        #[arg(long, default_value = "4101")]
        offset: u64,
        /// Distance between candidates
        #[arg(long, default_value = "2")]
        stride: u64,
        /// Cunningham chain length
        #[arg(long, default_value = "6")]
        length: u32,
    },
}

struct MiningJob {
    job_id: [u8; 32],
    header: BlockHeader,
    chain_length: u32,
    parent_work_commitment: Hash256,
    reward_view_public: [u8; 32],
    reward_spend_public: [u8; 32],
}

struct MinerState {
    current_job: RwLock<Option<Arc<MiningJob>>>,
    job_generation: AtomicU64,
    current_share_difficulty: AtomicU64,
    difficulty_generation: AtomicU64,
    estimated_compute_rate: AtomicU64,
    total_operations: AtomicU64,
    shares_accepted: AtomicU64,
    shares_rejected: AtomicU64,
    blocks_found: AtomicU64,
    running: AtomicBool,
    /// Witnesses this miner has already handed to the pool. The pool enforces
    /// single use chain-wide; locally we only avoid paying for our own
    /// duplicates, which lets a worker keep scanning instead of restarting.
    submitted_witnesses: RwLock<std::collections::HashSet<u64>>,
    /// Incremented on each new connection so stale mining threads exit.
    connection_generation: AtomicU64,
    receipt_sequence: AtomicU64,
    receipt_head: RwLock<[u8; 32]>,
    pending_submission_hashes: RwLock<VecDeque<[u8; 32]>>,
    /// Job generation for which a neural solution was already submitted.
    ///
    /// Every worker searches the same job concurrently; without this the
    /// second one to finish burns a full solution on a submission the pool
    /// classifies as a stale job.
    neural_solution_generation: AtomicU64,
    /// Native sieve plugin, when one was found and accepted the ABI.
    accelerator: Option<Accelerator>,
    /// Set after the first device error so every worker falls back to CPU.
    accelerator_failed: AtomicBool,
}

impl MinerState {
    #[cfg(test)]
    fn new() -> Self {
        Self::with_accelerator(None)
    }

    fn with_accelerator(accelerator: Option<Accelerator>) -> Self {
        Self {
            current_job: RwLock::new(None),
            job_generation: AtomicU64::new(0),
            current_share_difficulty: AtomicU64::new(100),
            difficulty_generation: AtomicU64::new(0),
            estimated_compute_rate: AtomicU64::new(0),
            total_operations: AtomicU64::new(0),
            shares_accepted: AtomicU64::new(0),
            shares_rejected: AtomicU64::new(0),
            blocks_found: AtomicU64::new(0),
            running: AtomicBool::new(true),
            submitted_witnesses: RwLock::new(std::collections::HashSet::new()),
            connection_generation: AtomicU64::new(0),
            receipt_sequence: AtomicU64::new(0),
            receipt_head: RwLock::new([0u8; 32]),
            pending_submission_hashes: RwLock::new(VecDeque::new()),
            neural_solution_generation: AtomicU64::new(u64::MAX),
            accelerator,
            accelerator_failed: AtomicBool::new(false),
        }
    }

    fn is_locally_claimed(&self, witness: u64) -> bool {
        self.submitted_witnesses.read().contains(&witness)
    }

    fn mark_submitted(&self, witnesses: &[u64]) {
        let mut claimed = self.submitted_witnesses.write();
        for witness in witnesses {
            claimed.insert(*witness);
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();

    if let Some(Commands::Keygen { output }) = &cli.command {
        let sk = SecretKey::generate();
        std::fs::write(output, sk.0)?;
        println!("Key generated successfully:");
        println!("  Secret key file : {output}");
        println!("  Public key (hex): {}", sk.public_key());
        println!("\nKeep the secret key file safe. Never share it.");
        return Ok(());
    }

    if let Some(Commands::AcceleratorCheck {
        batch,
        offset,
        stride,
        length,
    }) = &cli.command
    {
        let directory =
            (!cli.accelerator_dir.is_empty()).then(|| PathBuf::from(&cli.accelerator_dir));
        return run_accelerator_check(*batch, *offset, *stride, *length, directory.as_deref())
            .map_err(Into::into);
    }

    let cfg = match cli.network.as_str() {
        "mainnet" => ChainConfig::mainnet(),
        "testnet" => ChainConfig::testnet(),
        "devnet" => ChainConfig::devnet(),
        "neural-devnet" => ChainConfig::neural_devnet(),
        _ => unreachable!("clap validates network"),
    };

    if cli.network == "mainnet" && cli.key_file.is_empty() {
        return Err(
            "mainnet mining requires --key-file; ephemeral miner identities are unsafe".into(),
        );
    }
    if cli.batch_size == 0 {
        return Err("--batch-size must be greater than zero".into());
    }

    let threads = if cli.threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        cli.threads
    };

    let sk = if cli.key_file.is_empty() {
        let sk = SecretKey::generate();
        info!("Generated ephemeral miner key: {}", sk.public_key());
        sk
    } else {
        let data = std::fs::read(&cli.key_file)?;
        if data.len() != 32 {
            return Err("key file must be exactly 32 bytes".into());
        }
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&data);
        SecretKey(bytes)
    };

    info!(
        "Hyphen Miner starting: pool={}, threads={}, network={}",
        cli.pool, threads, cfg.network_name
    );

    let accelerator = if cli.no_accelerator {
        None
    } else {
        let directory =
            (!cli.accelerator_dir.is_empty()).then(|| PathBuf::from(&cli.accelerator_dir));
        match Accelerator::load(directory.as_deref()) {
            Ok(accelerator) => {
                info!(
                    "Native accelerator: {} on {} via {} ({} candidate window)",
                    accelerator.backend(),
                    accelerator.device_name(),
                    accelerator.path().display(),
                    accelerator.max_candidates(),
                );
                Some(accelerator)
            }
            Err(error) => {
                info!("Native accelerator unavailable ({error}); using CPU witness search");
                None
            }
        }
    };

    let state = Arc::new(MinerState::with_accelerator(accelerator));

    // The pool link is dialled directly, or through the operator's proxy when
    // one was named. Either way the pool protocol above this point is
    // untouched: a proxy carries a stream, it is not a second protocol.
    let proxy = match cli.via_proxy.as_deref() {
        Some(spec) => {
            let proxy = proxy::PoolProxy::new(proxy::ProxySpec::parse(spec)?)?;
            info!("Pool link routed through {}", proxy.routing_note(&cli.pool));
            Some(proxy)
        }
        None => None,
    };

    let wallet_pubkey = parse_wallet_address(&cli.wallet_address, &cfg)?;
    info!("Payout wallet: {}", hex_encode(wallet_pubkey));

    let mut backoff_secs = 5u64;
    loop {
        if !state.running.load(Ordering::Relaxed) {
            break;
        }

        let conn_start = std::time::Instant::now();

        match connect_and_mine(
            &cli.pool,
            proxy.as_ref(),
            &sk,
            &cli.user_agent,
            &cfg,
            threads,
            &state,
            cli.batch_size,
            &wallet_pubkey,
            cli.allow_shared_reward_recipient,
        )
        .await
        {
            Ok(()) => break,
            Err(e) => {
                error!("Connection lost: {e}");
                // Reset backoff if we were connected for a while
                if conn_start.elapsed().as_secs() > 60 {
                    backoff_secs = 5;
                }
                info!("Reconnecting in {backoff_secs} seconds...");
                tokio::time::sleep(std::time::Duration::from_secs(backoff_secs)).await;
                backoff_secs = (backoff_secs * 2).min(120);
            }
        }
    }

    // Nothing outlives the mining loop: a stopped miner leaves no listener and
    // no proxy thread bound on the operator's machine, which is exactly why the
    // proxy is a plain dialler rather than an embedded engine.
    Ok(())
}

fn parse_wallet_address(address: &str, cfg: &ChainConfig) -> Result<[u8; 64], String> {
    let encoded = address.strip_prefix("hy1").ok_or_else(|| {
        format!(
            "pool protocol v{POOL_PROTOCOL_VERSION} requires --wallet-address with a hy1... address"
        )
    })?;
    let payload = hyphen_crypto::base58::decode(encoded)
        .map_err(|_| "wallet address contains invalid base58".to_string())?;
    if payload.len() != 69 {
        return Err(format!(
            "invalid wallet address length: expected 69 decoded bytes, got {}",
            payload.len()
        ));
    }
    let expected_version = if cfg.network_magic == [0x48, 0x59, 0x50, 0x4e] {
        0x01
    } else {
        0x02
    };
    if payload[0] != expected_version {
        return Err("wallet address belongs to a different Hyphen network".into());
    }
    let hash = hyphen_crypto::blake3_hash(&payload[..65]);
    if payload[65..69] != hash.as_bytes()[..4] {
        return Err("wallet address checksum mismatch".into());
    }
    payload[1..65]
        .try_into()
        .map_err(|_| "wallet address payload is malformed".to_string())
}

#[allow(clippy::too_many_arguments)]
async fn connect_and_mine(
    pool_addr: &str,
    proxy: Option<&proxy::PoolProxy>,
    sk: &SecretKey,
    user_agent: &str,
    cfg: &ChainConfig,
    threads: usize,
    state: &Arc<MinerState>,
    cli_batch_size: u64,
    wallet_pubkey: &[u8],
    allow_shared_reward_recipient: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Bump connection generation so stale mining threads from a previous
    // connection will see the change and exit.
    state.connection_generation.fetch_add(1, Ordering::Release);
    *state.current_job.write() = None;
    state.job_generation.fetch_add(1, Ordering::Release);

    let identity = ChainIdentity::of(cfg);

    let mut stream = match proxy {
        // The handshake is blocking I/O with a bounded deadline: it must not be
        // spliced into the reactor's async accept path. The socket is handed to
        // tokio afterwards, so the pool link itself stays async.
        Some(proxy) => {
            let proxied = tokio::task::block_in_place(|| proxy.open(pool_addr))?;
            proxied.set_nonblocking(true)?;
            TcpStream::from_std(proxied)?
        }
        None => TcpStream::connect(pool_addr).await?,
    };

    // Enable TCP keepalive to detect dead connections through NAT/firewalls.
    {
        let std_sock = stream.into_std()?;
        let sock = socket2::Socket::from(std_sock);
        let keepalive = socket2::TcpKeepalive::new()
            .with_time(std::time::Duration::from_secs(30))
            .with_interval(std::time::Duration::from_secs(10));
        sock.set_tcp_keepalive(&keepalive)?;
        sock.set_tcp_nodelay(true)?;
        let std_sock: std::net::TcpStream = sock.into();
        std_sock.set_nonblocking(true)?;
        stream = TcpStream::from_std(std_sock)?;
    }

    info!("Connected to pool at {pool_addr}");

    let estimated_operations_per_second = state
        .estimated_compute_rate
        .load(Ordering::Acquire)
        .max((threads as u64).saturating_mul(128));

    let login = LoginRequest {
        miner_id: hex_encode(sk.public_key().as_bytes()),
        user_agent: user_agent.to_string(),
        payout_pubkey: wallet_pubkey.to_vec(),
        estimated_operations_per_second,
        thread_count: threads as u32,
        network_magic: cfg.network_magic.to_vec(),
        protocol_version: POOL_PROTOCOL_VERSION,
        consensus_params_hash: identity.consensus_params_hash.to_vec(),
        genesis_hash: identity.genesis_hash.as_bytes().to_vec(),
    };
    let env = PoolEnvelope::sign(MSG_LOGIN, login.encode_to_vec(), sk);
    PoolCodec::write_envelope(&mut stream, &env).await?;

    let ack_env = PoolCodec::read_envelope(&mut stream).await?;
    ack_env.verify()?;

    if ack_env.msg_type != MSG_LOGIN_ACK {
        return Err("expected LOGIN_ACK".into());
    }

    let ack = LoginAck::decode(&ack_env.payload[..])?;
    if !ack.accepted {
        return Err(format!("login rejected: {}", ack.error).into());
    }
    if ack.protocol_version != POOL_PROTOCOL_VERSION
        || ack.network_magic.as_slice() != cfg.network_magic
        || ack.network_name != cfg.network_name
        || ack.consensus_params_hash.as_slice() != identity.consensus_params_hash
        || ack.genesis_hash.as_slice() != identity.genesis_hash.as_bytes()
    {
        return Err(format!(
            "pool network mismatch: expected {} ({}) v{}, got {} ({}) v{}",
            cfg.network_name,
            hex_encode(cfg.network_magic),
            POOL_PROTOCOL_VERSION,
            ack.network_name,
            hex_encode(&ack.network_magic),
            ack.protocol_version,
        )
        .into());
    }

    info!(
        "Login accepted by pool {}, chain height={}, block_diff={}, share_diff={}, network={}",
        ack.pool_id, ack.chain_height, ack.block_difficulty, ack.share_difficulty, ack.network_name
    );
    info!(
        "Synced with pool: tip_hash={}, block_time_target={}ms",
        hex_encode(&ack.chain_tip_hash),
        ack.block_time_target_ms,
    );

    state
        .current_share_difficulty
        .store(ack.share_difficulty, Ordering::Release);
    state.difficulty_generation.fetch_add(1, Ordering::AcqRel);

    let pool_pubkey: [u8; 32] = ack_env
        .sender_pubkey
        .as_slice()
        .try_into()
        .map_err(|_| "pool public key must be exactly 32 bytes")?;
    state.receipt_sequence.store(0, Ordering::Release);
    *state.receipt_head.write() = [0u8; 32];
    state.pending_submission_hashes.write().clear();

    let conn_gen = state.connection_generation.load(Ordering::Acquire);

    let cfg_clone = cfg.clone();
    let state_clone = Arc::clone(state);
    let submit_tx = start_mining_threads(
        threads,
        state_clone,
        MiningLaunchConfig {
            chain: cfg_clone.clone(),
            batch_size: cli_batch_size,
            connection_generation: conn_gen,
            miner_secret: sk.clone(),
        },
    );

    let compute_rate_state = Arc::clone(state);
    let compute_rate_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(10));
        let mut last_operations: u64 = 0;
        loop {
            interval.tick().await;
            let current = compute_rate_state.total_operations.load(Ordering::Relaxed);
            let delta = current.saturating_sub(last_operations);
            last_operations = current;
            let rate = delta as f64 / 10.0;
            compute_rate_state
                .estimated_compute_rate
                .store(rate.round() as u64, Ordering::Release);
            info!(
                "Scientific compute: {:.2} ops/s | Checkpoints: {} accepted, {} rejected | Blocks: {}",
                rate,
                compute_rate_state.shares_accepted.load(Ordering::Relaxed),
                compute_rate_state.shares_rejected.load(Ordering::Relaxed),
                compute_rate_state.blocks_found.load(Ordering::Relaxed),
            );
        }
    });

    let keepalive_sk = sk.clone();
    let (read_half, write_half) = tokio::io::split(stream);
    let read_half = Arc::new(tokio::sync::Mutex::new(read_half));
    let write_half = Arc::new(tokio::sync::Mutex::new(write_half));

    let writer = Arc::clone(&write_half);
    let ka_sk = keepalive_sk.clone();
    let keepalive_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(15));
        loop {
            interval.tick().await;
            let env = PoolEnvelope::sign(MSG_KEEPALIVE, Vec::new(), &ka_sk);
            let data = env.encode_to_vec();
            let mut w = writer.lock().await;
            use tokio::io::AsyncWriteExt;
            if w.write_u32(data.len() as u32).await.is_err() {
                break;
            }
            if w.write_all(&data).await.is_err() {
                break;
            }
        }
    });

    let report_writer = Arc::clone(&write_half);
    let report_sk = sk.clone();
    let report_state = Arc::clone(state);
    let start_time = std::time::Instant::now();
    let compute_rate_report_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(5));
        let mut last_operations: u64 = 0;
        loop {
            interval.tick().await;
            let current = report_state.total_operations.load(Ordering::Relaxed);
            let delta = current.saturating_sub(last_operations);
            last_operations = current;
            let rate = delta / 5;
            report_state
                .estimated_compute_rate
                .store(rate, Ordering::Release);
            let report = ComputeRateReport {
                operations_per_second: rate,
                total_operations: current,
                uptime_secs: start_time.elapsed().as_secs(),
            };
            let env =
                PoolEnvelope::sign(MSG_COMPUTE_RATE_REPORT, report.encode_to_vec(), &report_sk);
            let data = env.encode_to_vec();
            let mut w = report_writer.lock().await;
            use tokio::io::AsyncWriteExt;
            if w.write_u32(data.len() as u32).await.is_err() {
                break;
            }
            if w.write_all(&data).await.is_err() {
                break;
            }
        }
    });

    let mut share_rx = submit_tx;

    let loop_result: Result<(), Box<dyn std::error::Error>> = async {
        let read_timeout = std::time::Duration::from_secs(90);
        loop {
            tokio::select! {
                result = tokio::time::timeout(read_timeout, read_envelope_from(&read_half)) => {
                    let env = match result {
                        Ok(r) => r?,
                        Err(_) => return Err("pool read timeout (90s with no data)".into()),
                    };
                    env.verify()?;

                    match env.msg_type {
                        MSG_JOB => {
                            let template = JobTemplate::decode(&env.payload[..])?;
                            handle_new_job(
                                &template,
                                cfg,
                                state,
                                *sk.public_key().as_bytes(),
                                wallet_pubkey,
                                allow_shared_reward_recipient,
                            )?;
                        }
                        MSG_SUBMIT_RESULT => {
                            let result = SubmitResult::decode(&env.payload[..])?;
                            let submission_hash = state
                                .pending_submission_hashes
                                .write()
                                .pop_front()
                                .ok_or("pool returned a share result with no pending submission")?;
                            if result.accepted {
                                verify_share_receipt(
                                    &env,
                                    state,
                                    &pool_pubkey,
                                    sk.public_key().as_bytes(),
                                    &submission_hash,
                                )?;
                                state.shares_accepted.fetch_add(1, Ordering::Relaxed);
                                if result.block_found {
                                    state.blocks_found.fetch_add(1, Ordering::Relaxed);
                                    info!(
                                        "BLOCK FOUND! hash={}",
                                        hex_encode(&result.block_hash)
                                    );
                                }
                            } else {
                                state.shares_rejected.fetch_add(1, Ordering::Relaxed);
                                warn!("Share rejected: {}", result.error);
                            }
                        }
                        MSG_BLOCK_FOUND => {
                            let notify = BlockFoundNotify::decode(&env.payload[..])?;
                            info!(
                                "Block found at height {} by {}",
                                notify.height,
                                hex_encode(&notify.finder_pubkey)
                            );
                        }
                        MSG_SET_DIFFICULTY => {
                            let set_diff = SetDifficulty::decode(&env.payload[..])?;
                            info!(
                                "VarDiff: pool adjusted share difficulty to {}",
                                set_diff.share_difficulty
                            );
                            state.current_share_difficulty.store(
                                set_diff.share_difficulty,
                                Ordering::Release,
                            );
                            state.difficulty_generation.fetch_add(1, Ordering::AcqRel);
                        }
                        MSG_CHAIN_STATE => {
                            let chain_info = ChainStateInfo::decode(&env.payload[..])?;
                            info!(
                                "Chain state update: height={}, block_diff={}, tip={}",
                                chain_info.height,
                                chain_info.difficulty,
                                hex_encode(&chain_info.tip_hash),
                            );
                        }
                        _ => {}
                    }
                }

                share = share_rx.recv() => {
                    if let Some(candidate) = share {
                        let submission = ShareSubmission {
                            job_id: candidate.job_id.to_vec(),
                            chain_length: candidate.chain_length as u64,
                            witnesses: candidate.witnesses.clone(),
                            work_commitment: candidate.work_commitment.as_bytes().to_vec(),
                            block_authorization: candidate.block_authorization,
                        };
                        let submission_payload = submission.encode_to_vec();
                        let submission_hash = *hyphen_crypto::blake3_hash(&submission_payload).as_bytes();
                        {
                            let mut pending = state.pending_submission_hashes.write();
                            if pending.len() >= 4_096 {
                                return Err("pool left too many share submissions unanswered".into());
                            }
                            pending.push_back(submission_hash);
                        }
                        let env = PoolEnvelope::sign(
                            MSG_SUBMIT,
                            submission_payload,
                            sk,
                        );
                        let data = env.encode_to_vec();
                        let mut w = write_half.lock().await;
                        use tokio::io::AsyncWriteExt;
                        w.write_u32(data.len() as u32).await?;
                        w.write_all(&data).await?;
                    }
                }
            }
        }
    }
    .await;

    // Abort background tasks so they don't leak on reconnect
    compute_rate_handle.abort();
    keepalive_handle.abort();
    compute_rate_report_handle.abort();

    loop_result
}

async fn read_envelope_from(
    reader: &Arc<tokio::sync::Mutex<tokio::io::ReadHalf<TcpStream>>>,
) -> Result<PoolEnvelope, PoolError> {
    use tokio::io::AsyncReadExt;
    let mut r = reader.lock().await;
    let len = r.read_u32().await?;
    if len > 1024 * 1024 {
        return Err(PoolError::Internal(format!("frame too large: {len} bytes")));
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf).await?;
    let envelope = PoolEnvelope::decode(&buf[..])?;
    Ok(envelope)
}

fn handle_new_job(
    template: &JobTemplate,
    cfg: &ChainConfig,
    state: &Arc<MinerState>,
    miner_pubkey: [u8; 32],
    payout_keys: &[u8],
    allow_shared_reward_recipient: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let header: BlockHeader = wire_config(4096).deserialize(&template.header_data)?;

    if template.job_id.len() != 32 {
        return Err("job id must be exactly 32 bytes".into());
    }
    if template.epoch_seed.len() != 32 {
        return Err("epoch seed must be exactly 32 bytes".into());
    }
    if template.share_difficulty == 0 || template.block_difficulty == 0 {
        return Err("job difficulty must be greater than zero".into());
    }
    if template.share_difficulty > template.block_difficulty {
        return Err("share difficulty cannot exceed block difficulty".into());
    }
    // The Cunningham chain length is inert on the neural profile; the work
    // challenge has its own shape and the pool does not send a chain length for
    // it.
    if !cfg.feature_enabled(hyphen_core::FEATURE_NEURAL_WORK)
        && template.chain_length != cfg.pouw_chain_length as u64
    {
        return Err(format!(
            "pool sent chain length {} but this network fixes {}/{} elements",
            template.chain_length,
            cfg.pouw_chain_length,
            cfg.pouw_chain_length + 1,
        )
        .into());
    }
    if !(hyphen_pow::vms::MIN_CHAIN_LENGTH..=hyphen_pow::vms::MAX_CHAIN_LENGTH)
        .contains(&cfg.pouw_chain_length)
    {
        return Err("network configuration has an unsupported chain length".into());
    }
    // On the neural profile this field is an acceptance threshold, not a witness
    // count, so the witness-count ceiling does not apply.
    if !cfg.feature_enabled(hyphen_core::FEATURE_NEURAL_WORK)
        && template.block_difficulty > hyphen_pow::vms::MAX_WITNESSES_PER_BLOCK
    {
        return Err("pool requested more witnesses than a block may claim".into());
    }
    if template.parent_work_commitment.len() != 32 {
        return Err("job parent work commitment must be exactly 32 bytes".into());
    }
    let identity = ChainIdentity::of(cfg);
    if template.consensus_params_hash.as_slice() != identity.consensus_params_hash
        || template.genesis_hash.as_slice() != identity.genesis_hash.as_bytes()
    {
        return Err("pool sent a job for an incompatible chain identity".into());
    }

    if template.epoch_seed.len() != 32 {
        return Err("epoch seed must be exactly 32 bytes".into());
    }
    if template.share_difficulty == 0 || template.block_difficulty == 0 {
        return Err("job difficulty must be greater than zero".into());
    }
    if template.reward_view_public.len() != 32 || template.reward_spend_public.len() != 32 {
        return Err("job reward keys must both be exactly 32 bytes".into());
    }
    if payout_keys.len() != 64 {
        return Err("local payout keys must be exactly 64 bytes".into());
    }
    let mut reward_view_public = [0u8; 32];
    reward_view_public.copy_from_slice(&template.reward_view_public);
    let mut reward_spend_public = [0u8; 32];
    reward_spend_public.copy_from_slice(&template.reward_spend_public);
    if reward_view_public == [0u8; 32] || reward_spend_public == [0u8; 32] {
        return Err("pool supplied a zero reward key".into());
    }
    if !allow_shared_reward_recipient
        && (reward_view_public != payout_keys[..32] || reward_spend_public != payout_keys[32..64])
    {
        return Err(
            "pool redirected the reward address; use --allow-shared-reward-recipient only after reviewing the pool policy"
                .into(),
        );
    }
    if header.miner_pubkey != [0u8; 32] && header.miner_pubkey != miner_pubkey {
        return Err("job header is already bound to a different miner identity".into());
    }
    let mut header = header;
    header.miner_pubkey = miner_pubkey;

    if header.version != FROZEN_BLOCK_VERSION
        || header.height != template.height
        || header.difficulty != template.block_difficulty
        || header.epoch_seed.as_bytes() != template.epoch_seed.as_slice()
        || header.prev_hash.as_bytes() != template.prev_hash.as_slice()
    {
        return Err("job header metadata mismatch".into());
    }
    if compute_tx_root(&template.transactions) != header.tx_root {
        return Err("job transaction list does not match the committed tx_root".into());
    }

    let mut job_id = [0u8; 32];
    job_id.copy_from_slice(&template.job_id);

    let mut parent_work_commitment = [0u8; 32];
    parent_work_commitment.copy_from_slice(&template.parent_work_commitment);

    let job = Arc::new(MiningJob {
        job_id,
        header,
        chain_length: cfg.pouw_chain_length,
        parent_work_commitment: Hash256::from_bytes(parent_work_commitment),
        reward_view_public,
        reward_spend_public,
    });

    state
        .current_share_difficulty
        .store(template.share_difficulty, Ordering::Release);
    state.difficulty_generation.fetch_add(1, Ordering::AcqRel);

    let active_share_diff = state.current_share_difficulty.load(Ordering::Acquire);
    info!(
        "New job: height={}, witnesses={}, chain_length={}, credit_at={}, active_credit_at={}, clean={}",
        template.height,
        template.block_difficulty,
        template.chain_length,
        template.share_difficulty,
        active_share_diff,
        template.clean_jobs
    );

    *state.current_job.write() = Some(job.clone());
    state.job_generation.fetch_add(1, Ordering::Release);

    Ok(())
}

struct ShareCandidate {
    job_id: [u8; 32],
    chain_length: u32,
    /// Ascending single-use witness list.
    witnesses: Vec<u64>,
    /// Commitment over `parent_work_commitment`, height and the exact list.
    work_commitment: Hash256,
    /// Miner authorization; empty for a credit-only share.
    block_authorization: Vec<u8>,
}

struct MiningThreadConfig {
    thread_id: usize,
    thread_count: usize,
    chain: ChainConfig,
    batch_size: u64,
    connection_generation: u64,
    miner_secret: SecretKey,
}

struct MiningLaunchConfig {
    chain: ChainConfig,
    batch_size: u64,
    connection_generation: u64,
    miner_secret: SecretKey,
}

fn start_mining_threads(
    threads: usize,
    state: Arc<MinerState>,
    launch: MiningLaunchConfig,
) -> tokio::sync::mpsc::UnboundedReceiver<ShareCandidate> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

    for thread_id in 0..threads {
        let cfg = launch.chain.clone();
        let state = Arc::clone(&state);
        let tx = tx.clone();
        let miner_secret = launch.miner_secret.clone();

        std::thread::spawn(move || {
            mining_thread(
                MiningThreadConfig {
                    thread_id,
                    thread_count: threads,
                    chain: cfg,
                    batch_size: launch.batch_size,
                    connection_generation: launch.connection_generation,
                    miner_secret,
                },
                state,
                tx,
            );
        });
    }

    rx
}

fn mining_thread(
    config: MiningThreadConfig,
    state: Arc<MinerState>,
    tx: tokio::sync::mpsc::UnboundedSender<ShareCandidate>,
) {
    let MiningThreadConfig {
        thread_id,
        thread_count,
        chain: cfg,
        batch_size,
        connection_generation: conn_gen,
        miner_secret,
    } = config;
    let identity = ChainIdentity::of(&cfg);
    info!("Mining thread {thread_id}/{thread_count} started (batch_size={batch_size})");

    // Witnesses must be odd, so an odd per-thread offset with a stride of
    // `2 * thread_count` partitions the candidate space exactly once.
    let stride = 2 * thread_count as u64;
    let mut cursor = 3 + 2 * thread_id as u64;

    loop {
        if !state.running.load(Ordering::Relaxed) {
            break;
        }
        if state.connection_generation.load(Ordering::Acquire) != conn_gen {
            break;
        }

        let job = match state.current_job.read().clone() {
            Some(job) => job,
            None => {
                std::thread::sleep(std::time::Duration::from_millis(50));
                continue;
            }
        };
        let generation = state.job_generation.load(Ordering::Acquire);

        if cfg.feature_enabled(hyphen_core::FEATURE_NEURAL_WORK) {
            if let Err(error) = mine_neural_block(
                &cfg,
                &state,
                &job,
                generation,
                thread_id,
                thread_count,
                &tx,
                &miner_secret,
            ) {
                error!("Thread {thread_id}: {error}");
            }
            // A solution only becomes a block once the pool submits it, so wait
            // for the next job instead of burning a second solution.
            while state.running.load(Ordering::Relaxed)
                && state.connection_generation.load(Ordering::Acquire) == conn_gen
                && state.job_generation.load(Ordering::Acquire) == generation
            {
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
            continue;
        }

        let required = job.header.difficulty;
        let credit_at = state
            .current_share_difficulty
            .load(Ordering::Acquire)
            .clamp(1, required);

        let mut witnesses = Vec::new();
        if credit_at < required {
            let credit = search_witnesses(
                &state,
                generation,
                job.chain_length,
                credit_at,
                batch_size,
                &mut cursor,
                stride,
            );
            if credit.job_changed {
                continue;
            }
            if credit.witnesses.len() as u64 == credit_at {
                let work_commitment = hyphen_pow::vms::witness_commitment(
                    &job.parent_work_commitment,
                    job.header.height,
                    &credit.witnesses,
                );
                state.mark_submitted(&credit.witnesses);
                info!("Thread {thread_id}: credit share with {credit_at} witnesses");
                witnesses = credit.witnesses;
                if tx
                    .send(ShareCandidate {
                        job_id: job.job_id,
                        chain_length: job.chain_length,
                        witnesses: witnesses.clone(),
                        work_commitment,
                        block_authorization: Vec::new(),
                    })
                    .is_err()
                {
                    break;
                }
            }
        }

        let remaining = required - witnesses.len() as u64;
        if remaining > 0 {
            let rest = search_witnesses(
                &state,
                generation,
                job.chain_length,
                remaining,
                batch_size,
                &mut cursor,
                stride,
            );
            if rest.job_changed || rest.witnesses.len() as u64 != remaining {
                continue;
            }
            witnesses.extend(rest.witnesses);
            witnesses.sort_unstable();
        }

        let work_commitment = hyphen_pow::vms::witness_commitment(
            &job.parent_work_commitment,
            job.header.height,
            &witnesses,
        );
        let mut header = job.header.clone();
        header.pow_commitment = work_commitment;
        let block_authorization = match BlockAuthorization::sign(
            &header,
            identity.network_magic,
            identity.consensus_params_hash,
            identity.genesis_hash,
            job.reward_view_public,
            job.reward_spend_public,
            &miner_secret,
        ) {
            Ok(authorization) => match wire_config(256).serialize(&authorization) {
                Ok(encoded) => encoded,
                Err(error) => {
                    error!("cannot serialise block authorization: {error}");
                    continue;
                }
            },
            Err(error) => {
                error!("refusing unsigned witness result: {error}");
                continue;
            }
        };

        state.mark_submitted(&witnesses);
        info!(
            "Thread {thread_id}: block candidate height={} witnesses={}",
            job.header.height,
            witnesses.len()
        );
        if tx
            .send(ShareCandidate {
                job_id: job.job_id,
                chain_length: job.chain_length,
                witnesses,
                work_commitment,
                block_authorization,
            })
            .is_err()
        {
            break;
        }

        // A candidate only becomes a block once the pool submits it, so wait
        // for the next job instead of burning a second list on this one.
        while state.running.load(Ordering::Relaxed)
            && state.connection_generation.load(Ordering::Acquire) == conn_gen
            && state.job_generation.load(Ordering::Acquire) == generation
        {
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
    }

    info!("Mining thread {thread_id} stopped");
}

/// Largest batch of candidates searched before the thread re-checks for a new
/// job. One candidate is one forward pass, so this bounds the latency between a
/// new job arriving and the search noticing.
const NEURAL_WORK_BATCH: u64 = 32_768;

/// Searches the one-way neural work challenge and submits one solution.
///
/// The challenge is fully determined by public data — the parent hash, the
/// height, the miner key and the header's threshold — so the search is a plain
/// forward-pass loop. Each thread walks its own counter residue class, which
/// keeps workers disjoint without coordination.
#[allow(clippy::too_many_arguments)]
fn mine_neural_block(
    cfg: &ChainConfig,
    state: &Arc<MinerState>,
    job: &MiningJob,
    generation: u64,
    thread_id: usize,
    thread_count: usize,
    tx: &tokio::sync::mpsc::UnboundedSender<ShareCandidate>,
    miner_secret: &hyphen_crypto::SecretKey,
) -> Result<(), String> {
    let identity = ChainIdentity::of(cfg);
    let spec = hyphen_neural::PROFILE_SPEC;
    let miner_pubkey = *miner_secret.public_key().as_bytes();
    let parent_hash = job.header.prev_hash;
    let seed = hyphen_core::neural_work_seed(&parent_hash, job.header.height, &miner_pubkey);
    let descriptor = hyphen_neural::WorkDescriptor {
        epoch: job.header.height,
        spec,
        threshold: job.header.difficulty,
        chain_seed: *seed.as_bytes(),
    };
    descriptor
        .validate()
        .map_err(|error| format!("pool sent an unusable work descriptor: {error}"))?;

    let mut weights = vec![0_i64; spec.weight_count()];
    descriptor
        .derive_weights(&mut weights)
        .map_err(|error| format!("cannot derive work weights: {error}"))?;
    let mut solution = vec![0_i64; spec.input_dim as usize];
    let mut hidden = vec![0_i64; spec.hidden_dim as usize];
    let mut output = vec![0_i64; spec.output_dim as usize];
    let mut scratch = vec![0_i64; spec.input_dim as usize];

    let stride = thread_count.max(1) as u64;
    let mut counter = thread_id as u64;
    let mut attempts: u64 = 0;
    loop {
        // `generation` moves whenever the job is replaced, and the outer loop
        // owns the connection check, so this is the whole stop condition. A
        // sibling worker that already solved this job stops us from spending a
        // second solution the pool would reject as a stale job.
        if !state.running.load(Ordering::Relaxed)
            || state.job_generation.load(Ordering::Acquire) != generation
            || state.neural_solution_generation.load(Ordering::Acquire) == generation
        {
            return Ok(());
        }
        let outcome = hyphen_neural::search_work(
            &descriptor,
            &weights,
            counter,
            stride,
            NEURAL_WORK_BATCH,
            &mut solution,
            &mut hidden,
            &mut output,
            &mut scratch,
        )
        .map_err(|error| format!("work search failed: {error}"))?;
        attempts = attempts.saturating_add(outcome.attempts);
        state
            .total_operations
            .fetch_add(outcome.attempts, Ordering::Relaxed);
        if outcome.found {
            break;
        }
        counter = counter.wrapping_add(stride.saturating_mul(NEURAL_WORK_BATCH));
    }

    // Claim the job before submitting: the first worker to finish wins and the
    // rest stop at their next loop check.
    if state
        .neural_solution_generation
        .swap(generation, Ordering::AcqRel)
        == generation
    {
        return Ok(());
    }

    let witnesses: Vec<u64> = solution.iter().map(|value| *value as u64).collect();
    let commitment = hyphen_core::neural_work_commitment(
        &parent_hash,
        job.header.height,
        job.header.difficulty,
        &miner_pubkey,
        &witnesses,
    );

    let mut header = job.header.clone();
    header.pow_commitment = commitment;
    let authorization = BlockAuthorization::sign(
        &header,
        identity.network_magic,
        identity.consensus_params_hash,
        identity.genesis_hash,
        job.reward_view_public,
        job.reward_spend_public,
        miner_secret,
    )
    .map_err(|error| format!("cannot sign the block authorization: {error}"))?;
    let block_authorization = wire_config(256)
        .serialize(&authorization)
        .map_err(|error| format!("cannot serialise the block authorization: {error}"))?;

    info!(
        "Thread {thread_id}: neural solution at height {} after {attempts} attempts",
        job.header.height
    );
    tx.send(ShareCandidate {
        job_id: job.job_id,
        chain_length: job.chain_length,
        witnesses,
        work_commitment: commitment,
        block_authorization,
    })
    .map_err(|_| "submission channel closed".to_string())
}

struct WitnessSearch {
    witnesses: Vec<u64>,
    job_changed: bool,
}

fn run_accelerator_check(
    batch: u32,
    offset: u64,
    stride: u64,
    length: u32,
    accelerator_dir: Option<&std::path::Path>,
) -> Result<(), String> {
    let limit = hyphen_pow::SMALL_PRIME_SIEVE_LIMIT as u64;
    if offset < limit {
        return Err(format!(
            "offset must be at least {limit}; below the sieve bound an element can equal a sieve prime"
        ));
    }
    if stride == 0 {
        return Err("stride must be non-zero".into());
    }
    if !(hyphen_pow::MIN_CHAIN_LENGTH..=hyphen_pow::MAX_CHAIN_LENGTH).contains(&length) {
        return Err("chain length is outside the consensus range".into());
    }
    let accelerator = Accelerator::load(accelerator_dir)?;
    let requested = batch.clamp(1, 1 << 24);
    offset
        .checked_add(stride.saturating_mul(requested as u64 - 1))
        .ok_or_else(|| "candidate window overflows u64".to_string())?;
    // Warm the CUDA/HIP context so the measured scan is not dominated by
    // first-call initialization.
    let warmup = requested.min(1_024);
    let _ = accelerator.scan(offset, stride, warmup, length, warmup)?;
    let started = std::time::Instant::now();
    let outcome = accelerator.scan(offset, stride, requested, length, requested)?;
    let elapsed = started.elapsed();
    let scanned = outcome.scanned.max(1);

    let mut cpu = Vec::new();
    for index in 0..scanned as u64 {
        let candidate = offset + index * stride;
        if hyphen_pow::vms::passes_small_prime_sieve(candidate, length) {
            cpu.push(candidate);
        }
    }

    let mut gpu = Vec::with_capacity(outcome.survivors.len());
    for candidate in &outcome.survivors {
        let relative = candidate.checked_sub(offset).ok_or_else(|| {
            format!("accelerator returned {candidate}, below the requested window")
        })?;
        if relative % stride != 0 || relative / stride >= scanned as u64 {
            return Err(format!(
                "accelerator returned {candidate}, outside the requested window"
            ));
        }
        if hyphen_pow::vms::chain_element(*candidate, length - 1).is_some() {
            gpu.push(*candidate);
        }
    }
    gpu.sort_unstable();
    cpu.sort_unstable();

    if !outcome.truncated && gpu != cpu {
        let first = gpu
            .iter()
            .zip(&cpu)
            .find(|(left, right)| left != right)
            .map(|(left, right)| format!("GPU has {left}, CPU has {right}"))
            .unwrap_or_else(|| {
                format!(
                    "GPU returned {} survivors, CPU returned {}",
                    gpu.len(),
                    cpu.len()
                )
            });
        return Err(format!("accelerator sieve mismatch: {first}"));
    }

    let chains = gpu
        .iter()
        .filter(|candidate| hyphen_pow::vms::chain_is_valid(**candidate, length))
        .count();
    println!("backend:      {}", accelerator.backend());
    println!("device:       {}", accelerator.device_name());
    println!("plugin:       {}", accelerator.path().display());
    println!("window:       {scanned} candidates from {offset} step {stride}");
    println!("survivors:    {} GPU, {} CPU", gpu.len(), cpu.len());
    println!("valid chains: {chains}");
    println!(
        "device time:  {:.3} ms",
        outcome.device_time_ns as f64 / 1e6
    );
    println!("wall time:    {:.3} ms", elapsed.as_secs_f64() * 1e3);
    if outcome.truncated {
        println!("note: survivor buffer was truncated; exact CPU comparison skipped");
    }
    Ok(())
}

/// Scans this worker's residue class until `count` unclaimed witnesses are
/// found. The cursor only moves forward, so successive calls never rescan.
fn search_witnesses(
    state: &Arc<MinerState>,
    generation: u64,
    chain_length: u32,
    count: u64,
    batch: u64,
    cursor: &mut u64,
    stride: u64,
) -> WitnessSearch {
    let mut witnesses: Vec<u64> = Vec::with_capacity(count as usize);
    while (witnesses.len() as u64) < count {
        if state.job_generation.load(Ordering::Acquire) != generation {
            return WitnessSearch {
                witnesses,
                job_changed: true,
            };
        }

        let accelerator_ready = *cursor >= hyphen_pow::SMALL_PRIME_SIEVE_LIMIT as u64;
        if accelerator_ready && !state.accelerator_failed.load(Ordering::Acquire) {
            if let Some(accelerator) = state.accelerator.as_ref() {
                let scanned = batch.min(accelerator.max_candidates() as u64).max(1) as u32;
                let missing = count - witnesses.len() as u64;
                let max_survivors = missing
                    .saturating_mul(8)
                    .saturating_add(1_024)
                    .clamp(1, scanned as u64) as u32;
                match accelerator.scan(*cursor, stride, scanned, chain_length, max_survivors) {
                    Ok(outcome) => {
                        state
                            .total_operations
                            .fetch_add(scanned as u64, Ordering::Relaxed);
                        let truncated = outcome.truncated;
                        for candidate in outcome.survivors {
                            if witnesses.contains(&candidate)
                                || !hyphen_pow::vms::chain_is_valid_after_sieve(
                                    candidate,
                                    chain_length,
                                )
                                || state.is_locally_claimed(candidate)
                            {
                                continue;
                            }
                            witnesses.push(candidate);
                            if witnesses.len() as u64 == count {
                                break;
                            }
                        }
                        *cursor = cursor.wrapping_add(stride.saturating_mul(scanned as u64));
                        if truncated && (witnesses.len() as u64) < count {
                            warn!(
                                target: "hyphen::miner",
                                "Accelerator survivor window truncated at candidate {}",
                                *cursor
                            );
                        }
                        continue;
                    }
                    Err(error) => {
                        warn!("Native accelerator disabled after device error: {error}");
                        state.accelerator_failed.store(true, Ordering::Release);
                    }
                }
            }
        }

        let is_claimed = |candidate: u64| state.is_locally_claimed(candidate);
        let params = hyphen_pow::vms::SearchParams {
            length: chain_length,
            offset: *cursor,
            stride,
            count: count - witnesses.len() as u64,
            max_candidates: batch,
            is_claimed: &is_claimed,
        };
        let outcome = match hyphen_pow::vms::search(&params) {
            Ok(outcome) => outcome,
            Err(error) => {
                error!("witness search rejected: {error}");
                return WitnessSearch {
                    witnesses,
                    job_changed: true,
                };
            }
        };
        state
            .total_operations
            .fetch_add(outcome.candidates_tested, Ordering::Relaxed);
        witnesses.extend(outcome.witnesses);
        *cursor = cursor.wrapping_add(stride.saturating_mul(outcome.candidates_tested.max(1)));
        if outcome.candidates_tested == 0 {
            break;
        }
    }
    WitnessSearch {
        witnesses,
        job_changed: false,
    }
}

fn compute_tx_root(transactions: &[Vec<u8>]) -> Hash256 {
    if transactions.is_empty() {
        return Hash256::ZERO;
    }
    let mut level: Vec<Hash256> = transactions
        .iter()
        .map(|transaction| primitives::blake3_hash(transaction))
        .collect();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        for pair in level.chunks(2) {
            next.push(if pair.len() == 2 {
                primitives::blake3_hash_many(&[pair[0].as_bytes(), pair[1].as_bytes()])
            } else {
                pair[0]
            });
        }
        level = next;
    }
    level[0]
}

fn verify_share_receipt(
    envelope: &PoolEnvelope,
    state: &Arc<MinerState>,
    pool_pubkey: &[u8; 32],
    miner_pubkey: &[u8; 32],
    submission_hash: &[u8; 32],
) -> Result<(), Box<dyn std::error::Error>> {
    let expected_sequence = state
        .receipt_sequence
        .load(Ordering::Acquire)
        .saturating_add(1);
    let previous = *state.receipt_head.read();
    let result_hash = *primitives::blake3_hash(&envelope.payload).as_bytes();
    let expected_receipt = share_receipt_hash(
        pool_pubkey,
        miner_pubkey,
        expected_sequence,
        &previous,
        submission_hash,
        &result_hash,
    );
    if envelope.receipt_sequence != expected_sequence
        || envelope.previous_receipt_hash.as_slice() != previous
        || envelope.receipt_hash.len() != 32
        || envelope.receipt_hash.as_slice() != expected_receipt
    {
        return Err("pool returned a discontinuous or malformed signed share receipt".into());
    }
    let mut receipt = [0u8; 32];
    receipt.copy_from_slice(&envelope.receipt_hash);
    *state.receipt_head.write() = receipt;
    state
        .receipt_sequence
        .store(expected_sequence, Ordering::Release);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn address_for(cfg: &ChainConfig) -> String {
        let version = if cfg.network_magic == [0x48, 0x59, 0x50, 0x4e] {
            0x01
        } else {
            0x02
        };
        let mut payload = vec![version];
        payload.extend_from_slice(&[11u8; 32]);
        payload.extend_from_slice(&[12u8; 32]);
        let hash = hyphen_crypto::blake3_hash(&payload);
        payload.extend_from_slice(&hash.as_bytes()[..4]);
        format!("hy1{}", hyphen_crypto::base58::encode(&payload))
    }

    #[test]
    fn wallet_address_parser_returns_errors_instead_of_panicking() {
        let cfg = ChainConfig::devnet();
        assert!(parse_wallet_address(&address_for(&cfg), &cfg).is_ok());
        assert!(parse_wallet_address("hy1not-base58-0OIl", &cfg).is_err());
        assert!(parse_wallet_address("hy1", &cfg).is_err());

        let mainnet_address = address_for(&ChainConfig::mainnet());
        assert!(parse_wallet_address(&mainnet_address, &cfg).is_err());
    }

    fn accepted_receipt(
        state: &Arc<MinerState>,
        pool_pubkey: &[u8; 32],
        miner_pubkey: &[u8; 32],
        submission_hash: &[u8; 32],
    ) -> PoolEnvelope {
        let payload = SubmitResult {
            accepted: true,
            error: String::new(),
            block_found: false,
            block_hash: Vec::new(),
        }
        .encode_to_vec();
        let result_hash = *primitives::blake3_hash(&payload).as_bytes();
        let previous = *state.receipt_head.read();
        let sequence = state.receipt_sequence.load(Ordering::Acquire) + 1;
        let receipt = share_receipt_hash(
            pool_pubkey,
            miner_pubkey,
            sequence,
            &previous,
            submission_hash,
            &result_hash,
        );
        PoolEnvelope {
            msg_type: MSG_SUBMIT_RESULT,
            payload,
            sender_pubkey: pool_pubkey.to_vec(),
            signature: vec![0u8; 64],
            timestamp: 0,
            nonce: 0,
            receipt_sequence: sequence,
            previous_receipt_hash: previous.to_vec(),
            receipt_hash: receipt.to_vec(),
        }
    }

    #[test]
    fn miner_recomputes_and_chains_accepted_share_receipts() {
        let state = Arc::new(MinerState::new());
        let pool_pubkey = [41u8; 32];
        let miner_pubkey = [42u8; 32];
        let first_submission = [43u8; 32];
        let first = accepted_receipt(&state, &pool_pubkey, &miner_pubkey, &first_submission);
        verify_share_receipt(
            &first,
            &state,
            &pool_pubkey,
            &miner_pubkey,
            &first_submission,
        )
        .unwrap();
        assert_eq!(state.receipt_sequence.load(Ordering::Acquire), 1);

        let second_submission = [44u8; 32];
        let second = accepted_receipt(&state, &pool_pubkey, &miner_pubkey, &second_submission);
        verify_share_receipt(
            &second,
            &state,
            &pool_pubkey,
            &miner_pubkey,
            &second_submission,
        )
        .unwrap();
        assert_eq!(state.receipt_sequence.load(Ordering::Acquire), 2);
    }

    #[test]
    fn miner_rejects_receipt_for_a_different_submission() {
        let state = Arc::new(MinerState::new());
        let pool_pubkey = [51u8; 32];
        let miner_pubkey = [52u8; 32];
        let submitted = [53u8; 32];
        let envelope = accepted_receipt(&state, &pool_pubkey, &miner_pubkey, &submitted);
        let substituted = [54u8; 32];

        assert!(
            verify_share_receipt(&envelope, &state, &pool_pubkey, &miner_pubkey, &substituted,)
                .is_err()
        );
        assert_eq!(state.receipt_sequence.load(Ordering::Acquire), 0);
    }

    #[test]
    fn witness_search_returns_verifiable_single_use_chains() {
        let cfg = ChainConfig::devnet();
        let length = cfg.pouw_chain_length;
        let claimed = std::sync::Mutex::new(std::collections::HashSet::new());
        let is_claimed = |candidate: u64| claimed.lock().expect("lock").contains(&candidate);
        let params = hyphen_pow::vms::SearchParams {
            length,
            offset: 3,
            stride: 2,
            count: 2,
            max_candidates: 5_000_000,
            is_claimed: &is_claimed,
        };
        let outcome = hyphen_pow::vms::search(&params).expect("search");
        assert_eq!(outcome.witnesses.len(), 2);
        hyphen_pow::vms::verify_witnesses(&outcome.witnesses, length, 2).expect("verifiable");
        for witness in &outcome.witnesses {
            claimed.lock().expect("lock").insert(*witness);
        }

        // Replaying the same window must skip everything already claimed.
        let again = hyphen_pow::vms::search(&hyphen_pow::vms::SearchParams {
            length,
            offset: 3,
            stride: 2,
            count: 2,
            max_candidates: 5_000_000,
            is_claimed: &is_claimed,
        })
        .expect("search");
        for witness in &again.witnesses {
            assert!(!outcome.witnesses.contains(witness));
        }
    }

    #[test]
    fn commitment_binds_parent_height_and_exact_list() {
        let parent = Hash256::from_bytes([9u8; 32]);
        let witnesses = [3u64, 11u64];
        let commitment = hyphen_pow::vms::witness_commitment(&parent, 7, &witnesses);
        assert_eq!(
            commitment,
            hyphen_pow::vms::witness_commitment(&parent, 7, &witnesses)
        );
        assert_ne!(
            commitment,
            hyphen_pow::vms::witness_commitment(&parent, 8, &witnesses)
        );
        assert_ne!(
            commitment,
            hyphen_pow::vms::witness_commitment(&parent, 7, &[3u64])
        );
    }

    #[test]
    fn local_claim_set_blocks_self_duplicates() {
        let state = MinerState::new();
        assert!(!state.is_locally_claimed(3));
        state.mark_submitted(&[3, 11]);
        assert!(state.is_locally_claimed(3));
        assert!(state.is_locally_claimed(11));
        assert!(!state.is_locally_claimed(5));
    }
}
