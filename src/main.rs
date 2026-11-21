mod accelerator;
mod claims;
mod primitives;
mod protocol;
mod proxy;

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use accelerator::Accelerator;
use clap::{Parser, Subcommand};
use parking_lot::RwLock;
use prost::Message;
use tokio::net::TcpStream;
use tracing::{debug, error, info, warn};

use primitives::{
    BlockAuthorization, BlockHeader, ChainConfig, ChainIdentity, FROZEN_BLOCK_VERSION, Hash256,
    SecretKey, fill_system_random, hex_encode, wire_config,
};
use protocol::*;

const MAX_UNANSWERED_SHARES: usize = 16;
const POOL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(90);

const FEATURE_ASSET_RULES_V1: u32 = 1 << 9;

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
        value_parser = ["neural-devnet", "testnet", "mainnet"],
        default_value = "neural-devnet"
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
    #[arg(long, global = true, default_value_t = false)]
    no_accelerator: bool,

    /// Enable a hardware plugin from this directory. Omit for CPU mining.
    #[arg(long, global = true, conflicts_with = "no_accelerator")]
    accelerator_dir: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Generate a new Ed25519 key pair and save to file
    Keygen {
        /// Output file path for the 32-byte secret key
        #[arg(long, default_value = "miner.key")]
        output: String,
    },
    /// Compare native HYN3 scores and complete nonce search with the Rust reference
    AcceleratorCheck {
        /// Number of consecutive samples in the comparison window
        #[arg(long, default_value = "1024")]
        batch: u32,
        /// First nonce
        #[arg(long, default_value = "0")]
        offset: u64,
        /// Distance between candidates
        #[arg(long, default_value = "1")]
        stride: u64,
        /// Time complete GPU and single-thread Rust searches over the same window.
        #[arg(long)]
        timings: bool,
    },
}

struct MiningJob {
    job_id: [u8; 32],
    candidate_in_flight: AtomicBool,
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
    submitted_witnesses: RwLock<claims::Claims>,
    /// Incremented on each new connection so stale mining threads exit.
    connection_generation: AtomicU64,
    receipt_sequence: AtomicU64,
    receipt_head: RwLock<[u8; 32]>,
    /// Submissions awaiting a pool answer, with the job each belongs to.
    pending_submission_hashes: RwLock<VecDeque<PendingSubmission>>,
    /// Explicitly selected full-search plugin for HYN3.
    accelerator: Option<Accelerator>,
    /// Records a device failure so an HYN3 mining session terminates on error.
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
            submitted_witnesses: RwLock::default(),
            connection_generation: AtomicU64::new(0),
            receipt_sequence: AtomicU64::new(0),
            receipt_head: RwLock::new([0u8; 32]),
            pending_submission_hashes: RwLock::new(VecDeque::new()),
            accelerator,
            accelerator_failed: AtomicBool::new(false),
        }
    }

    fn replace_job(&self, job: Option<Arc<MiningJob>>) {
        let mut current = self.current_job.write();
        *current = job;
        self.job_generation.fetch_add(1, Ordering::Release);
    }

    fn job_snapshot(&self) -> (Option<Arc<MiningJob>>, u64) {
        let current = self.current_job.read();
        (current.clone(), self.job_generation.load(Ordering::Acquire))
    }

    fn is_locally_claimed(&self, witness: u64) -> std::io::Result<bool> {
        self.submitted_witnesses.read().contains(witness)
    }

    fn mark_submitted(&self, witnesses: &[u64]) -> std::io::Result<()> {
        self.submitted_witnesses.write().insert(witnesses)
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
        use std::io::Write;
        let parent = std::path::Path::new(output)
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(std::path::Path::new("."));
        let sk = SecretKey::generate();
        let mut staged = tempfile::NamedTempFile::new_in(parent)?;
        staged.write_all(&sk.0)?;
        staged.as_file().sync_all()?;
        staged.persist_noclobber(output)?;
        #[cfg(unix)]
        std::fs::File::open(parent)?.sync_all()?;
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
        timings,
    }) = &cli.command
    {
        let directory = cli
            .accelerator_dir
            .as_deref()
            .ok_or("accelerator-check requires --accelerator-dir DIRECTORY")?;
        return run_accelerator_check(*batch, *offset, *stride, *timings, directory)
            .map_err(Into::into);
    }

    let cfg = match cli.network.as_str() {
        "neural-devnet" => ChainConfig::neural_devnet(),
        "testnet" => ChainConfig::testnet(),
        "mainnet" => ChainConfig::mainnet(),
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

    let detected = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let threads = if cli.threads == 0 {
        // Leave one core for the node, the pool and the tunnel this miner
        // reports to. A miner that claims every core starves them into
        // timeouts that look like provider faults.
        if detected > 2 { detected - 1 } else { detected }
    } else {
        cli.threads
    };
    let sk = if cli.key_file.is_empty() {
        let sk = SecretKey::generate();
        info!("Generated ephemeral miner key: {}", sk.public_key());
        sk
    } else {
        use std::io::Read;
        let mut file = std::fs::File::open(&cli.key_file)?;
        let mut key = SecretKey([0; 32]);
        file.read_exact(&mut key.0)?;
        if file.read(&mut [0])? != 0 {
            return Err("key file must be exactly 32 bytes".into());
        }
        key
    };

    let neural = cfg.feature_enabled(hyphen_core::FEATURE_NEURAL_WORK);
    let accelerator = cli
        .accelerator_dir
        .as_deref()
        .map(|dir| Accelerator::load(dir, neural))
        .transpose()?;
    if let Some(device) = &accelerator {
        info!(
            "Native accelerator: {} on {} via {}",
            device.backend(),
            device.device_name(),
            device.path().display()
        );
    } else {
        info!("CPU nonce search");
    }
    let threads = if neural && accelerator.is_some() {
        info!("GPU nonce search: neural score and BLAKE3 scratchpad on device");
        1
    } else {
        threads
    };
    info!(
        "Hyphen Miner starting: pool={}, threads={}, network={}",
        cli.pool, threads, cfg.network_name
    );
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
                if neural && state.accelerator_failed.load(Ordering::Acquire) {
                    return Err(e);
                }
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
    if encoded.len() > 95 {
        return Err("invalid wallet address length".into());
    }
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
    hyphen_crypto::stealth::StealthAddress {
        view_public: payload[1..33].try_into().unwrap(),
        spend_public: payload[33..65].try_into().unwrap(),
    }
    .recipient_points()
    .map_err(|error| error.to_string())?;
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
    state.replace_job(None);

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
    let ack_env = tokio::time::timeout(POOL_TIMEOUT, async {
        PoolCodec::write_envelope(&mut stream, &env).await?;
        PoolCodec::read_envelope(&mut stream).await
    })
    .await??;
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

    let session = MiningSession(state);
    let (mut share_rx, workers) = start_mining_threads(
        threads,
        Arc::clone(state),
        MiningLaunchConfig {
            chain: cfg.clone(),
            batch_size: cli_batch_size,
            connection_generation: conn_gen,
            miner_secret: sk.clone(),
        },
    )?;

    let mut reports = tokio::task::JoinSet::new();
    let compute_rate_state = Arc::clone(state);
    let rate_unit = if cfg.feature_enabled(hyphen_core::FEATURE_NEURAL_WORK) {
        "neural scores/s"
    } else {
        "operations/s"
    };
    reports.spawn(async move {
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
                "{rate_unit}: {:.2} | Shares: {} accepted, {} rejected | Blocks: {}",
                rate,
                compute_rate_state.shares_accepted.load(Ordering::Relaxed),
                compute_rate_state.shares_rejected.load(Ordering::Relaxed),
                compute_rate_state.blocks_found.load(Ordering::Relaxed),
            );
        }
    });

    let keepalive_sk = sk.clone();
    let (mut read_half, write_half) = stream.into_split();
    let write_half = Arc::new(tokio::sync::Mutex::new(write_half));

    let writer = Arc::clone(&write_half);
    let ka_sk = keepalive_sk.clone();
    reports.spawn(async move {
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
    reports.spawn(async move {
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

    let loop_result: Result<(), Box<dyn std::error::Error>> = async {
        loop {
            // Outgoing shares may arrive between any two bytes of a pool frame.
            let mut incoming = std::pin::pin!(tokio::time::timeout(
                POOL_TIMEOUT, PoolCodec::read_envelope(&mut read_half)
            ));
            loop {
            let (unanswered, oldest) = {
                let pending = state.pending_submission_hashes.read();
                (pending.len(), pending.front().map(|share| share.sent_at))
            };
            tokio::select! {
                _ = async {
                    if let Some(sent_at) = oldest {
                        tokio::time::sleep_until(sent_at + POOL_TIMEOUT).await;
                    }
                }, if oldest.is_some() => return Err("pool share acknowledgement timed out".into()),
                result = &mut incoming => {
                    let env = match result {
                        Ok(r) => r?,
                        Err(_) => return Err("pool read timeout (90s with no data)".into()),
                    };
                    env.verify()?;
                    if env.sender_pubkey.as_slice() != pool_pubkey {
                        return Err("pool signer changed during the session".into());
                    }

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
                            let pending = state
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
                                    &pending.payload_hash,
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
                                // A rejection for a job the pool has already
                                // replaced is the race a multi-threaded miner
                                // cannot avoid, not a fault: the pool retired
                                // the job while this submission was in flight.
                                let superseded = state.current_job.read().as_ref()
                                    .is_none_or(|current| !Arc::ptr_eq(current, &pending.job));
                                if superseded || result.error == POOL_STALE_JOB {
                                    debug!(
                                        "Share for job {} dropped: the pool had already moved on",
                                        hex_encode(pending.job.job_id)
                                    );
                                } else {
                                    warn!("Share rejected: {}", result.error);
                                }
                            }
                            pending.complete(&result);
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
                    break;
                }

                share = share_rx.recv(), if unanswered < MAX_UNANSWERED_SHARES => {
                        let candidate = share.ok_or("mining workers stopped")??;
                        // A candidate for a job the pool has already replaced
                        // only earns a stale-job rejection. Dropping it here
                        // saves a round trip and keeps the log free of a race
                        // the miner cannot win anyway.
                        let Some(job) = state.current_job.read().clone()
                            .filter(|job| job.job_id == candidate.job_id) else {
                            continue;
                        };
                        let submission = ShareSubmission {
                            job_id: candidate.job_id.to_vec(),
                            chain_length: candidate.chain_length as u64,
                            witnesses: candidate.witnesses,
                            work_commitment: candidate.work_commitment.as_bytes().to_vec(),
                            block_authorization: candidate.block_authorization,
                        };
                        let submission_payload = submission.encode_to_vec();
                        let submission_hash = *hyphen_crypto::blake3_hash(&submission_payload).as_bytes();
                        {
                            let mut pending = state.pending_submission_hashes.write();
                            pending.push_back(PendingSubmission {
                                payload_hash: submission_hash,
                                job,
                                sent_at: tokio::time::Instant::now(),
                                block_candidate: !submission.block_authorization.is_empty(),
                            });
                        }
                        let env = PoolEnvelope::sign(
                            MSG_SUBMIT,
                            submission_payload,
                            sk,
                        );
                        let data = env.encode_to_vec();
                        tokio::time::timeout(POOL_TIMEOUT, async {
                            use tokio::io::AsyncWriteExt;
                            let mut writer = write_half.lock().await;
                            writer.write_u32(data.len() as u32).await?;
                            writer.write_all(&data).await
                        }).await??;
                }
            }
            }
        }
    }
    .await;

    drop(session);
    drop(share_rx);
    reports.shutdown().await;
    tokio::task::spawn_blocking(move || {
        let mut panicked = false;
        for worker in workers {
            panicked |= worker.join().is_err();
        }
        if panicked {
            Err("mining worker panicked")
        } else {
            Ok(())
        }
    })
    .await??;
    loop_result
}

struct MiningSession<'a>(&'a MinerState);

impl Drop for MiningSession<'_> {
    fn drop(&mut self) {
        self.0.connection_generation.fetch_add(1, Ordering::Release);
        self.0.replace_job(None);
        self.0.pending_submission_hashes.write().clear();
    }
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
    if cfg.feature_enabled(FEATURE_ASSET_RULES_V1)
        && header.extra_nonce
            != *hyphen_crypto::blake3_hash_many(&[
                b"Hyphen/reward-recipient/v1",
                &reward_view_public,
                &reward_spend_public,
            ])
            .as_bytes()
    {
        return Err("job header does not commit to the advertised payout keys".into());
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
        candidate_in_flight: AtomicBool::new(false),
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

    // A variable-difficulty update re-sends the same job with a new share
    // threshold. It changes how much credit a submission earns, not the work
    // itself, so it must not invalidate the witnesses a worker has already
    // chained — bumping the generation here would reset every worker on every
    // VarDiff tick and the pool would never receive a block candidate.
    let replace = {
        let current = state.current_job.read();
        current
            .as_ref()
            .map(|existing| existing.job_id != job_id)
            .unwrap_or(true)
    };

    info!(
        "{}: height={}, block_difficulty={}, clean={}",
        if replace {
            "New job"
        } else {
            "Job threshold updated"
        },
        template.height,
        template.block_difficulty,
        template.clean_jobs
    );

    if replace {
        state.replace_job(Some(job));
    }

    Ok(())
}

/// Rejection the pool returns when it retired the job before the share
/// arrived. Mirrors the pool's own rejection text; a share rejected for any
/// other reason is a real fault and stays at WARN.
const POOL_STALE_JOB: &str = "stale job";

struct PendingSubmission {
    payload_hash: [u8; 32],
    sent_at: tokio::time::Instant,
    job: Arc<MiningJob>,
    block_candidate: bool,
}

impl PendingSubmission {
    fn complete(&self, result: &SubmitResult) {
        if self.block_candidate && !(result.accepted && result.block_found) {
            self.job.candidate_in_flight.store(false, Ordering::Release);
        }
    }
}

struct ShareCandidate {
    job_id: [u8; 32],
    chain_length: u32,
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
    lane: SearchLane,
}

struct MiningLaunchConfig {
    chain: ChainConfig,
    batch_size: u64,
    connection_generation: u64,
    miner_secret: SecretKey,
}

/// Width of one miner's private search lane.
///
/// Two miners that start at the same candidate find the same Cunningham
/// chains, and the pool rejects every duplicate as already claimed — the same
/// thing happens to a single miner after a restart. The lane is drawn at
/// random, so concurrent processes and restarts never rescan the same window;
/// a lane is wide enough for hundreds of blocks at the maximum difficulty.
const SEARCH_LANE_CANDIDATES: u64 = 1 << 32;

/// Largest origin whose complete chain still fits in a `u64`.
///
/// Element `k` of the chain from `n` is `n * 2^k + (2^k - 1)`. For the longest
/// supported chain the last element must stay below `u64::MAX`, which bounds
/// the origin. A lane drawn above this bound would scan numbers whose chain
/// cannot be represented, and the search would run at full speed finding
/// nothing.
fn max_chain_origin(chain_length: u32) -> u64 {
    let index = chain_length.saturating_sub(1).min(63);
    // Last element is `((n + 1) << index) - 1`, the same expression
    // `chain_element` evaluates; it must fit in a `u64`.
    let max_n_plus_one = (1u128 << 64) >> index;
    u64::try_from(max_n_plus_one.saturating_sub(1)).unwrap_or(u64::MAX)
}

/// A half-open candidate window `[start, end)` no other miner instance shares.
#[derive(Clone, Copy)]
struct SearchLane {
    start: u64,
    end: u64,
}

impl SearchLane {
    fn draw(chain_length: u32) -> Self {
        let mut bytes = [0u8; 8];
        fill_system_random(&mut bytes);
        Self::from_slot(u64::from_le_bytes(bytes), chain_length)
    }

    /// Slot `k` owns `[3 + k*w, 3 + (k+1)*w)`, with `w` even so every thread
    /// offset inside the lane stays odd: chain origins above two are odd.
    fn from_slot(slot: u64, chain_length: u32) -> Self {
        let limit = max_chain_origin(chain_length);
        let span = limit.saturating_sub(3);
        let width = (SEARCH_LANE_CANDIDATES.min(span.max(2)) / 2) * 2;
        let slots = (span / width).max(1);
        let start = 3 + (slot % slots) * width;
        Self {
            start,
            end: start.saturating_add(width).min(limit),
        }
    }

    fn cursor(self, thread_id: usize) -> u64 {
        self.start.saturating_add(2 * thread_id as u64)
    }
}

struct LaneCursor {
    next: u64,
    end: u64,
    stride: u64,
}

impl LaneCursor {
    fn new(lane: SearchLane, thread_id: usize, stride: u64) -> Self {
        Self {
            next: lane.cursor(thread_id),
            end: lane.end,
            stride,
        }
    }

    fn available(&self, batch: u64) -> u64 {
        batch.min(self.end.saturating_sub(self.next).div_ceil(self.stride))
    }

    fn advance(&mut self, scanned: u64) {
        self.next = self
            .next
            .saturating_add(self.stride.saturating_mul(scanned))
            .min(self.end);
    }
}

type ShareReceiver = tokio::sync::mpsc::Receiver<std::io::Result<ShareCandidate>>;

fn start_mining_threads(
    threads: usize,
    state: Arc<MinerState>,
    launch: MiningLaunchConfig,
) -> std::io::Result<(ShareReceiver, Vec<std::thread::JoinHandle<()>>)> {
    // The network writer handles one frame at a time. Backpressure the search
    // threads when the connection stalls instead of accumulating witness lists.
    let (tx, rx) = tokio::sync::mpsc::channel(2);
    let lane = SearchLane::draw(launch.chain.pouw_chain_length);
    info!(
        "Search lane {}..{} is private to this miner instance",
        lane.start, lane.end
    );

    let mut workers: Vec<std::thread::JoinHandle<()>> = Vec::new();
    for thread_id in 0..threads {
        let cfg = launch.chain.clone();
        let state = Arc::clone(&state);
        let tx = tx.clone();
        let miner_secret = launch.miner_secret.clone();

        let worker = std::thread::Builder::new().spawn(move || {
            if let Err(error) = mining_thread(
                MiningThreadConfig {
                    thread_id,
                    thread_count: threads,
                    chain: cfg,
                    batch_size: launch.batch_size,
                    connection_generation: launch.connection_generation,
                    miner_secret,
                    lane,
                },
                state,
                &tx,
            ) {
                let _ = tx.blocking_send(Err(error));
            }
        });
        match worker {
            Ok(handle) => workers.push(handle),
            Err(error) => {
                drop(rx);
                for handle in workers {
                    let _ = handle.join();
                }
                return Err(error);
            }
        }
    }

    Ok((rx, workers))
}

fn mining_thread(
    config: MiningThreadConfig,
    state: Arc<MinerState>,
    tx: &tokio::sync::mpsc::Sender<std::io::Result<ShareCandidate>>,
) -> std::io::Result<()> {
    let MiningThreadConfig {
        thread_id,
        thread_count,
        chain: cfg,
        batch_size,
        connection_generation: conn_gen,
        miner_secret,
        lane,
    } = config;
    let identity = ChainIdentity::of(&cfg);
    info!("Mining thread {thread_id}/{thread_count} started (batch_size={batch_size})");

    // Witnesses must be odd, so an odd per-thread offset with a stride of
    // `2 * thread_count` partitions the lane exactly once, and the lane itself
    // is private to this process.
    let stride = 2 * thread_count as u64;
    let mut cursor = LaneCursor::new(lane, thread_id, stride);
    // Witnesses chained for the job in flight, and whether its credit share
    // has already been handed to the pool.
    let mut accumulated: Vec<u64> = Vec::new();
    let mut accumulated_job: Option<[u8; 32]> = None;
    let mut credit_sent = false;

    loop {
        if tx.is_closed() || !state.running.load(Ordering::Relaxed) {
            break;
        }
        if state.connection_generation.load(Ordering::Acquire) != conn_gen {
            break;
        }
        if cursor.next >= cursor.end {
            let lane = SearchLane::draw(cfg.pouw_chain_length);
            cursor = LaneCursor::new(lane, thread_id, stride);
            info!(
                "Thread {thread_id}: lane exhausted, moving to {}..{}",
                lane.start, lane.end
            );
        }

        let (job, generation) = state.job_snapshot();
        let job = match job {
            Some(job) => job,
            None => {
                std::thread::sleep(std::time::Duration::from_millis(50));
                continue;
            }
        };

        if job.candidate_in_flight.load(Ordering::Acquire) {
            std::thread::sleep(std::time::Duration::from_millis(200));
            continue;
        }
        if cfg.feature_enabled(hyphen_core::FEATURE_NEURAL_WORK) {
            if let Err(error) = mine_neural_block(
                &cfg,
                &state,
                &job,
                generation,
                thread_id,
                thread_count,
                batch_size,
                tx,
                &miner_secret,
            ) {
                if state.accelerator_failed.load(Ordering::Acquire) {
                    return Err(std::io::Error::other(error));
                }
                error!("Thread {thread_id}: {error}");
            }
            continue;
        }

        let required = job.header.difficulty;
        let credit_at = state
            .current_share_difficulty
            .load(Ordering::Acquire)
            .clamp(1, required);

        // Witnesses already chained for this job. They survive a
        // share-difficulty update: a VarDiff tick only changes how much credit
        // a submission earns, and discarding the list here is what let a jumpy
        // controller starve the pool of block candidates.
        if accumulated_job != Some(job.job_id) {
            accumulated.clear();
            accumulated_job = Some(job.job_id);
            credit_sent = false;
        }

        if (accumulated.len() as u64) < credit_at {
            let need = credit_at - accumulated.len() as u64;
            let credit = search_witnesses(
                &state,
                generation,
                job.chain_length,
                need,
                batch_size,
                &mut cursor,
            )?;
            if credit.job_changed {
                continue;
            }
            accumulated.extend(credit.witnesses);
            accumulated.sort_unstable();
            accumulated.dedup();
        }

        // One credit share per job: it has to reach the threshold that is
        // active on the pool right now, and the block candidate reuses the same
        // witnesses afterwards.
        if !credit_sent && credit_at < required && (accumulated.len() as u64) >= credit_at {
            // Re-check the generation after the search: the pool may have
            // moved on while this worker was chaining witnesses, and a
            // submission for a retired job only earns a stale-job rejection.
            if state.job_generation.load(Ordering::Acquire) != generation {
                continue;
            }
            let credit_list: Vec<u64> = accumulated[..credit_at as usize].to_vec();
            let work_commitment = hyphen_pow::vms::witness_commitment(
                &job.parent_work_commitment,
                job.header.height,
                &credit_list,
            );
            state.mark_submitted(&credit_list)?;
            credit_sent = true;
            info!("Thread {thread_id}: credit share with {credit_at} witnesses");
            if tx
                .blocking_send(Ok(ShareCandidate {
                    job_id: job.job_id,
                    chain_length: job.chain_length,
                    witnesses: credit_list,
                    work_commitment,
                    block_authorization: Vec::new(),
                }))
                .is_err()
            {
                break;
            }
        }

        let remaining = required.saturating_sub(accumulated.len() as u64);
        if remaining > 0 {
            let rest = search_witnesses(
                &state,
                generation,
                job.chain_length,
                remaining,
                batch_size,
                &mut cursor,
            )?;
            if rest.job_changed {
                continue;
            }
            accumulated.extend(rest.witnesses);
            accumulated.sort_unstable();
            accumulated.dedup();
        }
        if (accumulated.len() as u64) < required {
            continue;
        }
        let witnesses = std::mem::take(&mut accumulated);
        accumulated_job = None;
        credit_sent = false;

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

        if state.job_generation.load(Ordering::Acquire) != generation
            || job.candidate_in_flight.swap(true, Ordering::AcqRel)
        {
            continue;
        }

        state.mark_submitted(&witnesses)?;
        info!(
            "Thread {thread_id}: block candidate height={} witnesses={}",
            job.header.height,
            witnesses.len()
        );
        if tx
            .blocking_send(Ok(ShareCandidate {
                job_id: job.job_id,
                chain_length: job.chain_length,
                witnesses,
                work_commitment,
                block_authorization,
            }))
            .is_err()
        {
            break;
        }
    }

    info!("Mining thread {thread_id} stopped");
    Ok(())
}

/// Searches nonce-derived neural inputs and applies the memory gate to finalists.
///
/// Each thread walks its own counter residue class for the parent, height and miner key.
#[allow(clippy::too_many_arguments)]
fn mine_neural_block(
    cfg: &ChainConfig,
    state: &Arc<MinerState>,
    job: &MiningJob,
    generation: u64,
    thread_id: usize,
    thread_count: usize,
    batch_size: u64,
    tx: &tokio::sync::mpsc::Sender<std::io::Result<ShareCandidate>>,
    miner_secret: &hyphen_crypto::SecretKey,
) -> Result<(), String> {
    let identity = ChainIdentity::of(cfg);
    let spec = hyphen_neural::PROFILE_SPEC;
    let miner_pubkey = *miner_secret.public_key().as_bytes();
    let parent_hash = job.header.prev_hash;
    let gpu_failure = |error: String| {
        state.accelerator_failed.store(true, Ordering::Release);
        format!("GPU mining failed: {error}")
    };
    let mut descriptor = hyphen_neural::WorkDescriptor {
        epoch: job.header.height,
        spec,
        threshold: job.header.difficulty,
        chain_seed: [0; 32],
    };
    descriptor
        .validate()
        .map_err(|error| format!("pool sent an unusable work descriptor: {error}"))?;

    let weights = if let Some(device) = &state.accelerator {
        let (seed, weights) = device
            .prepare_job(&parent_hash, descriptor.epoch, &miner_pubkey)
            .map_err(gpu_failure)?;
        descriptor.chain_seed = seed;
        weights
    } else {
        descriptor.chain_seed =
            *hyphen_core::neural_work_seed(&parent_hash, descriptor.epoch, &miner_pubkey)
                .as_bytes();
        let mut weights = vec![0; spec.weight_count()];
        descriptor
            .derive_weights(&mut weights)
            .map_err(|error| format!("cannot derive work weights: {error}"))?;
        weights
    };
    let stride = thread_count.max(1) as u64;
    let mut counter = thread_id as u64;
    let mut attempts: u64 = 0;
    let mut backend = "cpu";
    let started = std::time::Instant::now();
    let nonce = loop {
        // `generation` moves whenever the job is replaced, and the outer loop
        // owns the connection check, so this is the whole stop condition. A
        // sibling worker that already solved this job stops us from spending a
        // second solution the pool would reject as a stale job.
        if !state.running.load(Ordering::Relaxed)
            || state.job_generation.load(Ordering::Acquire) != generation
            || job.candidate_in_flight.load(Ordering::Acquire)
        {
            return Ok(());
        }
        if let Some(device) = &state.accelerator {
            let remaining = (u64::MAX - counter) / stride;
            // Smaller windows avoid unnecessary work at easier thresholds.
            let expected = descriptor.expected_attempts().saturating_mul(16);
            let limit = expected.clamp(1024, u64::from(accelerator::NEURAL_WINDOW));
            let window = remaining.saturating_add(1).min(limit).min(batch_size) as u32;
            let winners = device
                .neural_winners(&descriptor, &weights, counter, stride, window)
                .map_err(gpu_failure)?;
            attempts = attempts.saturating_add(u64::from(window));
            state
                .total_operations
                .fetch_add(u64::from(window), Ordering::Relaxed);
            if let Some(&index) = winners.first() {
                backend = device.backend();
                break counter + u64::from(index) * stride;
            }
            if remaining < u64::from(window) {
                return Ok(());
            }
            counter += stride * u64::from(window);
            continue;
        }
        let score = hyphen_neural::block_work::nonce_score(&descriptor, &weights, counter)
            .map_err(|e| e.to_string())?;
        attempts = attempts.saturating_add(1);
        state.total_operations.fetch_add(1, Ordering::Relaxed);
        if hyphen_neural::block_work::qualifies(&descriptor, counter, score)
            .map_err(|e| e.to_string())?
        {
            break counter;
        }
        let Some(next) = counter.checked_add(stride) else {
            return Ok(());
        };
        counter = next;
    };

    let witnesses = vec![nonce];
    let commitment = if let Some(device) = &state.accelerator {
        device
            .work_commitment(
                &parent_hash,
                descriptor.epoch,
                descriptor.threshold,
                &miner_pubkey,
                nonce,
            )
            .map_err(gpu_failure)?
    } else {
        hyphen_core::neural_work_commitment(
            &parent_hash,
            job.header.height,
            job.header.difficulty,
            &miner_pubkey,
            &witnesses,
        )
    };

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

    if state.job_generation.load(Ordering::Acquire) != generation
        || job.candidate_in_flight.swap(true, Ordering::AcqRel)
    {
        return Ok(());
    }
    info!(
        "Thread {thread_id}: neural solution at height {} after {attempts} scores (backend={backend}, search_us={})",
        job.header.height,
        started.elapsed().as_micros()
    );
    tx.blocking_send(Ok(ShareCandidate {
        job_id: job.job_id,
        chain_length: job.chain_length,
        witnesses,
        work_commitment: commitment,
        block_authorization,
    }))
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
    timings: bool,
    accelerator_dir: &std::path::Path,
) -> Result<(), String> {
    if !(1..=accelerator::NEURAL_WINDOW).contains(&batch) {
        return Err(format!(
            "batch must be within 1..={}",
            accelerator::NEURAL_WINDOW
        ));
    }
    if stride == 0 {
        return Err("stride must be non-zero".into());
    }
    if u64::from(batch - 1) > (u64::MAX - offset) / stride {
        return Err("nonce window overflows u64".into());
    }
    let cfg = ChainConfig::neural_devnet();
    let genesis = hyphen_core::build_genesis_block(&cfg).hash();
    let seed = hyphen_core::neural_work_seed(&genesis, 1, &[19; 32]);
    let work = hyphen_neural::WorkDescriptor {
        spec: hyphen_neural::PROFILE_SPEC,
        epoch: 1,
        threshold: cfg.neural_work_initial_threshold,
        chain_seed: *seed.as_bytes(),
    };
    let mut weights = vec![0; work.spec.weight_count()];
    work.derive_weights(&mut weights)
        .map_err(|e| e.to_string())?;
    let device = Accelerator::load(accelerator_dir, true)?;
    let (gpu_seed, gpu_weights) = device.prepare_job(&genesis, 1, &[19; 32])?;
    if gpu_seed != work.chain_seed || gpu_weights != weights {
        return Err("GPU and Rust job descriptors differ".into());
    }
    let device_start = std::time::Instant::now();
    let scores = device.neural_scores(&work, &weights, offset, stride, batch)?;
    let device_scoring = device_start.elapsed();
    let search_start = std::time::Instant::now();
    let device_winners = device.neural_winners(&work, &gpu_weights, offset, stride, batch)?;
    let device_total = search_start.elapsed();
    let mut winners = 0;
    let cpu_start = std::time::Instant::now();
    for (i, device_score) in scores.into_iter().enumerate() {
        let nonce = offset + i as u64 * stride;
        let score = hyphen_neural::block_work::nonce_score(&work, &weights, nonce)
            .map_err(|e| e.to_string())?;
        if device_score != score {
            return Err(format!(
                "nonce {nonce}: device score {device_score} differs from Rust score {score}"
            ));
        }
        if hyphen_neural::block_work::qualifies(&work, nonce, score).map_err(|e| e.to_string())? {
            if device_winners.get(winners) != Some(&(i as u32)) {
                return Err(format!("GPU and Rust winner lists differ at nonce {nonce}"));
            }
            winners += 1;
        }
    }
    let cpu_total = cpu_start.elapsed();
    if winners != device_winners.len() {
        return Err("accelerator and CPU winner counts differ".into());
    }
    if let Some(&index) = device_winners.first() {
        let nonce = offset + u64::from(index) * stride;
        let gpu = device.work_commitment(&genesis, 1, work.threshold, &[19; 32], nonce)?;
        if gpu
            != hyphen_core::neural_work_commitment(&genesis, 1, work.threshold, &[19; 32], &[nonce])
        {
            return Err("GPU and Rust work commitments differ".into());
        }
    }
    if timings {
        println!(
            "device_score_ns={} gpu_search_ns={} cpu_search_ns={}",
            device_scoring.as_nanos(),
            device_total.as_nanos(),
            cpu_total.as_nanos()
        );
    }
    println!(
        "backend={} device={} seed={seed} offset={offset} stride={stride} scores={batch} accepted={winners}",
        device.backend(),
        device.device_name()
    );
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
    cursor: &mut LaneCursor,
) -> std::io::Result<WitnessSearch> {
    let mut witnesses: Vec<u64> = Vec::with_capacity(count as usize);
    while (witnesses.len() as u64) < count {
        if state.job_generation.load(Ordering::Acquire) != generation {
            return Ok(WitnessSearch {
                witnesses,
                job_changed: true,
            });
        }

        let available = cursor.available(batch);
        if available == 0 {
            break;
        }
        let read_error = std::cell::Cell::new(None);
        let is_claimed = |candidate: u64| {
            state.is_locally_claimed(candidate).unwrap_or_else(|error| {
                read_error.set(Some(error));
                true
            })
        };
        let params = hyphen_pow::vms::SearchParams {
            length: chain_length,
            offset: cursor.next,
            stride: cursor.stride,
            count: count - witnesses.len() as u64,
            max_candidates: available,
            is_claimed: &is_claimed,
        };
        let outcome = match hyphen_pow::vms::search(&params) {
            Ok(outcome) => outcome,
            Err(error) => {
                error!("witness search rejected: {error}");
                return Ok(WitnessSearch {
                    witnesses,
                    job_changed: true,
                });
            }
        };
        if let Some(error) = read_error.take() {
            return Err(error);
        }
        state
            .total_operations
            .fetch_add(outcome.candidates_tested, Ordering::Relaxed);
        witnesses.extend(outcome.witnesses);
        cursor.advance(outcome.candidates_tested);
        if outcome.candidates_tested == 0 {
            break;
        }
    }
    Ok(WitnessSearch {
        witnesses,
        job_changed: false,
    })
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
        let (_, _, keys) = hyphen_crypto::stealth::generate_keys();
        payload.extend_from_slice(&keys.view_public);
        payload.extend_from_slice(&keys.spend_public);
        let hash = hyphen_crypto::blake3_hash(&payload);
        payload.extend_from_slice(&hash.as_bytes()[..4]);
        format!("hy1{}", hyphen_crypto::base58::encode(&payload))
    }

    #[test]
    fn a_worker_finishing_an_old_job_cannot_claim_its_successor() {
        let state = MinerState::new();
        let job = |id| {
            Arc::new(MiningJob {
                job_id: [id; 32],
                header: hyphen_core::genesis::build_genesis_block(
                    &ChainConfig::archived_neural_devnet_v1(),
                )
                .header,
                candidate_in_flight: AtomicBool::new(false),
                chain_length: 0,
                parent_work_commitment: Hash256::ZERO,
                reward_view_public: [1; 32],
                reward_spend_public: [2; 32],
            })
        };
        state.replace_job(Some(job(1)));
        let (old, first) = state.job_snapshot();
        state.replace_job(Some(job(2)));
        let (current, second) = state.job_snapshot();
        assert_ne!(first, second);
        let old = old.unwrap();
        old.candidate_in_flight.store(true, Ordering::Release);
        let current = current.unwrap();
        assert_eq!(current.job_id, [2; 32]);
        assert!(!current.candidate_in_flight.load(Ordering::Acquire));
        current.candidate_in_flight.store(true, Ordering::Release);
        let mut pending = PendingSubmission {
            payload_hash: [0; 32],
            sent_at: tokio::time::Instant::now(),
            job: old,
            block_candidate: false,
        };
        pending.complete(&SubmitResult::default());
        assert!(pending.job.candidate_in_flight.load(Ordering::Acquire));
        pending.block_candidate = true;
        pending.complete(&SubmitResult {
            accepted: true,
            block_found: true,
            ..Default::default()
        });
        assert!(pending.job.candidate_in_flight.load(Ordering::Acquire));
        pending.complete(&SubmitResult::default());
        assert!(!pending.job.candidate_in_flight.load(Ordering::Acquire));
        assert!(current.candidate_in_flight.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn disconnect_releases_every_worker_before_reconnecting() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = listener.local_addr().unwrap().to_string();
        let cfg = ChainConfig::archived_devnet_v2();
        let identity = ChainIdentity::of(&cfg);
        let ack = LoginAck {
            accepted: true,
            network_name: cfg.network_name.to_string(),
            protocol_version: POOL_PROTOCOL_VERSION,
            network_magic: cfg.network_magic.to_vec(),
            consensus_params_hash: identity.consensus_params_hash.to_vec(),
            genesis_hash: identity.genesis_hash.to_vec(),
            ..Default::default()
        };
        let state = Arc::new(MinerState::new());
        let pool_key = SecretKey::generate();
        let miner_key = SecretKey::generate();
        for _ in 0..64 {
            let pool = async {
                let (mut stream, _) = listener.accept().await.unwrap();
                let login = PoolCodec::read_envelope(&mut stream).await.unwrap();
                assert_eq!(login.msg_type, MSG_LOGIN);
                let envelope = PoolEnvelope::sign(MSG_LOGIN_ACK, ack.encode_to_vec(), &pool_key);
                PoolCodec::write_envelope(&mut stream, &envelope)
                    .await
                    .unwrap();
                let report = PoolCodec::read_envelope(&mut stream).await.unwrap();
                assert!(matches!(
                    report.msg_type,
                    MSG_KEEPALIVE | MSG_COMPUTE_RATE_REPORT
                ));
            };
            let client = connect_and_mine(
                &endpoint, None, &miner_key, "test", &cfg, 2, &state, 1000, &[1; 64], false,
            );
            let ((), result) = tokio::join!(pool, client);
            assert!(result.is_err());
            assert_eq!(Arc::strong_count(&state), 1);
        }
    }

    #[test]
    fn wallet_address_parser_returns_errors_instead_of_panicking() {
        let cfg = ChainConfig::archived_devnet_v2();
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
    fn partial_batches_stop_at_the_lane_end_without_wrapping() {
        let lane = SearchLane {
            start: u64::MAX - 18,
            end: u64::MAX,
        };
        let mut first = LaneCursor::new(lane, 0, 8);
        assert_eq!(first.available(100), 3);
        first.advance(2);
        assert_eq!(first.next, u64::MAX - 2);
        assert_eq!(first.available(100), 1);
        first.advance(1);
        assert_eq!(first.next, lane.end);
        assert_eq!(first.available(100), 0);
        let mut second = LaneCursor::new(lane, 1, 8);
        assert_eq!(second.available(100), 2);
        second.advance(2);
        assert_eq!(second.available(1), 0);
    }

    #[test]
    fn search_lanes_are_disjoint_odd_and_partitioned_per_thread() {
        let length = ChainConfig::archived_devnet_v2().pouw_chain_length;
        let first = SearchLane::from_slot(0, length);
        let second = SearchLane::from_slot(1, length);
        assert_eq!(first.start % 2, 1, "chain origins must be odd");
        assert_eq!(second.start % 2, 1, "chain origins must be odd");
        assert_eq!(
            second.start, first.end,
            "consecutive slots must not overlap or leave a gap"
        );

        let drawn = SearchLane::draw(length);
        assert_eq!(drawn.start % 2, 1);
        assert!(drawn.end > drawn.start);

        // Threads inside one process take disjoint residue classes and never
        // leave the lane.
        let thread_count = 4usize;
        let stride = 2 * thread_count as u64;
        let mut residues = std::collections::HashSet::new();
        for thread_id in 0..thread_count {
            let cursor = first.cursor(thread_id);
            assert_eq!(cursor % 2, 1);
            assert!(cursor < first.end);
            assert!(
                residues.insert(cursor % stride),
                "threads must not share candidates"
            );
        }
    }

    /// A lane drawn above the representable chain range would scan at full
    /// speed finding nothing: `n * 2^k + (2^k - 1)` must stay below
    /// `u64::MAX`, which bounds the origin.
    #[test]
    fn every_lane_stays_inside_the_representable_chain_range() {
        for length in hyphen_pow::vms::MIN_CHAIN_LENGTH..=hyphen_pow::vms::MAX_CHAIN_LENGTH {
            let limit = max_chain_origin(length);
            for slot in [0u64, 1, 7, u64::MAX] {
                let lane = SearchLane::from_slot(slot, length);
                assert!(lane.end > lane.start);
                assert_eq!(lane.start % 2, 1);
                assert!(
                    lane.start.saturating_add(2) <= limit,
                    "lane start for length {length} is outside the representable range"
                );
            }
            assert!(
                hyphen_pow::vms::chain_element(limit, length - 1).is_some(),
                "the largest lane origin must still produce a representable chain"
            );
            if let Some(next) = limit.checked_add(1) {
                assert!(
                    hyphen_pow::vms::chain_element(next, length - 1).is_none(),
                    "the bound must be tight at length {length}"
                );
            }
        }
    }

    #[test]
    fn witness_search_returns_verifiable_single_use_chains() {
        let cfg = ChainConfig::archived_devnet_v2();
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
        assert!(!state.is_locally_claimed(3).unwrap());
        state.mark_submitted(&[3, 11]).unwrap();
        assert!(state.is_locally_claimed(3).unwrap());
        assert!(state.is_locally_claimed(11).unwrap());
        assert!(!state.is_locally_claimed(5).unwrap());
    }
}
