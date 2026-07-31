mod accelerator;
mod pow;
mod primitives;
mod protocol;

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use clap::{Parser, Subcommand};
use parking_lot::RwLock;
use prost::Message;
use tokio::net::TcpStream;
use tracing::{error, info, warn};

use accelerator::{AcceleratorRegistry, BackendStatus};
use pow::{difficulty_to_target, evaluate_pow_with_epoch, EpochArena, EpochKernelParams};
use primitives::{
    BlockAuthorization, BlockHeader, ChainConfig, Hash256, SecretKey, FROZEN_BLOCK_VERSION,
};
use protocol::*;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct ArenaParamsData {
    pub arena_size: u64,
    pub page_size: u64,
}

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

    #[arg(long, default_value = "0")]
    threads: usize,

    #[arg(long, value_parser = ["mainnet", "testnet", "devnet"], default_value = "devnet")]
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

    /// Directory containing hyphen_backend_* native accelerator plugins.
    #[arg(long, default_value = "accelerators")]
    accelerator_dir: PathBuf,

    /// Refuse to start unless a device passes execution and independent CPU verification.
    #[arg(long, default_value_t = false)]
    require_accelerator: bool,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Generate a new Ed25519 key pair and save to file
    Keygen {
        /// Output file path for the 32-byte secret key
        #[arg(long, default_value = "miner.key")]
        output: String,
    },
    /// Enumerate plugins and run deterministic device/CPU cross-checks.
    Accelerators,
    /// Execute the deterministic diffusion PDE kernel and verify the complete device result.
    ScientificRun {
        /// Little-endian i32 input cells. Each cell must be in 0..=262143.
        #[arg(long)]
        input: PathBuf,
        /// Destination for the verified little-endian i32 result.
        #[arg(long)]
        output: PathBuf,
        /// Q12 diffusion coefficient in 0..=2048.
        #[arg(long, default_value_t = 512)]
        alpha_q12: u32,
        /// Number of deterministic evolution steps in 1..=1024.
        #[arg(long, default_value_t = 1)]
        iterations: u32,
        /// Restrict execution to a backend such as nvidia-cuda or intel-openvino.
        #[arg(long, default_value = "")]
        backend: String,
    },
}

struct MiningJob {
    job_id: [u8; 32],
    header: BlockHeader,
    #[allow(dead_code)]
    share_difficulty: u64,
    epoch_seed: Hash256,
    arena_size: usize,
    page_size: usize,
    reward_view_public: [u8; 32],
    reward_spend_public: [u8; 32],
}

struct MinerState {
    current_job: RwLock<Option<Arc<MiningJob>>>,
    job_generation: AtomicU64,
    current_share_difficulty: AtomicU64,
    difficulty_generation: AtomicU64,
    estimated_hashrate: AtomicU64,
    total_hashes: AtomicU64,
    shares_accepted: AtomicU64,
    shares_rejected: AtomicU64,
    blocks_found: AtomicU64,
    running: AtomicBool,
    shared_arena: RwLock<Option<(Hash256, Arc<EpochArena>)>>,
    /// Incremented on each new connection so stale mining threads exit.
    connection_generation: AtomicU64,
    receipt_sequence: AtomicU64,
    receipt_head: RwLock<[u8; 32]>,
    pending_submission_hashes: RwLock<VecDeque<[u8; 32]>>,
}

impl MinerState {
    fn new() -> Self {
        Self {
            current_job: RwLock::new(None),
            job_generation: AtomicU64::new(0),
            current_share_difficulty: AtomicU64::new(100),
            difficulty_generation: AtomicU64::new(0),
            estimated_hashrate: AtomicU64::new(0),
            total_hashes: AtomicU64::new(0),
            shares_accepted: AtomicU64::new(0),
            shares_rejected: AtomicU64::new(0),
            blocks_found: AtomicU64::new(0),
            running: AtomicBool::new(true),
            shared_arena: RwLock::new(None),
            connection_generation: AtomicU64::new(0),
            receipt_sequence: AtomicU64::new(0),
            receipt_head: RwLock::new([0u8; 32]),
            pending_submission_hashes: RwLock::new(VecDeque::new()),
        }
    }

    fn get_arena(
        &self,
        epoch_seed: Hash256,
        arena_size: usize,
        page_size: usize,
    ) -> Arc<EpochArena> {
        {
            let guard = self.shared_arena.read();
            if let Some((seed, ref a)) = *guard {
                if seed == epoch_seed {
                    return Arc::clone(a);
                }
            }
        }
        let mut guard = self.shared_arena.write();
        if let Some((seed, ref a)) = *guard {
            if seed == epoch_seed {
                return Arc::clone(a);
            }
        }
        info!(
            "Generating epoch arena ({}MiB) for seed {}...",
            arena_size / (1024 * 1024),
            epoch_seed
        );
        let arena = Arc::new(EpochArena::generate(epoch_seed, arena_size, page_size));
        *guard = Some((epoch_seed, Arc::clone(&arena)));
        arena
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

    if matches!(cli.command, Some(Commands::Accelerators)) {
        let registry = AcceleratorRegistry::discover(&cli.accelerator_dir);
        print_accelerator_reports(&registry);
        if cli.require_accelerator && registry.available_devices() == 0 {
            return Err("no accelerator passed the deterministic device self-test".into());
        }
        return Ok(());
    }

    if let Some(Commands::ScientificRun {
        input,
        output,
        alpha_q12,
        iterations,
        backend,
    }) = &cli.command
    {
        let input_bytes = std::fs::read(input)?;
        if !input_bytes.len().is_multiple_of(std::mem::size_of::<i32>()) {
            return Err("scientific input length must be a multiple of four bytes".into());
        }
        let cells = input_bytes
            .chunks_exact(std::mem::size_of::<i32>())
            .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("exact chunk size")))
            .collect::<Vec<_>>();
        let registry = AcceleratorRegistry::discover(&cli.accelerator_dir);
        let execution = registry.execute_verified(
            (!backend.is_empty()).then_some(backend.as_str()),
            &cells,
            *alpha_q12,
            *iterations,
        )?;
        let mut output_bytes =
            Vec::with_capacity(std::mem::size_of_val(execution.output.as_slice()));
        for value in &execution.output {
            output_bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(output, output_bytes)?;
        println!("Verified scientific device execution:");
        println!("  backend          : {}", execution.backend);
        println!("  device           : {}", execution.device);
        println!("  stable device ID : {}", execution.stable_id);
        println!("  input commitment : {}", execution.input_commitment);
        println!("  output commitment: {}", execution.output_commitment);
        println!("  operation count  : {}", execution.operation_count);
        println!("  device time (ns) : {}", execution.device_time_ns);
        println!("  verified output  : {}", output.display());
        return Ok(());
    }

    let cfg = match cli.network.as_str() {
        "mainnet" => ChainConfig::mainnet(),
        "testnet" => ChainConfig::testnet(),
        "devnet" => ChainConfig::devnet(),
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

    let accelerators = AcceleratorRegistry::discover(&cli.accelerator_dir);
    log_accelerator_reports(&accelerators);
    if cli.require_accelerator && accelerators.available_devices() == 0 {
        return Err("no accelerator passed the deterministic device self-test".into());
    }

    let state = Arc::new(MinerState::new());

    let wallet_pubkey = parse_wallet_address(&cli.wallet_address, &cfg)?;
    info!("Payout wallet: {}", hex::encode(wallet_pubkey));

    let mut backoff_secs = 5u64;
    loop {
        if !state.running.load(Ordering::Relaxed) {
            break;
        }

        let conn_start = std::time::Instant::now();

        match connect_and_mine(
            &cli.pool,
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

    Ok(())
}

fn print_accelerator_reports(registry: &AcceleratorRegistry) {
    println!(
        "Native accelerator backends: {} loaded, {} verified devices",
        registry.loaded_backend_count(),
        registry.available_devices()
    );
    for report in registry.reports() {
        let device = if report.name.is_empty() {
            "-".to_string()
        } else {
            format!("{} ({})", report.name, report.stable_id)
        };
        let runtime = if report.runtime.is_empty() {
            "-"
        } else {
            &report.runtime
        };
        println!(
            "  {:<18} {:<24} runtime={:<16} {}",
            report.backend, device, runtime, report.status
        );
        if let (Some(operations), Some(device_time_ns)) =
            (report.operation_count, report.device_time_ns)
        {
            println!(
                "    verified operations={operations}, device_time_ns={device_time_ns}, vendor={}",
                report.vendor
            );
        }
    }
}

fn log_accelerator_reports(registry: &AcceleratorRegistry) {
    for report in registry.reports() {
        match &report.status {
            BackendStatus::Available => info!(
                "accelerator verified: backend={}, device={}, stable_id={}, runtime={}, operations={}, device_time_ns={}",
                report.backend,
                report.name,
                report.stable_id,
                report.runtime,
                report.operation_count.unwrap_or(0),
                report.device_time_ns.unwrap_or(0)
            ),
            BackendStatus::Unavailable(reason) => {
                info!("accelerator unavailable: backend={}, reason={reason}", report.backend)
            }
            BackendStatus::SelfTestFailed(reason) => warn!(
                "accelerator self-test failed: backend={}, device={}, reason={reason}",
                report.backend, report.name
            ),
        }
    }
}

fn parse_wallet_address(address: &str, cfg: &ChainConfig) -> Result<[u8; 64], String> {
    let encoded = address.strip_prefix("hy1").ok_or_else(|| {
        "pool protocol v3 requires --wallet-address with a hy1... address".to_string()
    })?;
    let payload = bs58::decode(encoded)
        .into_vec()
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
    let hash = blake3::hash(&payload[..65]);
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

    let mut stream = TcpStream::connect(pool_addr).await?;

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

    let estimated_hashrate = state
        .estimated_hashrate
        .load(Ordering::Acquire)
        .max((threads as u64).saturating_mul(128));

    let login = LoginRequest {
        miner_id: hex::encode(sk.public_key().as_bytes()),
        user_agent: user_agent.to_string(),
        payout_pubkey: wallet_pubkey.to_vec(),
        estimated_hashrate,
        thread_count: threads as u32,
        network_magic: cfg.network_magic.to_vec(),
        protocol_version: POOL_PROTOCOL_VERSION,
        consensus_params_hash: cfg.consensus_params_hash().to_vec(),
        genesis_hash: cfg.genesis_hash().to_vec(),
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
        || ack.consensus_params_hash.as_slice() != cfg.consensus_params_hash()
        || ack.genesis_hash.as_slice() != cfg.genesis_hash()
    {
        return Err(format!(
            "pool network mismatch: expected {} ({}) v{}, got {} ({}) v{}",
            cfg.network_name,
            hex::encode(cfg.network_magic),
            POOL_PROTOCOL_VERSION,
            ack.network_name,
            hex::encode(&ack.network_magic),
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
        hex::encode(&ack.chain_tip_hash),
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
        cfg_clone.clone(),
        state_clone,
        cli_batch_size,
        conn_gen,
        sk.clone(),
    );

    let hashrate_state = Arc::clone(state);
    let hashrate_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(10));
        let mut last_hashes: u64 = 0;
        loop {
            interval.tick().await;
            let current = hashrate_state.total_hashes.load(Ordering::Relaxed);
            let delta = current.saturating_sub(last_hashes);
            last_hashes = current;
            let rate = delta as f64 / 10.0;
            hashrate_state
                .estimated_hashrate
                .store(rate.round() as u64, Ordering::Release);
            info!(
                "Hashrate: {:.2} H/s | Shares: {} accepted, {} rejected | Blocks: {}",
                rate,
                hashrate_state.shares_accepted.load(Ordering::Relaxed),
                hashrate_state.shares_rejected.load(Ordering::Relaxed),
                hashrate_state.blocks_found.load(Ordering::Relaxed),
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
    let hashrate_report_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(5));
        let mut last_hashes: u64 = 0;
        loop {
            interval.tick().await;
            let current = report_state.total_hashes.load(Ordering::Relaxed);
            let delta = current.saturating_sub(last_hashes);
            last_hashes = current;
            let rate = delta / 5;
            report_state
                .estimated_hashrate
                .store(rate, Ordering::Release);
            let report = HashrateReport {
                hashrate: rate,
                total_hashes: current,
                uptime_secs: start_time.elapsed().as_secs(),
            };
            let env = PoolEnvelope::sign(MSG_HASHRATE_REPORT, report.encode_to_vec(), &report_sk);
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
                                        hex::encode(&result.block_hash)
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
                                hex::encode(&notify.finder_pubkey)
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
                                hex::encode(&chain_info.tip_hash),
                            );
                        }
                        _ => {}
                    }
                }

                share = share_rx.recv() => {
                    if let Some((nonce, extra_nonce, pow_hash, job_id, block_authorization)) = share {
                        let submission = ShareSubmission {
                            job_id: job_id.to_vec(),
                            nonce,
                            extra_nonce: extra_nonce.to_vec(),
                            pow_hash: pow_hash.as_bytes().to_vec(),
                            block_authorization,
                        };
                        let submission_payload = submission.encode_to_vec();
                        let submission_hash = *primitives::blake3_hash(&submission_payload).as_bytes();
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
    hashrate_handle.abort();
    keepalive_handle.abort();
    hashrate_report_handle.abort();

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
    let header: BlockHeader = hyphen_codec::deserialize_with_limit(&template.header_data, 4096)?;

    if template.job_id.len() != 32 {
        return Err("job id must be exactly 32 bytes".into());
    }
    if template.epoch_seed.len() != 32 {
        return Err("epoch seed must be exactly 32 bytes".into());
    }
    if template.share_difficulty == 0 || template.block_difficulty == 0 {
        return Err("job difficulty must be greater than zero".into());
    }
    if template.consensus_params_hash.as_slice() != cfg.consensus_params_hash()
        || template.genesis_hash.as_slice() != cfg.genesis_hash()
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
    if !allow_shared_reward_recipient
        && (reward_view_public != payout_keys[..32] || reward_spend_public != payout_keys[32..64])
    {
        return Err(
            "pool redirected the reward address; use --allow-shared-reward-recipient only after reviewing the pool policy"
                .into(),
        );
    }
    if header.version != FROZEN_BLOCK_VERSION
        || header.height != template.height
        || header.difficulty != template.block_difficulty
        || header.epoch_seed.as_bytes() != template.epoch_seed.as_slice()
        || header.miner_pubkey != miner_pubkey
    {
        return Err("job header metadata mismatch".into());
    }
    if compute_tx_root(&template.transactions) != header.tx_root {
        return Err("job transaction list does not match the committed tx_root".into());
    }

    let mut job_id = [0u8; 32];
    job_id.copy_from_slice(&template.job_id);

    let epoch_seed = Hash256::from_bytes(template.epoch_seed.clone().try_into().unwrap());

    let (arena_size, page_size) = if template.arena_params.is_empty() {
        (cfg.arena_size, cfg.page_size)
    } else {
        let params: ArenaParamsData =
            hyphen_codec::deserialize_with_limit(&template.arena_params, 64)?;
        if params.arena_size != cfg.arena_size as u64 || params.page_size != cfg.page_size as u64 {
            return Err(format!(
                "pool sent incompatible arena parameters: {}/{} (expected {}/{})",
                params.arena_size, params.page_size, cfg.arena_size, cfg.page_size,
            )
            .into());
        }
        (params.arena_size as usize, params.page_size as usize)
    };

    let job = Arc::new(MiningJob {
        job_id,
        header,
        share_difficulty: template.share_difficulty,
        epoch_seed,
        arena_size,
        page_size,
        reward_view_public,
        reward_spend_public,
    });

    state
        .current_share_difficulty
        .store(template.share_difficulty, Ordering::Release);
    state.difficulty_generation.fetch_add(1, Ordering::AcqRel);

    let active_share_diff = state.current_share_difficulty.load(Ordering::Acquire);
    info!(
        "New job: height={}, block_diff={}, template_share_diff={}, active_share_diff={}, clean={}",
        template.height,
        template.block_difficulty,
        template.share_difficulty,
        active_share_diff,
        template.clean_jobs
    );

    *state.current_job.write() = Some(job.clone());
    state.job_generation.fetch_add(1, Ordering::Release);

    Ok(())
}

type ShareResult = (u64, [u8; 32], Hash256, [u8; 32], Vec<u8>);

struct MiningThreadConfig {
    thread_id: usize,
    thread_count: usize,
    chain: ChainConfig,
    batch_size: u64,
    connection_generation: u64,
    miner_secret: SecretKey,
}

fn start_mining_threads(
    threads: usize,
    cfg: ChainConfig,
    state: Arc<MinerState>,
    batch_size: u64,
    conn_gen: u64,
    miner_secret: SecretKey,
) -> tokio::sync::mpsc::UnboundedReceiver<ShareResult> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

    for thread_id in 0..threads {
        let cfg = cfg.clone();
        let state = Arc::clone(&state);
        let tx = tx.clone();
        let miner_secret = miner_secret.clone();

        std::thread::spawn(move || {
            mining_thread(
                MiningThreadConfig {
                    thread_id,
                    thread_count: threads,
                    chain: cfg,
                    batch_size,
                    connection_generation: conn_gen,
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
    tx: tokio::sync::mpsc::UnboundedSender<ShareResult>,
) {
    let MiningThreadConfig {
        thread_id,
        thread_count,
        chain: cfg,
        batch_size,
        connection_generation: conn_gen,
        miner_secret,
    } = config;
    info!("Mining thread {thread_id}/{thread_count} started (batch_size={batch_size})");

    #[allow(unused_assignments)]
    let mut last_gen: u64 = 0;

    loop {
        if !state.running.load(Ordering::Relaxed) {
            break;
        }
        // Exit if the connection has been replaced by a new one
        if state.connection_generation.load(Ordering::Acquire) != conn_gen {
            break;
        }

        let current_gen = state.job_generation.load(Ordering::Acquire);
        let current_diff_gen = state.difficulty_generation.load(Ordering::Acquire);

        let job = match state.current_job.read().clone() {
            Some(j) => j,
            None => {
                std::thread::sleep(std::time::Duration::from_millis(50));
                continue;
            }
        };

        last_gen = current_gen;

        let arena = state.get_arena(job.epoch_seed, job.arena_size, job.page_size);

        let share_diff = state.current_share_difficulty.load(Ordering::Acquire);
        let share_target = difficulty_to_target(share_diff);
        let block_target = difficulty_to_target(job.header.difficulty);

        // Pre-compute epoch kernel params once per job (same epoch → same params)
        let epoch = EpochKernelParams::derive(arena.params.epoch_seed.as_bytes());

        let base_nonce: u64 =
            rand::random::<u64>() / thread_count as u64 * thread_count as u64 + thread_id as u64;

        let mut extra_nonce: [u8; 32] = rand::random();
        extra_nonce[0] = thread_id as u8;

        let mut candidate = job.header.clone();
        candidate.extra_nonce = extra_nonce;

        for i in 0..batch_size {
            if i & 0xFF == 0
                && (state.job_generation.load(Ordering::Acquire) != last_gen
                    || state.difficulty_generation.load(Ordering::Acquire) != current_diff_gen)
            {
                break;
            }

            let nonce = base_nonce.wrapping_add(i * thread_count as u64);
            candidate.nonce = nonce;

            let hash = evaluate_pow_with_epoch(&candidate, &arena, &cfg, &epoch);
            state.total_hashes.fetch_add(1, Ordering::Relaxed);

            if hash_below_target(&hash, &share_target) {
                info!(
                    "Thread {thread_id}: share found nonce={nonce} hash={}",
                    hash
                );
                let block_authorization = if hash_below_target(&hash, &block_target) {
                    match BlockAuthorization::sign(
                        &candidate,
                        &cfg,
                        job.reward_view_public,
                        job.reward_spend_public,
                        &miner_secret,
                    )
                    .and_then(|authorization| {
                        hyphen_codec::serialize_with_limit(&authorization, 256)
                            .map_err(|error| error.to_string())
                    }) {
                        Ok(encoded) => encoded,
                        Err(error) => {
                            error!("refusing unsigned block solution: {error}");
                            continue;
                        }
                    }
                } else {
                    Vec::new()
                };
                let _ = tx.send((nonce, extra_nonce, hash, job.job_id, block_authorization));
            }
        }
    }

    info!("Mining thread {thread_id} stopped");
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

fn hash_below_target(hash: &Hash256, target: &[u8; 32]) -> bool {
    for (h, t) in hash.as_bytes().iter().zip(target.iter()) {
        match h.cmp(t) {
            std::cmp::Ordering::Less => return true,
            std::cmp::Ordering::Greater => return false,
            std::cmp::Ordering::Equal => continue,
        }
    }
    true
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
        let hash = blake3::hash(&payload);
        payload.extend_from_slice(&hash.as_bytes()[..4]);
        format!("hy1{}", bs58::encode(payload).into_string())
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
}
