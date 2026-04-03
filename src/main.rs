mod pow;
mod primitives;
mod protocol;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use clap::{Parser, Subcommand};
use parking_lot::RwLock;
use prost::Message;
use tokio::net::TcpStream;
use tracing::{error, info, warn};

use pow::{difficulty_to_target, evaluate_pow, evaluate_pow_with_epoch, EpochArena, EpochKernelParams};
use primitives::{BlockHeader, ChainConfig, Hash256, SecretKey};
use protocol::*;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct ArenaParamsData {
    pub arena_size: u64,
    pub page_size: u64,
}

#[derive(Parser, Debug)]
#[command(name = "hyphen-miner", about = "Hyphen standalone CPU miner")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    #[arg(long, default_value = "127.0.0.1:3340")]
    pool: String,

    #[arg(long, default_value = "0")]
    threads: usize,

    #[arg(long, default_value = "testnet")]
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
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Generate a new Ed25519 key pair and save to file
    Keygen {
        /// Output file path for the 32-byte secret key
        #[arg(long, default_value = "miner.key")]
        output: String,
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

    let cfg = match cli.network.as_str() {
        "mainnet" => ChainConfig::mainnet(),
        _ => ChainConfig::testnet(),
    };

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
        "Hyphen Miner starting – pool={}, threads={}, network={}",
        cli.pool, threads, cfg.network_name
    );

    let state = Arc::new(MinerState::new());

    let wallet_pubkey = if cli.wallet_address.is_empty() {
        sk.public_key().as_bytes().to_vec()
    } else if cli.wallet_address.starts_with("hy1") {
        let encoded = &cli.wallet_address[3..];
        let payload = bs58::decode(encoded)
            .into_vec()
            .expect("--wallet_address: invalid hy1 address base58");
        // payload = version(1) + view_pub(32) + spend_pub(32) + checksum(4) = 69
        if payload.len() != 69 {
            eprintln!(
                "Error: invalid hy1 address length: expected 69 decoded bytes, got {}",
                payload.len()
            );
            std::process::exit(1);
        }
        let checksum = &payload[65..69];
        let hash = blake3::hash(&payload[..65]);
        if checksum != &hash.as_bytes()[..4] {
            eprintln!("Error: hy1 address checksum mismatch");
            std::process::exit(1);
        }
        // Return view_public (32) + spend_public (32) = 64 bytes
        payload[1..65].to_vec()
    } else {
        hex::decode(&cli.wallet_address)
            .expect("--wallet_address must be a hy1... address or 64 hex chars")
    };
    if wallet_pubkey.len() != 32 && wallet_pubkey.len() != 64 {
        eprintln!("Error: wallet address must decode to 32 or 64 bytes");
        std::process::exit(1);
    }
    info!("Payout wallet: {}", hex::encode(&wallet_pubkey));

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
        sock.set_nodelay(true)?;
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

    let pool_pubkey = ack_env.sender_pubkey.clone();

    let conn_gen = state.connection_generation.load(Ordering::Acquire);

    let cfg_clone = cfg.clone();
    let state_clone = Arc::clone(state);
    let submit_tx = start_mining_threads(threads, cfg_clone.clone(), state_clone, cli_batch_size, conn_gen);

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
                if !pool_pubkey.is_empty() {
                    env.verify()?;
                }

                match env.msg_type {
                    MSG_JOB => {
                        let template = JobTemplate::decode(&env.payload[..])?;
                        handle_new_job(&template, cfg, state)?;
                    }
                    MSG_SUBMIT_RESULT => {
                        let result = SubmitResult::decode(&env.payload[..])?;
                        if result.accepted {
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
                if let Some((nonce, extra_nonce, pow_hash, job_id)) = share {
                    let submission = ShareSubmission {
                        job_id: job_id.to_vec(),
                        nonce,
                        extra_nonce: extra_nonce.to_vec(),
                        pow_hash: pow_hash.as_bytes().to_vec(),
                    };
                    let env = PoolEnvelope::sign(
                        MSG_SUBMIT,
                        submission.encode_to_vec(),
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
    }.await;

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
    if len > 64 * 1024 * 1024 {
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
) -> Result<(), Box<dyn std::error::Error>> {
    let header: BlockHeader = bincode::deserialize(&template.header_data)?;

    let mut job_id = [0u8; 32];
    if template.job_id.len() == 32 {
        job_id.copy_from_slice(&template.job_id);
    }

    let mut epoch_seed = Hash256::ZERO;
    if template.epoch_seed.len() == 32 {
        epoch_seed = Hash256::from_bytes(template.epoch_seed.clone().try_into().unwrap());
    }

    let (arena_size, page_size) = if template.arena_params.is_empty() {
        (cfg.arena_size, cfg.page_size)
    } else {
        let params: ArenaParamsData = bincode::deserialize(&template.arena_params)?;
        (params.arena_size as usize, params.page_size as usize)
    };

    let job = Arc::new(MiningJob {
        job_id,
        header,
        share_difficulty: template.share_difficulty,
        epoch_seed,
        arena_size,
        page_size,
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

type ShareResult = (u64, [u8; 32], Hash256, [u8; 32]);

fn start_mining_threads(
    threads: usize,
    cfg: ChainConfig,
    state: Arc<MinerState>,
    batch_size: u64,
    conn_gen: u64,
) -> tokio::sync::mpsc::UnboundedReceiver<ShareResult> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

    for thread_id in 0..threads {
        let cfg = cfg.clone();
        let state = Arc::clone(&state);
        let tx = tx.clone();

        std::thread::spawn(move || {
            mining_thread(thread_id, threads, cfg, state, tx, batch_size, conn_gen);
        });
    }

    rx
}

fn mining_thread(
    thread_id: usize,
    thread_count: usize,
    cfg: ChainConfig,
    state: Arc<MinerState>,
    tx: tokio::sync::mpsc::UnboundedSender<ShareResult>,
    batch_size: u64,
    conn_gen: u64,
) {
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
                let _ = tx.send((nonce, extra_nonce, hash, job.job_id));
            }
        }
    }

    info!("Mining thread {thread_id} stopped");
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
