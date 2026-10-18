//! Prints a `hy1...` payout address for a fixed pair of test keys.
//!
//! Development aid for local devnet/neural-devnet runs, where an operator needs
//! a syntactically and cryptographically valid address before a wallet exists on
//! the machine. The keys are constant, so the output is reproducible and must
//! never be used to hold value.

fn main() {
    let cfg = hyphen_core::ChainConfig::neural_devnet();
    let view_public = hyphen_crypto::ViewKey([11; 32])
        .public_point()
        .compress()
        .to_bytes();
    let spend_public = hyphen_crypto::SpendKey([12; 32])
        .public_point()
        .compress()
        .to_bytes();
    let version = if cfg.network_magic == [0x48, 0x59, 0x50, 0x4e] {
        0x01
    } else {
        0x02
    };
    let mut payload = vec![version];
    payload.extend_from_slice(&view_public);
    payload.extend_from_slice(&spend_public);
    let checksum = hyphen_crypto::blake3_hash(&payload);
    payload.extend_from_slice(&checksum.as_bytes()[..4]);
    println!("hy1{}", hyphen_crypto::base58::encode(&payload));
}
