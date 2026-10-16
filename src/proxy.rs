//! Operator-provided egress for the pool link, tunnelled by Corduit.
//!
//! A miner that cannot open a TCP connection to its pool — a locked-down
//! facility network, a host that must route all traffic through a proxy
//! endpoint — can still mine by handing that problem to Corduit:
//!
//! ```text
//! hyphen-miner --via-proxy socks5://user:pass@proxy.internal:1080 --pool pool.example:3340 …
//! ```
//!
//! The miner starts Corduit's engine **in this process** with a loopback
//! `mixed` inbound and the operator's outbound, then dials that loopback
//! address and asks it for the pool. Corduit owns everything that makes a proxy
//! hard — the protocol implementations, the router, the connection tracking, the
//! thread pool — and the share protocol, keepalive and framing on this side are
//! untouched: from [`PoolTunnel::local_addr`] outward the miner still speaks the
//! pool's plain TCP dialect.
//!
//! ## Why the miner speaks SOCKS5 at all
//!
//! Corduit's inbounds *are* proxy endpoints: an application connects to them and
//! asks for a destination. That is the boundary where a proxy engine is useful
//! and also the boundary where it stays lawful — the operator configures a
//! proxy they are entitled to use, and this crate never picks one for them. The
//! client half of RFC 1928 lives here because it is the *interop* layer, not a
//! proxy implementation: a greeting, an optional RFC 1929 credential exchange,
//! one CONNECT, one reply. Every read is bounded and every failure names the
//! response code, because "the tunnel is up but the pool is unreachable" is a
//! diagnosis an operator has to be able to make from the log alone.
//!
//! The SOCKS5 request carries the pool's **name**, not an address, whenever the
//! operator wrote a name: the proxy resolves it, this host never does, and the
//! pool address stays out of the miner's DNS traffic.

use std::io::{Read, Write};
use std::net::{IpAddr, SocketAddr, TcpStream};
use std::time::Duration;

use corduit::engine::{
    Config, Corduit, GeneralConfig, InboundConfig, InboundType, Mode, OutboundConfig, OutboundType,
    RuleConfig, RuleType,
};

/// Handshake deadline. Dialling through a proxy is a two-hop TCP open plus DNS,
/// so this is generous compared to a direct connect, and still finite.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(20);

/// First loopback port the auto-selection tries.
const FIRST_LOCAL_PORT: u16 = 17_890;

/// How many consecutive ports the auto-selection may try before giving up.
const LOCAL_PORT_ATTEMPTS: u16 = 64;

/// SOCKS5 protocol version, and the sub-negotiation versions of RFC 1929.
const SOCKS5_VERSION: u8 = 0x05;
const SOCKS5_AUTH_VERSION: u8 = 0x01;

const SOCKS5_METHOD_NO_AUTH: u8 = 0x00;
const SOCKS5_METHOD_USER_PASS: u8 = 0x02;
const SOCKS5_METHOD_NONE_ACCEPTABLE: u8 = 0xFF;

const SOCKS5_CMD_CONNECT: u8 = 0x01;

const SOCKS5_ATYP_IPV4: u8 = 0x01;
const SOCKS5_ATYP_DOMAIN: u8 = 0x03;
const SOCKS5_ATYP_IPV6: u8 = 0x04;

/// Failure to build or use the pool tunnel.
#[derive(Debug, thiserror::Error)]
pub enum TunnelError {
    /// The `--via-proxy` value could not be understood.
    #[error("invalid --via-proxy value '{spec}': {reason}")]
    Spec {
        /// The offending value.
        spec: String,
        /// What is wrong with it.
        reason: String,
    },
    /// Corduit could not build or start the engine.
    #[error("cannot start the local proxy engine: {0}")]
    Engine(String),
    /// No loopback port was free for the engine's inbound.
    #[error("no free loopback port in {FIRST_LOCAL_PORT}..{end}")]
    NoLocalPort {
        /// One past the last port that was tried.
        end: u16,
    },
    /// The proxy rejected the handshake.
    #[error("proxy handshake with {proxy} failed: {reason}")]
    Handshake {
        /// Proxy endpoint the miner talked to.
        proxy: SocketAddr,
        /// Human-readable cause, including the SOCKS5 reply code when there is one.
        reason: String,
    },
}

/// The operator's outbound, parsed from `--via-proxy`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProxySpec {
    outbound: OutboundType,
    server: String,
    port: u16,
    username: Option<String>,
    password: Option<String>,
}

impl ProxySpec {
    /// Parses `socks5://[user:pass@]host:port`, `socks4://…` or `http://…`.
    ///
    /// The scheme is mandatory: a bare `host:port` would be ambiguous with the
    /// pool address it sits next to on the command line, and guessing wrong
    /// would send the share protocol to a proxy as if it were the pool.
    pub fn parse(spec: &str) -> Result<Self, TunnelError> {
        let invalid = |reason: &str| TunnelError::Spec {
            spec: spec.to_string(),
            reason: reason.to_string(),
        };

        let (scheme, rest) = spec
            .split_once("://")
            .ok_or_else(|| invalid("expected a scheme: socks5://, socks4:// or http://"))?;
        let outbound = match scheme.to_ascii_lowercase().as_str() {
            "socks5" | "socks" => OutboundType::Socks5,
            "socks4" | "socks4a" => OutboundType::Socks4,
            "http" | "https" => OutboundType::Http,
            other => {
                return Err(invalid(&format!(
                    "unsupported proxy scheme '{other}'; use socks5://, socks4:// or http://"
                )))
            }
        };

        let (credentials, authority) = match rest.rsplit_once('@') {
            Some((credentials, authority)) => (Some(credentials), authority),
            None => (None, rest),
        };
        let (username, password) = match credentials {
            None => (None, None),
            Some(credentials) => {
                let (user, pass) = credentials
                    .split_once(':')
                    .ok_or_else(|| invalid("credentials must be user:password"))?;
                if user.is_empty() || pass.is_empty() {
                    return Err(invalid("credentials must be non-empty user:password"));
                }
                (Some(user.to_string()), Some(pass.to_string()))
            }
        };

        let (server, port) = split_host_port(authority)
            .ok_or_else(|| invalid("expected host:port after the scheme"))?;
        if server.is_empty() {
            return Err(invalid("proxy host is empty"));
        }
        Ok(Self {
            outbound,
            server,
            port,
            username,
            password,
        })
    }

    /// Proxy endpoint, for log lines.
    pub fn endpoint(&self) -> String {
        format!("{}:{}", self.server, self.port)
    }

    /// Builds Corduit's typed configuration: one loopback `mixed` inbound, the
    /// operator's outbound, and a catch-all rule so *only* that outbound carries
    /// pool traffic.
    ///
    /// The rule is not decoration. Corduit's rule mode falls back to a
    /// continent-based direct shortcut when nothing matches, and an implicit
    /// direct egress is exactly the failure this tunnel exists to prevent: the
    /// operator routed the miner through a proxy, and a silent bypass would leak
    /// the pool link out of the machine they hardened. A `Match` rule placed
    /// first wins for every connection, so "through the proxy" is a property of
    /// the configuration rather than of the fallback order. Corduit also
    /// requires a `DIRECT` outbound to exist for its own bookkeeping; it can
    /// never be selected with this rule table.
    fn engine_config(&self, local_port: u16) -> Config {
        let mut outbound_options = std::collections::HashMap::new();
        if let (Some(user), Some(pass)) = (&self.username, &self.password) {
            outbound_options.insert(
                "username".to_string(),
                nextjson::Value::String(user.clone()),
            );
            outbound_options.insert(
                "password".to_string(),
                nextjson::Value::String(pass.clone()),
            );
        }

        Config {
            general: GeneralConfig {
                mixed_port: Some(local_port),
                bind_address: "127.0.0.1".to_string(),
                mode: Mode::Rule,
                ..GeneralConfig::default()
            },
            inbounds: vec![InboundConfig {
                inbound_type: InboundType::Mixed,
                tag: "hyphen-miner-egress".to_string(),
                listen: "127.0.0.1".to_string(),
                port: local_port,
                options: Default::default(),
            }],
            outbounds: vec![
                OutboundConfig {
                    outbound_type: self.outbound,
                    tag: OPERATOR_OUTBOUND.to_string(),
                    server: Some(self.server.clone()),
                    port: Some(self.port),
                    options: outbound_options,
                },
                OutboundConfig {
                    outbound_type: OutboundType::Direct,
                    tag: "DIRECT".to_string(),
                    server: None,
                    port: None,
                    options: Default::default(),
                },
            ],
            rules: vec![RuleConfig {
                rule_type: RuleType::Match,
                payload: String::new(),
                outbound: OPERATOR_OUTBOUND.to_string(),
                process_name: None,
                ..RuleConfig::default()
            }],
            ..Config::default()
        }
    }
}

/// Tag of the outbound the catch-all rule selects.
const OPERATOR_OUTBOUND: &str = "pool-egress";

/// A started Corduit engine plus the loopback address the pool link dials.
pub struct PoolTunnel {
    engine: Corduit,
    local: SocketAddr,
    spec: ProxySpec,
}

impl PoolTunnel {
    /// Starts the engine, auto-selecting a free loopback port.
    pub fn start(spec: ProxySpec) -> Result<Self, TunnelError> {
        Self::start_on(spec, None)
    }

    /// Starts the engine on `local` when the operator named a port.
    ///
    /// Without an explicit port each candidate is *attempted* rather than probed
    /// with a throwaway bind: a probe would have to release the port before
    /// Corduit binds it, and the gap between those two calls is a race the
    /// operator would see as an intermittent startup failure.
    pub fn start_on(spec: ProxySpec, local: Option<SocketAddr>) -> Result<Self, TunnelError> {
        match local {
            Some(address) => {
                let port = address.port();
                let engine = start_engine(&spec, port)?;
                Ok(Self {
                    engine,
                    local: address,
                    spec,
                })
            }
            None => {
                for port in FIRST_LOCAL_PORT..FIRST_LOCAL_PORT.saturating_add(LOCAL_PORT_ATTEMPTS) {
                    match start_engine(&spec, port) {
                        Ok(engine) => {
                            return Ok(Self {
                                engine,
                                local: SocketAddr::from(([127, 0, 0, 1], port)),
                                spec,
                            })
                        }
                        Err(error) if is_port_conflict(&error) => continue,
                        Err(error) => return Err(error),
                    }
                }
                Err(TunnelError::NoLocalPort {
                    end: FIRST_LOCAL_PORT.saturating_add(LOCAL_PORT_ATTEMPTS),
                })
            }
        }
    }

    /// The loopback address the pool link must dial.
    pub fn local_addr(&self) -> SocketAddr {
        self.local
    }

    /// The operator's proxy, for log lines.
    pub fn proxy_endpoint(&self) -> String {
        self.spec.endpoint()
    }

    /// Opens a tunnel to `pool` through the local inbound.
    ///
    /// `pool` is the operator's `--pool` value, so a name stays a name: the
    /// SOCKS5 request carries a domain and the proxy resolves it.
    pub fn connect_pool(&self, pool: &str) -> Result<TcpStream, TunnelError> {
        let mut stream =
            TcpStream::connect_timeout(&self.local, HANDSHAKE_TIMEOUT).map_err(|error| {
                TunnelError::Handshake {
                    proxy: self.local,
                    reason: error.to_string(),
                }
            })?;
        stream
            .set_read_timeout(Some(HANDSHAKE_TIMEOUT))
            .and_then(|()| stream.set_write_timeout(Some(HANDSHAKE_TIMEOUT)))
            .map_err(|error| TunnelError::Handshake {
                proxy: self.local,
                reason: error.to_string(),
            })?;

        negotiate_socks5(&mut stream, &self.spec, pool).map_err(|reason| {
            TunnelError::Handshake {
                proxy: self.local,
                reason,
            }
        })?;
        Ok(stream)
    }

    /// Stops the engine. Call at shutdown; dropping does the same.
    pub fn stop(self) -> Result<(), TunnelError> {
        self.engine
            .stop()
            .map_err(|error| TunnelError::Engine(error.to_string()))
    }
}

impl Drop for PoolTunnel {
    fn drop(&mut self) {
        let _ = self.engine.stop();
    }
}

/// Builds the engine for one loopback port and starts its listener.
fn start_engine(spec: &ProxySpec, local_port: u16) -> Result<Corduit, TunnelError> {
    let engine = Corduit::new(spec.engine_config(local_port))
        .map_err(|error| TunnelError::Engine(error.to_string()))?;
    engine
        .start()
        .map_err(|error| TunnelError::Engine(error.to_string()))?;
    Ok(engine)
}

/// Whether an engine start failure is "that port is taken" rather than a real
/// configuration error. Corduit reports a bind failure through its error chain,
/// so the check looks for the OS error rather than a variant that could change.
fn is_port_conflict(error: &TunnelError) -> bool {
    let text = error.to_string().to_ascii_lowercase();
    text.contains("address already in use")
        || text.contains("only one usage of each socket address")
        || text.contains("os error 10048")
        || text.contains("os error 98")
}

/// Performs the SOCKS5 greeting, optional credentials and CONNECT.
///
/// Returns a human-readable reason on failure; the caller attaches the proxy
/// endpoint. Every read is exact-length and bounded, so a peer that answers with
/// nothing, or with more than the protocol allows, fails the handshake instead
/// of hanging the miner.
fn negotiate_socks5(stream: &mut TcpStream, spec: &ProxySpec, pool: &str) -> Result<(), String> {
    let authenticate = spec.username.is_some();
    let greeting: &[u8] = if authenticate {
        &[
            SOCKS5_VERSION,
            2,
            SOCKS5_METHOD_NO_AUTH,
            SOCKS5_METHOD_USER_PASS,
        ]
    } else {
        &[SOCKS5_VERSION, 1, SOCKS5_METHOD_NO_AUTH]
    };
    stream
        .write_all(greeting)
        .map_err(|error| format!("greeting write failed: {error}"))?;

    let mut reply = [0u8; 2];
    stream
        .read_exact(&mut reply)
        .map_err(|error| format!("greeting read failed: {error}"))?;
    if reply[0] != SOCKS5_VERSION {
        return Err(format!(
            "proxy answered SOCKS version {} instead of {SOCKS5_VERSION}",
            reply[0]
        ));
    }
    match reply[1] {
        SOCKS5_METHOD_NO_AUTH => {}
        SOCKS5_METHOD_USER_PASS => {
            let (user, pass) = match (&spec.username, &spec.password) {
                (Some(user), Some(pass)) => (user.as_bytes(), pass.as_bytes()),
                _ => {
                    return Err(
                        "proxy requires username/password authentication; put credentials in \
                         --via-proxy (socks5://user:pass@host:port)"
                            .into(),
                    )
                }
            };
            if user.len() > 255 || pass.len() > 255 {
                return Err("credentials longer than 255 bytes are not expressible".into());
            }
            let mut request = Vec::with_capacity(3 + user.len() + pass.len());
            request.extend_from_slice(&[SOCKS5_AUTH_VERSION, user.len() as u8]);
            request.extend_from_slice(user);
            request.push(pass.len() as u8);
            request.extend_from_slice(pass);
            stream
                .write_all(&request)
                .map_err(|error| format!("credential write failed: {error}"))?;

            let mut auth_reply = [0u8; 2];
            stream
                .read_exact(&mut auth_reply)
                .map_err(|error| format!("credential read failed: {error}"))?;
            if auth_reply[1] != 0x00 {
                return Err("proxy rejected the supplied credentials".into());
            }
        }
        SOCKS5_METHOD_NONE_ACCEPTABLE => {
            return Err("proxy accepts neither no-auth nor username/password".into())
        }
        other => return Err(format!("proxy selected unsupported method 0x{other:02x}")),
    }

    let destination = Destination::parse(pool)?;
    let mut request = vec![SOCKS5_VERSION, SOCKS5_CMD_CONNECT, 0x00];
    request.extend_from_slice(&destination.encoded);
    stream
        .write_all(&request)
        .map_err(|error| format!("connect write failed: {error}"))?;

    let mut head = [0u8; 4];
    stream
        .read_exact(&mut head)
        .map_err(|error| format!("connect read failed: {error}"))?;
    if head[1] != 0x00 {
        return Err(format!(
            "proxy refused CONNECT: {}",
            socks5_reply_text(head[1])
        ));
    }
    // The bound address must be consumed before the stream carries pool bytes.
    let address_len = match head[3] {
        SOCKS5_ATYP_IPV4 => 4,
        SOCKS5_ATYP_IPV6 => 16,
        SOCKS5_ATYP_DOMAIN => {
            let mut length = [0u8; 1];
            stream
                .read_exact(&mut length)
                .map_err(|error| format!("connect read failed: {error}"))?;
            usize::from(length[0])
        }
        other => {
            return Err(format!(
                "proxy replied with unknown address type 0x{other:02x}"
            ))
        }
    };
    let mut discard = vec![0u8; address_len + 2];
    stream
        .read_exact(&mut discard)
        .map_err(|error| format!("connect read failed: {error}"))?;
    Ok(())
}

/// SOCKS5 destination encoding for the pool address.
struct Destination {
    encoded: Vec<u8>,
}

impl Destination {
    /// Encodes `host:port` as an IPv4, IPv6 or domain destination.
    fn parse(pool: &str) -> Result<Self, String> {
        let (host, port) =
            split_host_port(pool).ok_or_else(|| format!("pool '{pool}' is not host:port"))?;
        if host.is_empty() {
            return Err(format!("pool '{pool}' has an empty host"));
        }
        let mut encoded = Vec::with_capacity(1 + host.len() + 2);
        match host.parse::<IpAddr>() {
            Ok(IpAddr::V4(address)) => {
                encoded.push(SOCKS5_ATYP_IPV4);
                encoded.extend_from_slice(&address.octets());
            }
            Ok(IpAddr::V6(address)) => {
                encoded.push(SOCKS5_ATYP_IPV6);
                encoded.extend_from_slice(&address.octets());
            }
            Err(_) => {
                if host.len() > 255 {
                    return Err(format!("pool host '{host}' is longer than 255 bytes"));
                }
                encoded.push(SOCKS5_ATYP_DOMAIN);
                encoded.push(host.len() as u8);
                encoded.extend_from_slice(host.as_bytes());
            }
        }
        encoded.extend_from_slice(&port.to_be_bytes());
        Ok(Self { encoded })
    }
}

/// Splits `host:port`, accepting `[v6]:port` and rejecting anything else.
fn split_host_port(authority: &str) -> Option<(String, u16)> {
    let (host, port) = if let Some(rest) = authority.strip_prefix('[') {
        let (host, tail) = rest.split_once(']')?;
        (host, tail.strip_prefix(':')?)
    } else {
        authority.rsplit_once(':')?
    };
    let port = port.parse::<u16>().ok()?;
    Some((host.to_string(), port))
}

/// Human-readable SOCKS5 CONNECT reply code (RFC 1928 §6).
fn socks5_reply_text(code: u8) -> String {
    let reason = match code {
        0x01 => "general proxy failure",
        0x02 => "connection not allowed by ruleset",
        0x03 => "network unreachable",
        0x04 => "host unreachable",
        0x05 => "connection refused",
        0x06 => "TTL expired",
        0x07 => "command not supported",
        0x08 => "address type not supported",
        _ => "unknown reply code",
    };
    format!("0x{code:02x} ({reason})")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn parses_socks5_with_credentials() {
        let spec = ProxySpec::parse("socks5://miner:s3cret@proxy.internal:1080").unwrap();
        assert_eq!(spec.outbound, OutboundType::Socks5);
        assert_eq!(spec.server, "proxy.internal");
        assert_eq!(spec.port, 1080);
        assert_eq!(spec.username.as_deref(), Some("miner"));
        assert_eq!(spec.password.as_deref(), Some("s3cret"));

        let config = spec.engine_config(17_890);
        assert_eq!(config.general.mode, Mode::Rule);
        assert_eq!(config.inbounds.len(), 1);
        assert_eq!(config.inbounds[0].port, 17_890);
        // The operator's outbound is the one the catch-all rule selects, and an
        // outbound that could bypass it must be declared *because Corduit
        // requires it*, never because a rule reaches it.
        let rule = config.rules.first().expect("a catch-all rule");
        assert_eq!(rule.rule_type, RuleType::Match);
        assert_eq!(rule.outbound, OPERATOR_OUTBOUND);
        assert_eq!(config.outbounds[0].tag, OPERATOR_OUTBOUND);
        assert_eq!(
            config.outbounds[0].server.as_deref(),
            Some("proxy.internal")
        );
        assert!(!config.rules.iter().any(|rule| rule.outbound == "DIRECT"));
    }

    #[test]
    fn parses_http_and_ipv6_authorities() {
        let plain = ProxySpec::parse("http://proxy.internal:3128").unwrap();
        assert_eq!(plain.outbound, OutboundType::Http);
        assert!(plain.username.is_none());

        let v6 = ProxySpec::parse("socks5://[2001:db8::1]:1080").unwrap();
        assert_eq!(v6.server, "2001:db8::1");
        assert_eq!(v6.port, 1080);
    }

    #[test]
    fn rejects_specs_that_would_silently_misdial() {
        for spec in [
            "proxy.internal:1080",
            "socks5://proxy.internal",
            "socks5://user@proxy.internal:1080",
            "socks5://:1080",
            "socks5://proxy.internal:99999",
            "ftp://proxy.internal:21",
        ] {
            assert!(
                ProxySpec::parse(spec).is_err(),
                "{spec} must not be accepted"
            );
        }
    }

    #[test]
    fn destination_prefers_concrete_addresses() {
        let v4 = Destination::parse("203.0.113.9:3340").unwrap();
        assert_eq!(v4.encoded[0], SOCKS5_ATYP_IPV4);
        assert_eq!(&v4.encoded[1..5], &[203, 0, 113, 9]);
        assert_eq!(&v4.encoded[5..], &3340u16.to_be_bytes());

        let name = Destination::parse("pool.example:3340").unwrap();
        assert_eq!(name.encoded[0], SOCKS5_ATYP_DOMAIN);
        assert_eq!(name.encoded[1], "pool.example".len() as u8);
        assert_eq!(
            std::str::from_utf8(&name.encoded[2..2 + "pool.example".len()]).unwrap(),
            "pool.example"
        );

        let v6 = Destination::parse("[2001:db8::2]:3340").unwrap();
        assert_eq!(v6.encoded[0], SOCKS5_ATYP_IPV6);
    }

    #[test]
    fn reply_codes_are_explained() {
        assert!(socks5_reply_text(0x05).contains("connection refused"));
        assert!(socks5_reply_text(0x42).contains("unknown"));
    }

    /// The handshake, directly against an independent SOCKS5 server: the pool
    /// name must travel as a domain, so the proxy resolves it and this host
    /// never does.
    #[test]
    fn connect_carries_the_pool_name_and_port() {
        let proxy = StubSocks5::start(None);
        let spec = ProxySpec::parse(&format!("socks5://{}", proxy.addr)).unwrap();

        let mut stream = TcpStream::connect(proxy.addr).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        negotiate_socks5(&mut stream, &spec, "pool.example:3340").unwrap();
        stream.write_all(b"ping").unwrap();
        let mut reply = [0u8; 4];
        stream.read_exact(&mut reply).unwrap();
        assert_eq!(&reply, b"pong");

        let (kind, host, port) = proxy.join();
        assert_eq!(kind, SOCKS5_ATYP_DOMAIN);
        assert_eq!(host, b"pool.example");
        assert_eq!(port, 3340);
    }

    #[test]
    fn connect_encodes_a_numeric_pool_address() {
        let proxy = StubSocks5::start(None);
        let spec = ProxySpec::parse(&format!("socks5://{}", proxy.addr)).unwrap();

        let mut stream = TcpStream::connect(proxy.addr).unwrap();
        negotiate_socks5(&mut stream, &spec, "203.0.113.9:3340").unwrap();

        let (kind, host, port) = proxy.join();
        assert_eq!(kind, SOCKS5_ATYP_IPV4);
        assert_eq!(host, vec![203, 0, 113, 9]);
        assert_eq!(port, 3340);
    }

    /// Credentials offered in `--via-proxy` are presented when the proxy asks
    /// for them, and the handshake completes.
    #[test]
    fn credentials_are_presented_when_the_proxy_demands_them() {
        let proxy = StubSocks5::start(Some(("miner", "s3cret")));
        let spec = ProxySpec::parse(&format!("socks5://miner:s3cret@{}", proxy.addr)).unwrap();

        let mut stream = TcpStream::connect(proxy.addr).unwrap();
        negotiate_socks5(&mut stream, &spec, "pool.example:3340").unwrap();
        proxy.join();
    }

    /// A proxy that demands credentials the operator did not supply is a
    /// diagnosable failure, not a hang and not an unauthenticated attempt.
    #[test]
    fn missing_credentials_are_reported() {
        let proxy = StubSocks5::start(Some(("miner", "s3cret")));
        let spec = ProxySpec::parse(&format!("socks5://{}", proxy.addr)).unwrap();

        let mut stream = TcpStream::connect(proxy.addr).unwrap();
        let error = negotiate_socks5(&mut stream, &spec, "pool.example:3340")
            .expect_err("the tunnel must refuse an unauthenticated request");
        assert!(
            error.contains("proxy accepts neither"),
            "unexpected error: {error}"
        );
    }

    /// A minimal, independent SOCKS5 server for the tests below.
    ///
    /// Deliberately not Corduit: the client half of RFC 1928 is what the unit
    /// tests are about, and checking it against the implementation that produced
    /// it would prove nothing. It speaks no-auth when `credentials` is `None`,
    /// otherwise it requires method 0x02 and checks one RFC 1929 exchange. After
    /// a successful CONNECT it echoes four bytes, and it records the destination
    /// it was asked for.
    struct StubSocks5 {
        addr: SocketAddr,
        seen: std::sync::mpsc::Receiver<(u8, Vec<u8>, u16)>,
        handle: Option<std::thread::JoinHandle<()>>,
    }

    impl StubSocks5 {
        fn start(credentials: Option<(&'static str, &'static str)>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let (sender, seen) = std::sync::mpsc::channel();
            let handle = std::thread::spawn(move || {
                let (mut client, _) = listener.accept().unwrap();
                client
                    .set_read_timeout(Some(Duration::from_secs(10)))
                    .unwrap();
                // Greeting: version, method count, offered methods.
                let mut head = [0u8; 2];
                client.read_exact(&mut head).unwrap();
                assert_eq!(head[0], SOCKS5_VERSION);
                let mut methods = vec![0u8; usize::from(head[1])];
                client.read_exact(&mut methods).unwrap();

                match credentials {
                    None => {
                        assert!(methods.contains(&SOCKS5_METHOD_NO_AUTH));
                        client
                            .write_all(&[SOCKS5_VERSION, SOCKS5_METHOD_NO_AUTH])
                            .unwrap();
                    }
                    Some((user, pass)) => {
                        if !methods.contains(&SOCKS5_METHOD_USER_PASS) {
                            // The protocol's own answer to "I need a method you
                            // did not offer": none acceptable, then close.
                            client
                                .write_all(&[SOCKS5_VERSION, SOCKS5_METHOD_NONE_ACCEPTABLE])
                                .unwrap();
                            return;
                        }
                        client
                            .write_all(&[SOCKS5_VERSION, SOCKS5_METHOD_USER_PASS])
                            .unwrap();
                        let mut auth = [0u8; 2];
                        client.read_exact(&mut auth).unwrap();
                        assert_eq!(auth[0], SOCKS5_AUTH_VERSION);
                        let mut given_user = vec![0u8; usize::from(auth[1])];
                        client.read_exact(&mut given_user).unwrap();
                        let mut length = [0u8; 1];
                        client.read_exact(&mut length).unwrap();
                        let mut given_pass = vec![0u8; usize::from(length[0])];
                        client.read_exact(&mut given_pass).unwrap();
                        let ok = given_user == user.as_bytes() && given_pass == pass.as_bytes();
                        client
                            .write_all(&[SOCKS5_AUTH_VERSION, if ok { 0x00 } else { 0x01 }])
                            .unwrap();
                        assert!(ok, "the tunnel must present the operator's credentials");
                    }
                }

                // CONNECT: version, command, reserved, address type.
                let mut request = [0u8; 4];
                client.read_exact(&mut request).unwrap();
                assert_eq!(request[0], SOCKS5_VERSION);
                assert_eq!(request[1], SOCKS5_CMD_CONNECT);
                let (kind, host) = match request[3] {
                    SOCKS5_ATYP_DOMAIN => {
                        let mut length = [0u8; 1];
                        client.read_exact(&mut length).unwrap();
                        let mut name = vec![0u8; usize::from(length[0])];
                        client.read_exact(&mut name).unwrap();
                        (SOCKS5_ATYP_DOMAIN, name)
                    }
                    SOCKS5_ATYP_IPV4 => {
                        let mut address = [0u8; 4];
                        client.read_exact(&mut address).unwrap();
                        (SOCKS5_ATYP_IPV4, address.to_vec())
                    }
                    other => panic!("unexpected address type 0x{other:02x}"),
                };
                let mut port = [0u8; 2];
                client.read_exact(&mut port).unwrap();
                client
                    .write_all(&[
                        SOCKS5_VERSION,
                        0x00,
                        0x00,
                        SOCKS5_ATYP_IPV4,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ])
                    .unwrap();

                sender.send((kind, host, u16::from_be_bytes(port))).unwrap();
                let mut payload = [0u8; 4];
                if client.read_exact(&mut payload).is_ok() {
                    client.write_all(b"pong").unwrap();
                }
            });
            Self {
                addr,
                seen,
                handle: Some(handle),
            }
        }

        /// The destination the client asked for, and the payload it sent.
        fn join(mut self) -> (u8, Vec<u8>, u16) {
            self.handle.take().unwrap().join().unwrap();
            self.seen.recv().unwrap()
        }
    }

    /// With the proxy unusable the tunnel refuses to start rather than quietly
    /// dialling the pool directly. Port 0 is not a connectable destination, so this
    /// fails immediately and independently of whatever else is listening on the
    /// machine ? a "free port" would be taken by a parallel test instead.
    #[test]
    fn unusable_proxy_is_refused_instead_of_dialling_direct() {
        let spec = ProxySpec::parse("socks5://127.0.0.1:0").unwrap();
        let error = PoolTunnel::start(spec)
            .err()
            .expect("a tunnel to port 0 must not start")
            .to_string();
        assert!(error.contains("proxy engine"), "unexpected error: {error}");
    }
}
