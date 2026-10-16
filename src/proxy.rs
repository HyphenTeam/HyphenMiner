//! Operator-provided egress for the pool link.
//!
//! A miner that cannot open a TCP connection to its pool — a facility network
//! that only permits traffic to a proxy endpoint, a host whose policy routes
//! everything through one — can still mine:
//!
//! ```text
//! hyphen-miner --via-proxy socks5://user:pass@proxy.internal:1080 --pool pool.example:3340 …
//! hyphen-miner --via-proxy http://proxy.internal:3128          --pool pool.example:3340 …
//! ```
//!
//! The miner connects to the proxy and asks it for the pool; after that the
//! share protocol, keepalive and framing are the same bytes a direct link sends.
//! Two standard handshakes carry the request:
//!
//! * **SOCKS5** (RFC 1928) with optional username/password sub-negotiation
//!   (RFC 1929) — what an operator's SOCKS endpoint speaks, including a Corduit
//!   `socks5` inbound.
//! * **HTTP CONNECT** (RFC 9110 §9.3.6) with optional `Proxy-Authorization` —
//!   what a corporate proxy speaks, including a Corduit `http` inbound.
//!
//! Nothing here picks a proxy for the operator and nothing here is a proxy
//! implementation: the endpoint is theirs to configure and to be entitled to, so
//! an obfuscating proxy engine stays an external service rather than a library
//! this binary must link.
//!
//! ## Why the handshakes live in this file
//!
//! A proxy's *client* half is interop glue, not proxying: a greeting, one
//! exchange, one request line. It is written here against the RFCs and tested
//! against independent servers — never against the implementation that will sit
//! on the far side, which would prove nothing. Every read is bounded, every
//! failure names the response code, and "the tunnel is up but the pool is
//! unreachable" is a diagnosis an operator can make from the log alone.
//!
//! The HTTP path encodes `Proxy-Authorization` with Courierust's Base64, so the
//! one line of encoding it needs comes from a tested implementation.

use std::io::{Read, Write};
use std::net::{IpAddr, SocketAddr, TcpStream, ToSocketAddrs};
use std::time::Duration;

/// Deadline for reaching the proxy and completing the handshake. A proxied open
/// is two hops and a round trip, so this is generous next to a direct connect —
/// and still finite.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(20);

const SOCKS5_VERSION: u8 = 0x05;
const SOCKS5_AUTH_VERSION: u8 = 0x01;
const SOCKS5_METHOD_NO_AUTH: u8 = 0x00;
const SOCKS5_METHOD_USER_PASS: u8 = 0x02;
const SOCKS5_METHOD_NONE_ACCEPTABLE: u8 = 0xFF;
const SOCKS5_CMD_CONNECT: u8 = 0x01;
const SOCKS5_ATYP_IPV4: u8 = 0x01;
const SOCKS5_ATYP_DOMAIN: u8 = 0x03;
const SOCKS5_ATYP_IPV6: u8 = 0x04;

/// Longest HTTP status line accepted from a CONNECT proxy.
const MAX_STATUS_LINE: usize = 1024;

/// Longest header block accepted from a CONNECT proxy.
const MAX_HEADER_BYTES: usize = 16 * 1024;

/// Failure to parse a proxy spec or to open the tunnelled pool link.
#[derive(Debug, thiserror::Error)]
pub enum ProxyError {
    /// The `--via-proxy` value could not be understood.
    #[error("invalid --via-proxy value '{spec}': {reason}")]
    Spec {
        /// The offending value.
        spec: String,
        /// What is wrong with it.
        reason: String,
    },
    /// The proxy host does not resolve.
    #[error("proxy '{proxy}' does not resolve: {reason}")]
    Resolve {
        /// The proxy endpoint as written.
        proxy: String,
        /// Resolver error.
        reason: String,
    },
    /// The proxy could not be reached, or the exchange broke.
    #[error("talking to proxy '{proxy}' failed: {reason}")]
    Transport {
        /// The proxy endpoint.
        proxy: String,
        /// Transport error.
        reason: String,
    },
    /// The proxy answered, and the answer was not success.
    #[error("proxy '{proxy}' refused the tunnel to {pool}: {reason}")]
    Refused {
        /// The proxy endpoint.
        proxy: String,
        /// The pool address that was requested.
        pool: String,
        /// Protocol reply, decoded into words.
        reason: String,
    },
    /// A reply was malformed, truncated or oversized.
    #[error("proxy '{proxy}' sent an unusable reply: {reason}")]
    Protocol {
        /// The proxy endpoint.
        proxy: String,
        /// What was wrong with the bytes.
        reason: String,
    },
}

/// The protocol the operator's proxy speaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProxyKind {
    /// SOCKS5, RFC 1928, optional RFC 1929 authentication.
    Socks5,
    /// HTTP `CONNECT`, optional `Proxy-Authorization`.
    HttpConnect,
}

impl ProxyKind {
    /// Name for log lines.
    pub fn label(self) -> &'static str {
        match self {
            ProxyKind::Socks5 => "SOCKS5",
            ProxyKind::HttpConnect => "HTTP CONNECT",
        }
    }
}

/// The operator's outbound, parsed from `--via-proxy`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProxySpec {
    kind: ProxyKind,
    server: String,
    port: u16,
    username: Option<String>,
    password: Option<String>,
}

impl ProxySpec {
    /// Parses `socks5://[user:pass@]host:port` or `http://[user:pass@]host:port`.
    ///
    /// The scheme is mandatory: a bare `host:port` would be ambiguous with the
    /// pool address it sits next to on the command line, and guessing wrong
    /// would send the share protocol to a proxy as if it were the pool.
    pub fn parse(spec: &str) -> Result<Self, ProxyError> {
        let invalid = |reason: &str| ProxyError::Spec {
            spec: spec.to_string(),
            reason: reason.to_string(),
        };

        let (scheme, rest) = spec
            .split_once("://")
            .ok_or_else(|| invalid("expected a scheme: socks5:// or http://"))?;
        let kind = match scheme.to_ascii_lowercase().as_str() {
            "socks5" | "socks" => ProxyKind::Socks5,
            "http" | "https" => ProxyKind::HttpConnect,
            other => {
                return Err(invalid(&format!(
                    "unsupported proxy scheme '{other}'; use socks5:// or http://"
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
            kind,
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
}

/// An operator proxy the pool link can be opened through.
#[derive(Clone, Debug)]
pub struct PoolProxy {
    spec: ProxySpec,
    address: SocketAddr,
}

impl PoolProxy {
    /// Resolves the proxy endpoint once, at startup.
    ///
    /// Resolving here means a typo in `--via-proxy` fails the miner before it
    /// starts hashing, with the proxy named, instead of at the first reconnect.
    pub fn new(spec: ProxySpec) -> Result<Self, ProxyError> {
        let endpoint = spec.endpoint();
        let address = (spec.server.as_str(), spec.port)
            .to_socket_addrs()
            .map_err(|error| ProxyError::Resolve {
                proxy: endpoint.clone(),
                reason: error.to_string(),
            })?
            .next()
            .ok_or_else(|| ProxyError::Resolve {
                proxy: endpoint.clone(),
                reason: "no address records".to_string(),
            })?;
        Ok(Self { spec, address })
    }

    /// Opens a tunnelled connection to `pool`, or fails with the reason.
    ///
    /// `pool` is the operator's `--pool` value; a name stays a name, so the
    /// proxy resolves it and this host's resolver never learns the pool address.
    pub fn open(&self, pool: &str) -> Result<TcpStream, ProxyError> {
        let endpoint = self.spec.endpoint();
        let mut stream =
            TcpStream::connect_timeout(&self.address, HANDSHAKE_TIMEOUT).map_err(|error| {
                ProxyError::Transport {
                    proxy: endpoint.clone(),
                    reason: error.to_string(),
                }
            })?;
        stream
            .set_read_timeout(Some(HANDSHAKE_TIMEOUT))
            .and_then(|()| stream.set_write_timeout(Some(HANDSHAKE_TIMEOUT)))
            .map_err(|error| ProxyError::Transport {
                proxy: endpoint.clone(),
                reason: error.to_string(),
            })?;

        match self.spec.kind {
            ProxyKind::Socks5 => negotiate_socks5(&mut stream, &self.spec, pool)
                .map_err(|failure| failure.into_error(&endpoint, pool))?,
            ProxyKind::HttpConnect => negotiate_connect(&mut stream, &self.spec, pool)
                .map_err(|failure| failure.into_error(&endpoint, pool))?,
        }

        // The handshake deadline was the tunnel's, not the pool protocol's: a
        // pool decides how long a share exchange takes, and a proxy must not
        // impose its timeout on bytes that are no longer its business.
        stream
            .set_read_timeout(None)
            .and_then(|()| stream.set_write_timeout(None))
            .map_err(|error| ProxyError::Transport {
                proxy: endpoint,
                reason: error.to_string(),
            })?;
        Ok(stream)
    }

    /// One-line description of what the pool link will do, for the startup log.
    pub fn routing_note(&self, pool: &str) -> String {
        format!(
            "{} {} -> {pool}",
            self.spec.kind.label(),
            self.spec.endpoint()
        )
    }
}

/// A handshake failure, before it is attributed to a proxy and a pool.
#[derive(Debug)]
enum HandshakeFailure {
    /// The proxy refused the tunnel, with its own code.
    Refused(String),
    /// The reply could not be used.
    Protocol(String),
    /// The bytes could not be exchanged.
    Io(std::io::Error),
}

impl HandshakeFailure {
    fn into_error(self, proxy: &str, pool: &str) -> ProxyError {
        match self {
            HandshakeFailure::Refused(reason) => ProxyError::Refused {
                proxy: proxy.to_string(),
                pool: pool.to_string(),
                reason,
            },
            HandshakeFailure::Protocol(reason) => ProxyError::Protocol {
                proxy: proxy.to_string(),
                reason,
            },
            HandshakeFailure::Io(error) => ProxyError::Transport {
                proxy: proxy.to_string(),
                reason: error.to_string(),
            },
        }
    }
}

impl From<std::io::Error> for HandshakeFailure {
    fn from(error: std::io::Error) -> Self {
        HandshakeFailure::Io(error)
    }
}

type HandshakeResult = Result<(), HandshakeFailure>;

/// Performs the SOCKS5 greeting, optional credentials and CONNECT.
fn negotiate_socks5(stream: &mut TcpStream, spec: &ProxySpec, pool: &str) -> HandshakeResult {
    let greeting: &[u8] = if spec.username.is_some() {
        &[
            SOCKS5_VERSION,
            2,
            SOCKS5_METHOD_NO_AUTH,
            SOCKS5_METHOD_USER_PASS,
        ]
    } else {
        &[SOCKS5_VERSION, 1, SOCKS5_METHOD_NO_AUTH]
    };
    stream.write_all(greeting)?;

    let mut reply = [0u8; 2];
    stream.read_exact(&mut reply)?;
    if reply[0] != SOCKS5_VERSION {
        return Err(HandshakeFailure::Protocol(format!(
            "SOCKS version {} instead of {SOCKS5_VERSION}",
            reply[0]
        )));
    }
    match reply[1] {
        SOCKS5_METHOD_NO_AUTH => {}
        SOCKS5_METHOD_USER_PASS => {
            let (user, pass) = match (&spec.username, &spec.password) {
                (Some(user), Some(pass)) => (user.as_bytes(), pass.as_bytes()),
                _ => {
                    return Err(HandshakeFailure::Refused(
                        "it requires username/password authentication; put credentials in \
                         --via-proxy (socks5://user:pass@host:port)"
                            .to_string(),
                    ))
                }
            };
            if user.len() > 255 || pass.len() > 255 {
                return Err(HandshakeFailure::Protocol(
                    "credentials longer than 255 bytes are not expressible".to_string(),
                ));
            }
            let mut request = Vec::with_capacity(3 + user.len() + pass.len());
            request.extend_from_slice(&[SOCKS5_AUTH_VERSION, user.len() as u8]);
            request.extend_from_slice(user);
            request.push(pass.len() as u8);
            request.extend_from_slice(pass);
            stream.write_all(&request)?;

            let mut auth_reply = [0u8; 2];
            stream.read_exact(&mut auth_reply)?;
            if auth_reply[1] != 0x00 {
                return Err(HandshakeFailure::Refused(
                    "it rejected the supplied credentials".to_string(),
                ));
            }
        }
        SOCKS5_METHOD_NONE_ACCEPTABLE => {
            return Err(HandshakeFailure::Refused(
                "it accepts neither no-auth nor username/password".to_string(),
            ))
        }
        other => {
            return Err(HandshakeFailure::Protocol(format!(
                "it selected unsupported method 0x{other:02x}"
            )))
        }
    }

    let destination = Socks5Destination::parse(pool).map_err(HandshakeFailure::Protocol)?;
    let mut request = vec![SOCKS5_VERSION, SOCKS5_CMD_CONNECT, 0x00];
    request.extend_from_slice(&destination.encoded);
    stream.write_all(&request)?;

    let mut head = [0u8; 4];
    stream.read_exact(&mut head)?;
    if head[1] != 0x00 {
        return Err(HandshakeFailure::Refused(socks5_reply_text(head[1])));
    }
    // The bound address belongs to the tunnel, not to the pool protocol: consume
    // it before the stream carries pool bytes.
    let address_len = match head[3] {
        SOCKS5_ATYP_IPV4 => 4,
        SOCKS5_ATYP_IPV6 => 16,
        SOCKS5_ATYP_DOMAIN => {
            let mut length = [0u8; 1];
            stream.read_exact(&mut length)?;
            usize::from(length[0])
        }
        other => {
            return Err(HandshakeFailure::Protocol(format!(
                "it replied with unknown address type 0x{other:02x}"
            )))
        }
    };
    let mut discard = vec![0u8; address_len + 2];
    stream.read_exact(&mut discard)?;
    Ok(())
}

/// Performs an HTTP `CONNECT` tunnel request.
fn negotiate_connect(stream: &mut TcpStream, spec: &ProxySpec, pool: &str) -> HandshakeResult {
    let authority = normalise_authority(pool)
        .ok_or_else(|| HandshakeFailure::Protocol(format!("pool '{pool}' is not host:port")))?;

    let mut request = Vec::with_capacity(256);
    request.extend_from_slice(format!("CONNECT {authority} HTTP/1.1\r\n").as_bytes());
    request.extend_from_slice(format!("Host: {authority}\r\n").as_bytes());
    if let (Some(user), Some(pass)) = (&spec.username, &spec.password) {
        let token =
            courierust::courierust_crypto::base64::encode(format!("{user}:{pass}").as_bytes());
        request.extend_from_slice(format!("Proxy-Authorization: Basic {token}\r\n").as_bytes());
    }
    request.extend_from_slice(b"Proxy-Connection: keep-alive\r\n\r\n");
    stream.write_all(&request)?;

    let head = read_http_head(stream)?;
    let status = parse_status_code(&head)
        .ok_or_else(|| HandshakeFailure::Protocol("no HTTP status line".to_string()))?;
    if status != 200 {
        return Err(HandshakeFailure::Refused(format!(
            "it answered HTTP {status}"
        )));
    }
    Ok(())
}

/// Reads an HTTP response head (through the empty line) with hard size bounds.
fn read_http_head<R: Read>(stream: &mut R) -> Result<Vec<u8>, HandshakeFailure> {
    let mut head = Vec::with_capacity(256);
    let mut byte = [0u8; 1];
    loop {
        stream.read_exact(&mut byte)?;
        head.push(byte[0]);
        if head.ends_with(b"\r\n\r\n") {
            return Ok(head);
        }
        let status_line_done = head.windows(2).any(|pair| pair == b"\r\n");
        let limit = if status_line_done {
            MAX_HEADER_BYTES
        } else {
            MAX_STATUS_LINE
        };
        if head.len() > limit {
            return Err(HandshakeFailure::Protocol(format!(
                "reply head exceeds {limit} bytes"
            )));
        }
    }
}

/// Extracts the status code from `HTTP/1.1 200 Connection Established`.
fn parse_status_code(head: &[u8]) -> Option<u16> {
    let text = std::str::from_utf8(head).ok()?;
    let line = text.split("\r\n").next()?;
    let mut parts = line.split(' ');
    let version = parts.next()?;
    if !version.starts_with("HTTP/") {
        return None;
    }
    parts.next()?.parse().ok()
}

/// `host:port`, with IPv6 hosts bracketed as the CONNECT target form requires.
fn normalise_authority(pool: &str) -> Option<String> {
    let (host, port) = split_host_port(pool)?;
    Some(
        if host
            .parse::<IpAddr>()
            .is_ok_and(|address| address.is_ipv6())
        {
            format!("[{host}]:{port}")
        } else {
            format!("{host}:{port}")
        },
    )
}

/// SOCKS5 destination encoding for the pool address.
struct Socks5Destination {
    encoded: Vec<u8>,
}

impl Socks5Destination {
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

    /// An independent SOCKS5 server, written from RFC 1928 for these tests.
    ///
    /// Deliberately not the implementation that will sit on the far side in
    /// production: checking a client against its own server proves nothing.
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
                        assert!(ok, "the miner must present the operator's credentials");
                    }
                }

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

        fn join(mut self) -> (u8, Vec<u8>, u16) {
            self.handle.take().unwrap().join().unwrap();
            self.seen.recv().unwrap()
        }
    }

    /// An independent HTTP CONNECT proxy for these tests.
    struct StubConnect {
        addr: SocketAddr,
        request: std::sync::mpsc::Receiver<String>,
        handle: Option<std::thread::JoinHandle<()>>,
    }

    impl StubConnect {
        fn start(status: u16) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let (sender, request) = std::sync::mpsc::channel();
            let handle = std::thread::spawn(move || {
                let (mut client, _) = listener.accept().unwrap();
                client
                    .set_read_timeout(Some(Duration::from_secs(10)))
                    .unwrap();
                let head = read_http_head(&mut client).unwrap();
                sender
                    .send(String::from_utf8_lossy(&head).to_string())
                    .unwrap();
                if status == 200 {
                    client
                        .write_all(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                        .unwrap();
                    let mut payload = [0u8; 4];
                    if client.read_exact(&mut payload).is_ok() {
                        client.write_all(b"pong").unwrap();
                    }
                } else {
                    client
                        .write_all(format!("HTTP/1.1 {status} Denied\r\n\r\n").as_bytes())
                        .unwrap();
                }
            });
            Self {
                addr,
                request,
                handle: Some(handle),
            }
        }

        fn join(mut self) -> String {
            self.handle.take().unwrap().join().unwrap();
            self.request.recv().unwrap()
        }
    }

    #[test]
    fn parses_socks5_and_http_specs() {
        let socks = ProxySpec::parse("socks5://miner:s3cret@proxy.internal:1080").unwrap();
        assert_eq!(socks.kind, ProxyKind::Socks5);
        assert_eq!(socks.server, "proxy.internal");
        assert_eq!(socks.port, 1080);
        assert_eq!(socks.username.as_deref(), Some("miner"));
        assert_eq!(socks.password.as_deref(), Some("s3cret"));

        let http = ProxySpec::parse("http://proxy.internal:3128").unwrap();
        assert_eq!(http.kind, ProxyKind::HttpConnect);
        assert!(http.username.is_none());

        let v6 = ProxySpec::parse("socks5://[2001:db8::1]:1080").unwrap();
        assert_eq!(v6.server, "2001:db8::1");
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
            assert!(ProxySpec::parse(spec).is_err(), "{spec} must be rejected");
        }
    }

    /// The pool name travels as a name, so the proxy resolves it and this host
    /// never asks its own resolver for the pool.
    #[test]
    fn socks5_carries_the_pool_name_and_port() {
        let stub = StubSocks5::start(None);
        let proxy =
            PoolProxy::new(ProxySpec::parse(&format!("socks5://{}", stub.addr)).unwrap()).unwrap();

        let mut stream = proxy.open("pool.example:3340").unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        stream.write_all(b"ping").unwrap();
        let mut reply = [0u8; 4];
        stream.read_exact(&mut reply).unwrap();
        assert_eq!(&reply, b"pong");

        let (kind, host, port) = stub.join();
        assert_eq!(kind, SOCKS5_ATYP_DOMAIN);
        assert_eq!(host, b"pool.example");
        assert_eq!(port, 3340);
    }

    #[test]
    fn socks5_encodes_a_numeric_pool_address() {
        let stub = StubSocks5::start(None);
        let proxy =
            PoolProxy::new(ProxySpec::parse(&format!("socks5://{}", stub.addr)).unwrap()).unwrap();

        let mut stream = proxy.open("203.0.113.9:3340").unwrap();
        stream.write_all(b"ping").unwrap();

        let (kind, host, port) = stub.join();
        assert_eq!(kind, SOCKS5_ATYP_IPV4);
        assert_eq!(host, vec![203, 0, 113, 9]);
        assert_eq!(port, 3340);
    }

    #[test]
    fn socks5_credentials_are_presented_when_demanded() {
        let stub = StubSocks5::start(Some(("miner", "s3cret")));
        let proxy = PoolProxy::new(
            ProxySpec::parse(&format!("socks5://miner:s3cret@{}", stub.addr)).unwrap(),
        )
        .unwrap();

        let mut stream = proxy.open("pool.example:3340").unwrap();
        stream.write_all(b"ping").unwrap();
        stub.join();
    }

    #[test]
    fn missing_socks5_credentials_are_reported() {
        let stub = StubSocks5::start(Some(("miner", "s3cret")));
        let proxy =
            PoolProxy::new(ProxySpec::parse(&format!("socks5://{}", stub.addr)).unwrap()).unwrap();

        let error = proxy.open("pool.example:3340").unwrap_err();
        assert!(
            matches!(error, ProxyError::Refused { .. }),
            "unexpected error: {error:?}"
        );
        assert!(error.to_string().contains("neither"));
    }

    #[test]
    fn http_connect_sends_the_authority_and_credentials() {
        let stub = StubConnect::start(200);
        let proxy = PoolProxy::new(
            ProxySpec::parse(&format!("http://miner:s3cret@{}", stub.addr)).unwrap(),
        )
        .unwrap();

        let mut stream = proxy.open("pool.example:3340").unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        stream.write_all(b"ping").unwrap();
        let mut reply = [0u8; 4];
        stream.read_exact(&mut reply).unwrap();
        assert_eq!(&reply, b"pong");

        let request = stub.join();
        assert!(
            request.starts_with("CONNECT pool.example:3340 HTTP/1.1\r\n"),
            "unexpected request head: {request}"
        );
        assert!(request.contains("Host: pool.example:3340\r\n"));
        // `miner:s3cret` in Base64, produced by Courierust's encoder.
        assert!(request.contains("Proxy-Authorization: Basic bWluZXI6czNjcmV0\r\n"));
    }

    #[test]
    fn http_connect_refusals_are_reported_with_the_status() {
        let stub = StubConnect::start(403);
        let proxy =
            PoolProxy::new(ProxySpec::parse(&format!("http://{}", stub.addr)).unwrap()).unwrap();

        let error = proxy.open("pool.example:3340").unwrap_err();
        assert!(error.to_string().contains("403"), "unexpected: {error}");
        stub.join();
    }

    #[test]
    fn ipv6_pool_addresses_are_bracketed_for_connect() {
        assert_eq!(
            normalise_authority("[2001:db8::2]:3340").unwrap(),
            "[2001:db8::2]:3340"
        );
        assert_eq!(
            normalise_authority("pool.example:3340").unwrap(),
            "pool.example:3340"
        );
        assert!(normalise_authority("pool.example").is_none());
    }

    #[test]
    fn status_codes_are_parsed_and_explained() {
        assert_eq!(
            parse_status_code(b"HTTP/1.1 200 Connection Established\r\n\r\n"),
            Some(200)
        );
        assert_eq!(
            parse_status_code(b"HTTP/1.0 407 Proxy Authentication Required\r\n"),
            Some(407)
        );
        assert_eq!(parse_status_code(b"not http\r\n"), None);
        assert!(socks5_reply_text(0x05).contains("connection refused"));
    }

    /// An unreachable proxy fails before any mining starts, with the proxy
    /// named. Port 0 is not a connectable destination, so this is immediate and
    /// independent of whatever else is listening on the machine.
    #[test]
    fn unreachable_proxy_is_reported_before_any_mining() {
        let proxy = PoolProxy::new(ProxySpec::parse("socks5://127.0.0.1:0").unwrap()).unwrap();
        let error = proxy.open("pool.example:3340").unwrap_err();
        assert!(
            matches!(error, ProxyError::Transport { .. }),
            "unexpected error: {error:?}"
        );
    }
}
