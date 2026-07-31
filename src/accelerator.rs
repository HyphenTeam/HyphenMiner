use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fmt;
use std::mem::{size_of, MaybeUninit};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::time::Instant;

use libloading::{Library, Symbol};

const ABI_VERSION: u32 = 1;
const KERNEL_DIFFUSION_Q12_V1: u32 = 1;
const CAP_DIFFUSION_Q12_V1: u64 = 1;
const MAX_DEVICES_PER_BACKEND: u32 = 128;
const MAX_RESULT_BYTES: usize = 512 * 1024 * 1024;
const SELF_TEST_ALPHA_Q12: u32 = 512;
const SELF_TEST_ITERATIONS: u32 = 7;
const SELF_TEST_INPUT: [i32; 16] = [
    32768, 65536, 98304, 131072, 229376, 196608, 163840, 131072, 98304, 65536, 32768, 16384, 8192,
    4096, 2048, 1024,
];

const EXPECTED_BACKENDS: [(&str, &str); 4] = [
    ("nvidia-cuda", "hyphen_backend_cuda"),
    ("amd-hip", "hyphen_backend_hip"),
    ("intel-openvino", "hyphen_backend_openvino"),
    ("qualcomm-qnn", "hyphen_backend_qnn"),
];

#[repr(C)]
#[derive(Clone, Copy)]
struct DeviceInfoV1 {
    struct_size: u32,
    device_ordinal: u32,
    device_kind: u32,
    hardware_accelerated: u32,
    capability_mask: u64,
    backend: [u8; 32],
    vendor: [u8; 64],
    name: [u8; 128],
    stable_id: [u8; 64],
    runtime: [u8; 64],
}

impl DeviceInfoV1 {
    fn empty() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            device_ordinal: 0,
            device_kind: 0,
            hardware_accelerated: 0,
            capability_mask: 0,
            backend: [0; 32],
            vendor: [0; 64],
            name: [0; 128],
            stable_id: [0; 64],
            runtime: [0; 64],
        }
    }
}

#[repr(C)]
struct ExecuteRequestV1 {
    struct_size: u32,
    kernel_id: u32,
    device_ordinal: u32,
    iterations: u32,
    alpha_q12: u32,
    reserved: u32,
    input: *const u8,
    input_len: usize,
}

#[repr(C)]
struct ExecuteResultV1 {
    struct_size: u32,
    output: *mut u8,
    output_len: usize,
    operation_count: u64,
    device_time_ns: u64,
}

type EnumerateDevicesFn = unsafe extern "C" fn(*mut DeviceInfoV1, u32, *mut u32) -> i32;
type ExecuteFn = unsafe extern "C" fn(*const ExecuteRequestV1, *mut ExecuteResultV1) -> i32;
type FreeResultFn = unsafe extern "C" fn(*mut u8, usize);
type GetLastErrorFn = unsafe extern "C" fn(*mut u8, usize) -> usize;

#[repr(C)]
struct BackendApiV1 {
    abi_version: u32,
    struct_size: u32,
    backend: [u8; 32],
    enumerate_devices: Option<EnumerateDevicesFn>,
    execute: Option<ExecuteFn>,
    free_result: Option<FreeResultFn>,
    get_last_error: Option<GetLastErrorFn>,
}

type GetApiFn = unsafe extern "C" fn(u32, *mut *const BackendApiV1) -> i32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendStatus {
    Available,
    Unavailable(String),
    SelfTestFailed(String),
}

impl fmt::Display for BackendStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Available => f.write_str("available"),
            Self::Unavailable(reason) => write!(f, "unavailable: {reason}"),
            Self::SelfTestFailed(reason) => write!(f, "self-test failed: {reason}"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DeviceReport {
    pub backend: String,
    pub vendor: String,
    pub name: String,
    pub stable_id: String,
    pub runtime: String,
    pub status: BackendStatus,
    pub operation_count: Option<u64>,
    pub device_time_ns: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct VerifiedExecution {
    pub backend: String,
    pub device: String,
    pub stable_id: String,
    pub input_commitment: String,
    pub output_commitment: String,
    pub output: Vec<i32>,
    pub operation_count: u64,
    pub device_time_ns: u64,
}

pub struct AcceleratorRegistry {
    loaded: Vec<LoadedBackend>,
    reports: Vec<DeviceReport>,
}

struct LoadedBackend {
    _library: Library,
    api: NonNull<BackendApiV1>,
    devices: Vec<DeviceInfoV1>,
}

impl AcceleratorRegistry {
    pub fn discover(directory: &Path) -> Self {
        let mut loaded = Vec::new();
        let mut reports = Vec::new();
        let mut seen = BTreeMap::<String, bool>::new();

        let candidates = plugin_candidates(directory);
        for path in candidates {
            match LoadedBackend::load(&path) {
                Ok((backend, mut backend_reports)) => {
                    seen.insert(backend.name(), true);
                    reports.append(&mut backend_reports);
                    loaded.push(backend);
                }
                Err(error) => {
                    let backend = backend_name_from_path(&path);
                    seen.insert(backend.clone(), true);
                    reports.push(unavailable_report(backend, error));
                }
            }
        }

        for (backend, _) in EXPECTED_BACKENDS {
            if !seen.contains_key(backend) {
                reports.push(unavailable_report(
                    backend.to_string(),
                    format!("plugin is not installed in {}", directory.display()),
                ));
            }
        }
        reports.sort_by(|left, right| {
            left.backend
                .cmp(&right.backend)
                .then_with(|| left.stable_id.cmp(&right.stable_id))
        });

        Self { loaded, reports }
    }

    pub fn reports(&self) -> &[DeviceReport] {
        &self.reports
    }

    pub fn available_devices(&self) -> usize {
        self.reports
            .iter()
            .filter(|report| report.status == BackendStatus::Available)
            .count()
    }

    pub fn loaded_backend_count(&self) -> usize {
        self.loaded.len()
    }

    pub fn execute_verified(
        &self,
        backend_filter: Option<&str>,
        input: &[i32],
        alpha_q12: u32,
        iterations: u32,
    ) -> Result<VerifiedExecution, String> {
        let expected = diffusion_q12_reference(input, alpha_q12, iterations)?;
        let input_bytes = encode_i32(input);
        let expected_output = encode_i32(&expected);
        let mut failures = Vec::new();

        for backend in &self.loaded {
            if backend_filter.is_some_and(|filter| backend.name() != filter) {
                continue;
            }
            for device in &backend.devices {
                if device.hardware_accelerated == 0
                    || device.capability_mask & CAP_DIFFUSION_Q12_V1 == 0
                {
                    continue;
                }
                match backend.execute_and_check(
                    device.device_ordinal,
                    &input_bytes,
                    &expected_output,
                    input.len(),
                    alpha_q12,
                    iterations,
                ) {
                    Ok((output, operation_count, device_time_ns)) => {
                        return Ok(VerifiedExecution {
                            backend: fixed_string(&device.backend),
                            device: fixed_string(&device.name),
                            stable_id: fixed_string(&device.stable_id),
                            input_commitment: execution_commitment(
                                b"Hyphen/AetherCompute/input/v1",
                                alpha_q12,
                                iterations,
                                &input_bytes,
                            ),
                            output_commitment: execution_commitment(
                                b"Hyphen/AetherCompute/output/v1",
                                alpha_q12,
                                iterations,
                                &expected_output,
                            ),
                            output,
                            operation_count,
                            device_time_ns,
                        });
                    }
                    Err(error) => failures.push(format!(
                        "{} {}: {error}",
                        fixed_string(&device.backend),
                        fixed_string(&device.name)
                    )),
                }
            }
        }

        if failures.is_empty() {
            let requested = backend_filter.unwrap_or("any");
            Err(format!(
                "no verified accelerator supports diffusion-q12-v1 (requested backend: {requested})"
            ))
        } else {
            Err(format!(
                "all matching accelerator executions failed: {}",
                failures.join("; ")
            ))
        }
    }
}

impl LoadedBackend {
    fn load(path: &Path) -> Result<(Self, Vec<DeviceReport>), String> {
        let absolute = path
            .canonicalize()
            .map_err(|error| format!("cannot resolve plugin path: {error}"))?;
        // Loading is restricted to an explicitly selected, canonical plugin file. The ABI is
        // validated before any backend function is called.
        let library = unsafe { Library::new(&absolute) }
            .map_err(|error| format!("cannot load {}: {error}", absolute.display()))?;
        let get_api: Symbol<GetApiFn> = unsafe { library.get(b"hyphen_backend_get_api\0") }
            .map_err(|error| format!("missing hyphen_backend_get_api: {error}"))?;
        let mut api_ptr = std::ptr::null();
        let status = unsafe { get_api(ABI_VERSION, &mut api_ptr) };
        if status != 0 {
            return Err(format!(
                "backend rejected ABI v{ABI_VERSION} with code {status}"
            ));
        }
        let api = NonNull::new(api_ptr.cast_mut()).ok_or("backend returned a null API table")?;
        let api_ref = unsafe { api.as_ref() };
        if api_ref.abi_version != ABI_VERSION
            || api_ref.struct_size < size_of::<BackendApiV1>() as u32
        {
            return Err("backend API table has an incompatible version or size".into());
        }
        let enumerate = api_ref
            .enumerate_devices
            .ok_or("backend has no device enumeration function")?;
        api_ref.execute.ok_or("backend has no execute function")?;
        api_ref
            .free_result
            .ok_or("backend has no result release function")?;

        let mut count = 0u32;
        let code = unsafe { enumerate(std::ptr::null_mut(), 0, &mut count) };
        if code != 0 {
            return Err(format!(
                "device enumeration failed with code {code}: {}",
                last_error(api_ref)
            ));
        }
        if count > MAX_DEVICES_PER_BACKEND {
            return Err(format!(
                "backend reported an unreasonable device count: {count}"
            ));
        }

        let name = fixed_string(&api_ref.backend);
        let mut backend = Self {
            _library: library,
            api,
            devices: vec![DeviceInfoV1::empty(); count as usize],
        };
        if count == 0 {
            let detail = last_error(api_ref);
            let reason = if detail.is_empty() {
                "no compatible device found".to_string()
            } else {
                detail
            };
            return Ok((backend, vec![unavailable_report(name, reason)]));
        }

        let mut returned = count;
        let code = unsafe { enumerate(backend.devices.as_mut_ptr(), count, &mut returned) };
        if code != 0 || returned > count {
            return Err(format!(
                "device enumeration fill failed with code {code}: {}",
                last_error(api_ref)
            ));
        }
        backend.devices.truncate(returned as usize);

        let reports = backend
            .devices
            .iter()
            .map(|device| backend.self_test(device))
            .collect();
        Ok((backend, reports))
    }

    fn name(&self) -> String {
        fixed_string(&unsafe { self.api.as_ref() }.backend)
    }

    fn self_test(&self, device: &DeviceInfoV1) -> DeviceReport {
        let mut report = DeviceReport {
            backend: fixed_string(&device.backend),
            vendor: fixed_string(&device.vendor),
            name: fixed_string(&device.name),
            stable_id: fixed_string(&device.stable_id),
            runtime: fixed_string(&device.runtime),
            status: BackendStatus::Available,
            operation_count: None,
            device_time_ns: None,
        };
        if device.struct_size < size_of::<DeviceInfoV1>() as u32 {
            report.status = BackendStatus::Unavailable("device record is too small".into());
            return report;
        }
        if device.hardware_accelerated == 0 {
            report.status =
                BackendStatus::Unavailable("provider is not hardware accelerated".into());
            return report;
        }
        if device.capability_mask & CAP_DIFFUSION_Q12_V1 == 0 {
            report.status = BackendStatus::Unavailable(
                "deterministic diffusion-q12-v1 kernel is unsupported".into(),
            );
            return report;
        }

        match self.execute_self_test(device.device_ordinal) {
            Ok((operations, device_time_ns)) => {
                report.operation_count = Some(operations);
                report.device_time_ns = Some(device_time_ns);
            }
            Err(error) => report.status = BackendStatus::SelfTestFailed(error),
        }
        report
    }

    fn execute_self_test(&self, device_ordinal: u32) -> Result<(u64, u64), String> {
        let input = encode_i32(&SELF_TEST_INPUT);
        let expected =
            diffusion_q12_reference(&SELF_TEST_INPUT, SELF_TEST_ALPHA_Q12, SELF_TEST_ITERATIONS)?;
        let (_, operations, device_time_ns) = self.execute_and_check(
            device_ordinal,
            &input,
            &encode_i32(&expected),
            SELF_TEST_INPUT.len(),
            SELF_TEST_ALPHA_Q12,
            SELF_TEST_ITERATIONS,
        )?;
        Ok((operations, device_time_ns))
    }

    fn execute_and_check(
        &self,
        device_ordinal: u32,
        input: &[u8],
        expected_output: &[u8],
        cell_count: usize,
        alpha_q12: u32,
        iterations: u32,
    ) -> Result<(Vec<i32>, u64, u64), String> {
        let request = ExecuteRequestV1 {
            struct_size: size_of::<ExecuteRequestV1>() as u32,
            kernel_id: KERNEL_DIFFUSION_Q12_V1,
            device_ordinal,
            iterations,
            alpha_q12,
            reserved: 0,
            input: input.as_ptr(),
            input_len: input.len(),
        };
        let mut result = ExecuteResultV1 {
            struct_size: size_of::<ExecuteResultV1>() as u32,
            output: std::ptr::null_mut(),
            output_len: 0,
            operation_count: 0,
            device_time_ns: 0,
        };
        let api = unsafe { self.api.as_ref() };
        let execute = api.execute.expect("validated execute function");
        let started = Instant::now();
        let code = unsafe { execute(&request, &mut result) };
        if code != 0 {
            return Err(format!(
                "device execution returned {code}: {}",
                last_error(api)
            ));
        }
        let free = api.free_result.expect("validated result release function");
        let output = ResultBuffer::new(result.output, result.output_len, free)?;
        if output.len() > MAX_RESULT_BYTES {
            return Err("backend result exceeds the configured safety limit".into());
        }
        if output.as_slice() != expected_output {
            return Err("device output does not match the independent Rust CPU result".into());
        }
        let expected_operations = (cell_count as u64)
            .checked_mul(iterations as u64)
            .and_then(|value| value.checked_mul(6))
            .ok_or("operation count overflow")?;
        if result.operation_count != expected_operations {
            return Err(format!(
                "backend operation count {} does not match expected {expected_operations}",
                result.operation_count
            ));
        }
        let device_time = if result.device_time_ns == 0 {
            started.elapsed().as_nanos().min(u64::MAX as u128) as u64
        } else {
            result.device_time_ns
        };
        let decoded = decode_i32(output.as_slice())?;
        Ok((decoded, result.operation_count, device_time))
    }
}

struct ResultBuffer {
    pointer: NonNull<u8>,
    len: usize,
    free: FreeResultFn,
}

impl ResultBuffer {
    fn new(pointer: *mut u8, len: usize, free: FreeResultFn) -> Result<Self, String> {
        if len == 0 || len > MAX_RESULT_BYTES {
            if !pointer.is_null() {
                unsafe { free(pointer, len) };
            }
            return Err(format!("backend returned invalid result length {len}"));
        }
        let pointer = NonNull::new(pointer).ok_or("backend returned a null result buffer")?;
        Ok(Self { pointer, len, free })
    }

    fn len(&self) -> usize {
        self.len
    }

    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.len) }
    }
}

impl Drop for ResultBuffer {
    fn drop(&mut self) {
        unsafe { (self.free)(self.pointer.as_ptr(), self.len) };
    }
}

fn plugin_candidates(directory: &Path) -> Vec<PathBuf> {
    let extension = if cfg!(windows) { "dll" } else { "so" };
    let mut paths = std::fs::read_dir(directory)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path.extension() == Some(OsStr::new(extension))
                && path
                    .file_stem()
                    .and_then(OsStr::to_str)
                    .is_some_and(|name| name.starts_with("hyphen_backend_"))
        })
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

fn backend_name_from_path(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(OsStr::to_str)
        .unwrap_or("unknown");
    EXPECTED_BACKENDS
        .iter()
        .find_map(|(backend, library)| (*library == stem).then(|| (*backend).to_string()))
        .unwrap_or_else(|| stem.to_string())
}

fn unavailable_report(backend: String, reason: impl Into<String>) -> DeviceReport {
    DeviceReport {
        backend,
        vendor: String::new(),
        name: String::new(),
        stable_id: String::new(),
        runtime: String::new(),
        status: BackendStatus::Unavailable(reason.into()),
        operation_count: None,
        device_time_ns: None,
    }
}

fn fixed_string(bytes: &[u8]) -> String {
    let end = bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

fn last_error(api: &BackendApiV1) -> String {
    let Some(get_last_error) = api.get_last_error else {
        return "backend supplied no error detail".into();
    };
    let mut buffer = [MaybeUninit::<u8>::uninit(); 512];
    let written = unsafe { get_last_error(buffer.as_mut_ptr().cast(), buffer.len()) };
    let initialized = written.min(buffer.len());
    let bytes = unsafe { std::slice::from_raw_parts(buffer.as_ptr().cast::<u8>(), initialized) };
    fixed_string(bytes)
}

fn encode_i32(values: &[i32]) -> Vec<u8> {
    let mut encoded = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        encoded.extend_from_slice(&value.to_le_bytes());
    }
    encoded
}

fn decode_i32(bytes: &[u8]) -> Result<Vec<i32>, String> {
    if !bytes.len().is_multiple_of(size_of::<i32>()) {
        return Err("backend returned a result that is not an i32 array".into());
    }
    Ok(bytes
        .chunks_exact(size_of::<i32>())
        .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("exact chunk size")))
        .collect())
}

fn execution_commitment(domain: &[u8], alpha_q12: u32, iterations: u32, data: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&KERNEL_DIFFUSION_Q12_V1.to_le_bytes());
    hasher.update(&alpha_q12.to_le_bytes());
    hasher.update(&iterations.to_le_bytes());
    hasher.update(&(data.len() as u64).to_le_bytes());
    hasher.update(data);
    hasher.finalize().to_hex().to_string()
}

fn diffusion_q12_reference(
    input: &[i32],
    alpha_q12: u32,
    iterations: u32,
) -> Result<Vec<i32>, String> {
    if input.len() < 3 {
        return Err("diffusion input requires at least three cells".into());
    }
    if !(1..=1024).contains(&iterations) {
        return Err("diffusion iterations must be in 1..=1024".into());
    }
    if alpha_q12 > 2048 {
        return Err("alpha_q12 must not exceed 2048".into());
    }
    if input.iter().any(|value| !(0..=262_143).contains(value)) {
        return Err("diffusion input cells must be in 0..=262143".into());
    }

    let alpha = i64::from(alpha_q12);
    let center_weight = 4096i64 - 2 * alpha;
    let mut current = input.to_vec();
    let mut next = vec![0i32; input.len()];
    for _ in 0..iterations {
        for index in 0..current.len() {
            let left = i64::from(current[(index + current.len() - 1) % current.len()]);
            let center = i64::from(current[index]);
            let right = i64::from(current[(index + 1) % current.len()]);
            let numerator = center_weight * center + alpha * left + alpha * right;
            next[index] =
                i32::try_from(numerator / 4096).map_err(|_| "diffusion result exceeds i32")?;
        }
        std::mem::swap(&mut current, &mut next);
    }
    Ok(current)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diffusion_reference_is_deterministic_and_bounded() {
        let first =
            diffusion_q12_reference(&SELF_TEST_INPUT, SELF_TEST_ALPHA_Q12, SELF_TEST_ITERATIONS)
                .unwrap();
        let second =
            diffusion_q12_reference(&SELF_TEST_INPUT, SELF_TEST_ALPHA_Q12, SELF_TEST_ITERATIONS)
                .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.iter().sum::<i32>(), 1_276_910);
        assert!(first.iter().all(|value| (0..=262_143).contains(value)));
    }

    #[test]
    fn missing_plugins_are_reported_as_unavailable() {
        let unique = format!("hyphen-miner-accelerator-test-{}", std::process::id());
        let directory = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&directory).unwrap();
        let registry = AcceleratorRegistry::discover(&directory);
        std::fs::remove_dir(&directory).unwrap();
        assert_eq!(registry.available_devices(), 0);
        assert_eq!(registry.reports().len(), EXPECTED_BACKENDS.len());
        assert!(registry
            .reports()
            .iter()
            .all(|report| matches!(report.status, BackendStatus::Unavailable(_))));
    }

    #[test]
    #[ignore = "requires a locally built native accelerator and compatible hardware"]
    fn installed_accelerator_verifies_non_self_test_work() {
        let registry = AcceleratorRegistry::discover(Path::new("accelerators"));
        let input = vec![
            1024, 2048, 8192, 32768, 65536, 131072, 196608, 262143, 196608, 131072, 65536, 32768,
        ];
        let execution = registry
            .execute_verified(Some("nvidia-cuda"), &input, 384, 19)
            .unwrap();
        assert_eq!(execution.backend, "nvidia-cuda");
        assert_eq!(execution.output.len(), input.len());
        assert_eq!(execution.operation_count, input.len() as u64 * 19 * 6);
        assert_eq!(execution.input_commitment.len(), 64);
        assert_eq!(execution.output_commitment.len(), 64);
        assert_ne!(execution.input_commitment, execution.output_commitment);
    }
}
