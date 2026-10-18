//! Runtime loader for the optional native PoUW sieve plugins.
//!
//! The consensus-critical verification stays in Rust. A plugin only receives a
//! candidate window and returns the values that survive the same small-prime
//! sieve that [`hyphen_pow::vms::passes_small_prime_sieve`] implements. The
//! miner then runs deterministic Miller-Rabin and single-use checks on every
//! returned value, so a broken or malicious plugin can lose work but cannot
//! make the miner accept an invalid block.

use std::path::{Path, PathBuf};
use std::ptr;

const ACCELERATOR_ABI_VERSION: u32 = 2;
const KERNEL_CUNNINGHAM_SIEVE: u32 = 3;
const CAP_CUNNINGHAM_SIEVE: u64 = 1 << 2;
const MAX_CANDIDATES_PER_CALL: u32 = 1 << 24;
const HEADER_BYTES: usize = 16;

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

impl Default for DeviceInfoV1 {
    fn default() -> Self {
        Self {
            struct_size: 0,
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

#[repr(C)]
struct CunninghamSieveRequestV1 {
    offset: u64,
    stride: u64,
    candidates: u32,
    chain_length: u32,
    max_survivors: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CunninghamSieveHeaderV1 {
    survivor_count: u64,
    truncated: u64,
}

type EnumerateDevices = unsafe extern "C" fn(*mut DeviceInfoV1, u32, *mut u32) -> i32;
type Execute = unsafe extern "C" fn(*const ExecuteRequestV1, *mut ExecuteResultV1) -> i32;
type FreeResult = unsafe extern "C" fn(*mut u8, usize);
type GetLastError = unsafe extern "C" fn(*mut u8, usize) -> usize;
type GetApi = unsafe extern "C" fn(u32, *mut *const BackendApiV1) -> i32;

#[repr(C)]
#[derive(Clone, Copy)]
struct BackendApiV1 {
    abi_version: u32,
    struct_size: u32,
    backend: [u8; 32],
    enumerate_devices: Option<EnumerateDevices>,
    execute: Option<Execute>,
    free_result: Option<FreeResult>,
    get_last_error: Option<GetLastError>,
}

pub struct ScanOutcome {
    pub survivors: Vec<u64>,
    pub scanned: u32,
    pub truncated: bool,
    pub device_time_ns: u64,
}

pub struct Accelerator {
    // Declared first: the function pointers below are only valid while the
    // library is loaded.
    _library: libloading::Library,
    api: BackendApiV1,
    device: DeviceInfoV1,
    backend: String,
    device_name: String,
}

impl Accelerator {
    /// Load the first plugin that exposes a device with the Cunningham sieve
    /// capability. `directory` overrides the default search path.
    pub fn load(directory: Option<&Path>) -> Result<Self, String> {
        let mut directories = Vec::new();
        if let Some(directory) = directory {
            directories.push(directory.to_path_buf());
        }
        if let Some(directory) = std::env::var_os("HYPHEN_ACCELERATOR_DIR") {
            directories.push(PathBuf::from(directory));
        }
        if let Ok(executable) = std::env::current_exe() {
            if let Some(parent) = executable.parent() {
                directories.push(parent.join("accelerators"));
                directories.push(parent.join("accelerators").join("Release"));
                directories.push(parent.to_path_buf());
            }
        }
        if let Ok(current) = std::env::current_dir() {
            directories.push(current.join("accelerators"));
            directories.push(current.join("accelerators").join("Release"));
        }
        directories.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("accelerators"));
        directories.push(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("accelerators")
                .join("Release"),
        );

        let extension = std::env::consts::DLL_EXTENSION;
        let mut problems = Vec::new();
        for directory in directories {
            for backend in ["cuda", "hip"] {
                let path = directory.join(format!("hyphen_backend_{backend}.{extension}"));
                if !path.is_file() {
                    continue;
                }
                match Self::load_file(&path) {
                    Ok(accelerator) => return Ok(accelerator),
                    Err(error) => problems.push(format!("{}: {error}", path.display())),
                }
            }
        }

        if problems.is_empty() {
            Err("no native accelerator plugin found".into())
        } else {
            Err(problems.join("; "))
        }
    }

    fn load_file(path: &Path) -> Result<Self, String> {
        let library = unsafe { libloading::Library::new(path) }
            .map_err(|error| format!("cannot load {}: {error}", path.display()))?;
        let get_api: libloading::Symbol<'_, GetApi> = unsafe {
            library
                .get(b"hyphen_backend_get_api\0")
                .map_err(|error| format!("missing hyphen_backend_get_api: {error}"))?
        };

        let mut raw_api: *const BackendApiV1 = ptr::null();
        let status = unsafe { get_api(ACCELERATOR_ABI_VERSION, &mut raw_api) };
        if status != 0 || raw_api.is_null() {
            return Err(format!(
                "plugin rejected ABI {ACCELERATOR_ABI_VERSION} with status {status}"
            ));
        }
        let api = unsafe { *raw_api };
        if api.abi_version != ACCELERATOR_ABI_VERSION
            || (api.struct_size as usize) < std::mem::size_of::<BackendApiV1>()
        {
            return Err("plugin reported an incompatible ABI".into());
        }
        let enumerate = api
            .enumerate_devices
            .ok_or_else(|| "plugin has no enumerate_devices entry point".to_string())?;
        if api.execute.is_none() || api.free_result.is_none() || api.get_last_error.is_none() {
            return Err("plugin is missing an execution entry point".into());
        }

        let mut count = 0u32;
        let status = unsafe { enumerate(ptr::null_mut(), 0, &mut count) };
        if status != 0 {
            return Err(format!("device enumeration failed with status {status}"));
        }
        if count == 0 {
            return Err("plugin reports no devices".into());
        }

        let mut devices = vec![DeviceInfoV1::default(); count as usize];
        let status = unsafe { enumerate(devices.as_mut_ptr(), count, &mut count) };
        if status != 0 {
            return Err(format!("device enumeration failed with status {status}"));
        }
        devices.truncate(count as usize);

        let device = devices
            .into_iter()
            .find(|device| {
                device.capability_mask & CAP_CUNNINGHAM_SIEVE != 0
                    && device.hardware_accelerated != 0
                    && (device.struct_size as usize) >= std::mem::size_of::<DeviceInfoV1>()
            })
            .ok_or_else(|| "no device advertises the Cunningham sieve capability".to_string())?;

        let backend = if fixed_string(&device.backend).is_empty() {
            fixed_string(&api.backend)
        } else {
            fixed_string(&device.backend)
        };
        let device_name = fixed_string(&device.name);
        Ok(Self {
            _library: library,
            api,
            device,
            backend,
            device_name,
        })
    }

    pub fn backend(&self) -> &str {
        &self.backend
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn max_candidates(&self) -> u32 {
        MAX_CANDIDATES_PER_CALL
    }

    pub fn scan(
        &self,
        offset: u64,
        stride: u64,
        candidates: u32,
        chain_length: u32,
        max_survivors: u32,
    ) -> Result<ScanOutcome, String> {
        if stride == 0 {
            return Err("accelerator stride must be non-zero".into());
        }
        let candidates = candidates.clamp(1, MAX_CANDIDATES_PER_CALL);
        let max_survivors = max_survivors.clamp(1, candidates);
        let span = (candidates as u64 - 1)
            .checked_mul(stride)
            .ok_or_else(|| "accelerator candidate span overflows u64".to_string())?;
        let window_end = offset
            .checked_add(span)
            .ok_or_else(|| "accelerator candidate window overflows u64".to_string())?;
        let request = CunninghamSieveRequestV1 {
            offset,
            stride,
            candidates,
            chain_length,
            max_survivors,
            reserved: 0,
        };
        let execute = self
            .api
            .execute
            .ok_or_else(|| "accelerator has no execute entry point".to_string())?;
        let mut result = ExecuteResultV1 {
            struct_size: std::mem::size_of::<ExecuteResultV1>() as u32,
            output: ptr::null_mut(),
            output_len: 0,
            operation_count: 0,
            device_time_ns: 0,
        };
        let request_image = ExecuteRequestV1 {
            struct_size: std::mem::size_of::<ExecuteRequestV1>() as u32,
            kernel_id: KERNEL_CUNNINGHAM_SIEVE,
            device_ordinal: self.device.device_ordinal,
            iterations: chain_length,
            alpha_q12: 0,
            reserved: 0,
            input: (&request as *const CunninghamSieveRequestV1).cast::<u8>(),
            input_len: std::mem::size_of::<CunninghamSieveRequestV1>(),
        };

        let status = unsafe { execute(&request_image, &mut result) };
        if status != 0 {
            return Err(self.last_error());
        }
        if result.output.is_null() || result.output_len < HEADER_BYTES {
            self.free(result.output, result.output_len);
            return Err("accelerator returned a malformed result".into());
        }

        let parsed = unsafe { self.parse_result(&result, max_survivors, candidates) };
        self.free(result.output, result.output_len);
        let mut outcome = parsed?;
        for candidate in &outcome.survivors {
            if *candidate < offset
                || *candidate > window_end
                || !(*candidate - offset).is_multiple_of(stride)
            {
                return Err(format!(
                    "accelerator returned {candidate}, outside the requested window"
                ));
            }
        }
        outcome.survivors.sort_unstable();
        outcome.survivors.dedup();
        Ok(outcome)
    }

    unsafe fn parse_result(
        &self,
        result: &ExecuteResultV1,
        max_survivors: u32,
        scanned: u32,
    ) -> Result<ScanOutcome, String> {
        let header = ptr::read_unaligned(result.output.cast::<CunninghamSieveHeaderV1>());
        let count = header.survivor_count.min(max_survivors as u64) as usize;
        let required = HEADER_BYTES
            .checked_add(count.saturating_mul(std::mem::size_of::<u64>()))
            .ok_or_else(|| "accelerator result length overflow".to_string())?;
        if result.output_len < required {
            return Err("accelerator result is shorter than its header claims".into());
        }
        let values = result.output.add(HEADER_BYTES).cast::<u64>();
        let mut survivors = Vec::with_capacity(count);
        for index in 0..count {
            survivors.push(ptr::read_unaligned(values.add(index)));
        }
        Ok(ScanOutcome {
            survivors,
            scanned,
            truncated: header.truncated != 0,
            device_time_ns: result.device_time_ns,
        })
    }

    fn last_error(&self) -> String {
        let Some(get_last_error) = self.api.get_last_error else {
            return "accelerator call failed without an error entry point".into();
        };
        let mut buffer = vec![0u8; 1024];
        let length = unsafe { get_last_error(buffer.as_mut_ptr(), buffer.len()) };
        let length = length.min(buffer.len().saturating_sub(1));
        if length == 0 {
            return "accelerator call failed".into();
        }
        String::from_utf8_lossy(&buffer[..length]).into_owned()
    }

    fn free(&self, output: *mut u8, output_len: usize) {
        if let Some(free_result) = self.api.free_result {
            unsafe { free_result(output, output_len) };
        }
    }
}

fn fixed_string(bytes: &[u8]) -> String {
    let end = bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// Compile-time assertion that the C and Rust images agree.
const _: () = {
    assert!(std::mem::size_of::<DeviceInfoV1>() == 376);
    assert!(std::mem::size_of::<ExecuteRequestV1>() == 40);
    assert!(std::mem::size_of::<ExecuteResultV1>() == 40);
    assert!(std::mem::size_of::<CunninghamSieveHeaderV1>() == 16);
    assert!(std::mem::size_of::<CunninghamSieveRequestV1>() == 32);
    assert!(std::mem::size_of::<BackendApiV1>() == 72);
};
