//! Runtime loader for the native Cunningham sieve and neural work plugins.
//! The miner checks candidate witnesses and winning neural scores in Rust.

use std::path::{Path, PathBuf};
use std::ptr;

const ACCELERATOR_ABI_VERSION: u32 = 2;
const KERNEL_CUNNINGHAM_SIEVE: u32 = 3;
const CAP_CUNNINGHAM_SIEVE: u64 = 1 << 2;
const KERNEL_NEURAL_WORK: u32 = 4;
const CAP_NEURAL_WORK: u64 = 1 << 3;
pub const NEURAL_WINDOW: u32 = 32768;
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

#[repr(C)]
struct NeuralWorkRequestV1 {
    start: u64,
    stride: u64,
    candidates: u32,
    reserved: u32,
    seed_words: [u32; 8],
    weights: [i64; 2128],
}

type EnumerateDevices = unsafe extern "C" fn(*mut DeviceInfoV1, u32, *mut u32) -> i32;
type Execute = unsafe extern "C" fn(*const ExecuteRequestV1, *mut ExecuteResultV1) -> i32;
type FreeResult = unsafe extern "C" fn(*mut u8, usize);

struct PluginBuffer<'a> {
    result: &'a ExecuteResultV1,
    free: FreeResult,
}

impl Drop for PluginBuffer<'_> {
    fn drop(&mut self) {
        unsafe { (self.free)(self.result.output, self.result.output_len) };
    }
}
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
    path: PathBuf,
    api: BackendApiV1,
    device: DeviceInfoV1,
    backend: String,
    device_name: String,
}

impl Accelerator {
    pub fn load(directory: Option<&Path>, neural: bool) -> Result<Self, String> {
        let capability = if neural {
            CAP_NEURAL_WORK
        } else {
            CAP_CUNNINGHAM_SIEVE
        };
        let mut directories = Vec::new();
        let override_directory = directory
            .map(Path::to_path_buf)
            .or_else(|| std::env::var_os("HYPHEN_ACCELERATOR_DIR").map(PathBuf::from));
        if let Some(directory) = override_directory {
            directories.push(directory);
        } else {
            if let Ok(executable) = std::env::current_exe()
                && let Some(parent) = executable.parent()
            {
                directories.push(parent.join("accelerators"));
                directories.push(parent.join("accelerators").join("Release"));
                directories.push(parent.to_path_buf());
            }
        }

        let extension = std::env::consts::DLL_EXTENSION;
        let mut problems = Vec::new();
        for directory in directories {
            for backend in ["cuda", "hip"] {
                let path = directory.join(format!("hyphen_backend_{backend}.{extension}"));
                if !path.is_file() {
                    continue;
                }
                match Self::load_file(&path, capability) {
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

    fn load_file(path: &Path, capability: u64) -> Result<Self, String> {
        #[cfg(windows)]
        let library: libloading::Library = {
            use libloading::os::windows::{
                LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, Library,
            };
            let absolute = std::path::absolute(path).map_err(|e| e.to_string())?;
            unsafe {
                Library::load_with_flags(
                    absolute,
                    LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR,
                )
            }
            .map_err(|e| format!("cannot load {}: {e}", path.display()))?
            .into()
        };
        #[cfg(not(windows))]
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
        // Only the two header words are available until the table size is checked.
        let header = unsafe { raw_api.cast::<[u32; 2]>().read_unaligned() };
        if header[0] != ACCELERATOR_ABI_VERSION
            || (header[1] as usize) < std::mem::size_of::<BackendApiV1>()
        {
            return Err("plugin reported an incompatible ABI".into());
        }
        let api = unsafe { raw_api.read_unaligned() };
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

        if count > 1024 {
            return Err("plugin device count exceeds 1024".into());
        }
        let mut devices = vec![DeviceInfoV1::default(); count as usize];
        let status = unsafe { enumerate(devices.as_mut_ptr(), count, &mut count) };
        if status != 0 {
            return Err(format!("device enumeration failed with status {status}"));
        }
        if count as usize > devices.len() {
            return Err("plugin device list grew during enumeration".into());
        }
        devices.truncate(count as usize);

        let device = devices
            .into_iter()
            .find(|device| {
                device.capability_mask & capability != 0
                    && device.hardware_accelerated != 0
                    && (device.struct_size as usize) >= std::mem::size_of::<DeviceInfoV1>()
            })
            .ok_or_else(|| format!("no device advertises capability {capability:#x}"))?;

        let backend = if fixed_string(&device.backend).is_empty() {
            fixed_string(&api.backend)
        } else {
            fixed_string(&device.backend)
        };
        let device_name = fixed_string(&device.name);
        Ok(Self {
            _library: library,
            path: path.to_path_buf(),
            api,
            device,
            backend,
            device_name,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
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
        let mut outcome =
            self.execute(KERNEL_CUNNINGHAM_SIEVE, chain_length, &request, |result| {
                if result.output_len < HEADER_BYTES
                    || result.operation_count != u64::from(candidates)
                {
                    return Err("accelerator returned an incomplete sieve result".into());
                }
                unsafe { self.parse_result(result, max_survivors, candidates) }
            })?;
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

    pub fn neural_scores(
        &self,
        descriptor: &hyphen_neural::WorkDescriptor,
        weights: &[i64],
        start: u64,
        stride: u64,
        candidates: u32,
    ) -> Result<Vec<u64>, String> {
        if descriptor.spec != hyphen_neural::PROFILE_SPEC
            || stride == 0
            || !(1..=NEURAL_WINDOW).contains(&candidates)
        {
            return Err("unsupported neural search window".into());
        }
        let mut seed_words = [0; 8];
        for (word, bytes) in seed_words
            .iter_mut()
            .zip(descriptor.chain_seed.as_chunks::<4>().0)
        {
            *word = u32::from_le_bytes(*bytes);
        }
        let request = NeuralWorkRequestV1 {
            start,
            stride,
            candidates,
            reserved: 0,
            seed_words,
            weights: weights
                .try_into()
                .map_err(|_| "invalid neural weight count")?,
        };
        self.execute(KERNEL_NEURAL_WORK, 1, &request, |result| {
            if result.operation_count != u64::from(candidates)
                || result.output_len != candidates as usize * 8
            {
                return Err("accelerator returned an incomplete score window".into());
            }
            let bytes = unsafe { std::slice::from_raw_parts(result.output, result.output_len) };
            Ok(bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|word| u64::from_ne_bytes(*word))
                .collect())
        })
    }

    fn execute<I, T>(
        &self,
        kernel_id: u32,
        iterations: u32,
        input: &I,
        parse: impl FnOnce(&ExecuteResultV1) -> Result<T, String>,
    ) -> Result<T, String> {
        let request = ExecuteRequestV1 {
            struct_size: size_of::<ExecuteRequestV1>() as u32,
            kernel_id,
            device_ordinal: self.device.device_ordinal,
            iterations,
            alpha_q12: 0,
            reserved: 0,
            input: ptr::from_ref(input).cast(),
            input_len: size_of::<I>(),
        };
        let mut result = ExecuteResultV1 {
            struct_size: size_of::<ExecuteResultV1>() as u32,
            output: ptr::null_mut(),
            output_len: 0,
            operation_count: 0,
            device_time_ns: 0,
        };
        let execute = self.api.execute.ok_or("missing execute entry point")?;
        let free = self
            .api
            .free_result
            .ok_or("missing free_result entry point")?;
        if unsafe { execute(&request, &mut result) } != 0 {
            return Err(self.last_error());
        }
        if result.output.is_null() {
            return Err("accelerator returned no output".into());
        }
        let buffer = PluginBuffer {
            result: &result,
            free,
        };
        parse(buffer.result)
    }

    unsafe fn parse_result(
        &self,
        result: &ExecuteResultV1,
        max_survivors: u32,
        scanned: u32,
    ) -> Result<ScanOutcome, String> {
        // The caller checked the plugin buffer contains HEADER_BYTES and keeps it alive until return.
        let header =
            unsafe { ptr::read_unaligned(result.output.cast::<CunninghamSieveHeaderV1>()) };
        if header.survivor_count > u64::from(max_survivors) {
            return Err("accelerator returned too many survivors".into());
        }
        let count = header.survivor_count as usize;
        let required = HEADER_BYTES
            .checked_add(count.saturating_mul(std::mem::size_of::<u64>()))
            .ok_or_else(|| "accelerator result length overflow".to_string())?;
        if result.output_len < required {
            return Err("accelerator result is shorter than its header claims".into());
        }
        let mut survivors = Vec::with_capacity(count);
        for index in 0..count {
            // `required` covers the header and every entry through `count`.
            let value = unsafe {
                ptr::read_unaligned(result.output.add(HEADER_BYTES).cast::<u64>().add(index))
            };
            survivors.push(value);
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
    assert!(size_of::<NeuralWorkRequestV1>() == 17080);
    assert!(std::mem::size_of::<BackendApiV1>() == 72);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a device advertising kernel 4 and its plugin"]
    fn native_neural_windows_match_rust() {
        let device = Accelerator::load(None, true).unwrap();
        for (epoch, start, stride, count) in [
            (0, 0, 1, 1),
            (7, u64::MAX - 17, 11, 8193),
            (300, 100, 4, NEURAL_WINDOW),
        ] {
            let descriptor = hyphen_neural::WorkDescriptor {
                epoch,
                spec: hyphen_neural::PROFILE_SPEC,
                threshold: 1_000_000,
                chain_seed: std::array::from_fn(|i| (i as u8).wrapping_add(epoch as u8)),
            };
            let mut weights = vec![0; descriptor.spec.weight_count()];
            descriptor.derive_weights(&mut weights).unwrap();
            let scores = device
                .neural_scores(&descriptor, &weights, start, stride, count)
                .unwrap();
            assert_eq!(scores.len(), count as usize);
            let (mut input, mut hidden, mut output) = ([0; 16], [0; 64], [0; 16]);
            for (i, score) in scores.into_iter().enumerate() {
                let counter = start.wrapping_add((i as u64).wrapping_mul(stride));
                descriptor.candidate(counter, &mut input).unwrap();
                assert_eq!(
                    descriptor
                        .score(&weights, &input, &mut hidden, &mut output)
                        .unwrap(),
                    score,
                    "epoch {epoch}, counter {counter}"
                );
            }
        }
    }
}
