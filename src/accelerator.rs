//! Runtime loader for native HYN3 work plugins.
//! HYN3 mining requires a backend implementing the complete nonce search.

use std::path::{Path, PathBuf};
use std::ptr;

const ACCELERATOR_ABI_VERSION: u32 = 2;
const KERNEL_NEURAL_WORK: u32 = 4;
const CAP_NEURAL_WORK: u64 = 1 << 3;
const KERNEL_NEURAL_SEARCH: u32 = 5;
const KERNEL_NEURAL_PREPARE: u32 = 6;
const KERNEL_NEURAL_COMMIT: u32 = 7;
const CAP_NEURAL_SEARCH: u64 = 1 << 4;
pub const NEURAL_WINDOW: u32 = 32768;

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
struct NeuralWorkRequestV1 {
    start: u64,
    stride: u64,
    candidates: u32,
    reserved: u32,
    seed_words: [u32; 8],
    weights: [i64; 2128],
}

#[repr(C)]
struct NeuralSearchRequestV1 {
    work: NeuralWorkRequestV1,
    threshold: u64,
}

#[repr(C)]
struct NeuralTranscript {
    height: u64,
    length: u32,
    reserved: u32,
    bytes: [u8; 128],
}

impl NeuralTranscript {
    fn new(height: u64, parts: &[&[u8]]) -> Self {
        let mut transcript = Self {
            height,
            length: 0,
            reserved: 0,
            bytes: [0; 128],
        };
        for part in parts {
            let start = transcript.length as usize;
            transcript.bytes[start..start + part.len()].copy_from_slice(part);
            transcript.length += part.len() as u32;
        }
        transcript
    }
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
    pub fn load(directory: &Path, neural: bool) -> Result<Self, String> {
        if !neural {
            return Err("full GPU nonce search requires the HYN3 neural-devnet profile".into());
        }
        let capability = CAP_NEURAL_WORK | CAP_NEURAL_SEARCH;
        let extension = std::env::consts::DLL_EXTENSION;
        let mut problems = Vec::new();
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
                device.capability_mask & capability == capability
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

    fn neural_request(
        descriptor: &hyphen_neural::WorkDescriptor,
        weights: &[i64],
        start: u64,
        stride: u64,
        candidates: u32,
    ) -> Result<NeuralWorkRequestV1, String> {
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
        Ok(NeuralWorkRequestV1 {
            start,
            stride,
            candidates,
            reserved: 0,
            seed_words,
            weights: weights
                .try_into()
                .map_err(|_| "invalid neural weight count")?,
        })
    }

    pub fn prepare_job(
        &self,
        parent: &hyphen_crypto::Hash256,
        height: u64,
        miner: &[u8; 32],
    ) -> Result<([u8; 32], Vec<i64>), String> {
        let request = NeuralTranscript::new(
            height,
            &[
                hyphen_core::NEURAL_WORK_SEED_DOMAIN,
                parent.as_bytes(),
                &height.to_le_bytes(),
                miner,
            ],
        );
        self.execute(KERNEL_NEURAL_PREPARE, 1, &request, |result| {
            if result.output_len != 32 + hyphen_neural::PROFILE_SPEC.weight_count() * 8
                || result.operation_count != 1
            {
                return Err("invalid CUDA job descriptor".into());
            }
            let bytes = unsafe { std::slice::from_raw_parts(result.output, result.output_len) };
            let (seed, weights) = bytes.split_at(32);
            Ok((
                seed.try_into().unwrap(),
                weights
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .copied()
                    .map(i64::from_ne_bytes)
                    .collect(),
            ))
        })
    }

    pub fn work_commitment(
        &self,
        parent: &hyphen_crypto::Hash256,
        height: u64,
        threshold: u64,
        miner: &[u8; 32],
        nonce: u64,
    ) -> Result<hyphen_crypto::Hash256, String> {
        let request = NeuralTranscript::new(
            height,
            &[
                hyphen_core::NEURAL_WORK_COMMITMENT_DOMAIN,
                parent.as_bytes(),
                &height.to_le_bytes(),
                &threshold.to_le_bytes(),
                miner,
                &nonce.to_le_bytes(),
            ],
        );
        self.execute(KERNEL_NEURAL_COMMIT, 1, &request, |result| {
            if result.output_len != 32 || result.operation_count != 1 {
                return Err("invalid CUDA work commitment".into());
            }
            let digest = unsafe { std::ptr::read(result.output.cast::<[u8; 32]>()) };
            Ok(hyphen_crypto::Hash256::from_bytes(digest))
        })
    }

    pub fn neural_winners(
        &self,
        descriptor: &hyphen_neural::WorkDescriptor,
        weights: &[i64],
        start: u64,
        stride: u64,
        candidates: u32,
    ) -> Result<Vec<u32>, String> {
        let request = NeuralSearchRequestV1 {
            work: Self::neural_request(descriptor, weights, start, stride, candidates)?,
            threshold: descriptor.threshold,
        };
        self.execute(KERNEL_NEURAL_SEARCH, 1, &request, |result| {
            if result.operation_count != u64::from(candidates)
                || result.output_len < 4
                || result.output_len > 4 * (candidates as usize + 1)
                || !result.output_len.is_multiple_of(4)
            {
                return Err("invalid GPU winner list length".into());
            }
            let bytes = unsafe { std::slice::from_raw_parts(result.output, result.output_len) };
            let (header, indices) = bytes.split_at(4);
            if u32::from_ne_bytes(header.try_into().unwrap()) as usize != indices.len() / 4 {
                return Err("invalid GPU winner count".into());
            }
            let mut winners: Vec<_> = indices
                .as_chunks::<4>()
                .0
                .iter()
                .copied()
                .map(u32::from_ne_bytes)
                .collect();
            winners.sort_unstable();
            if winners.last().is_some_and(|&i| i >= candidates)
                || winners.windows(2).any(|w| w[0] == w[1])
            {
                return Err("invalid GPU winner index".into());
            }
            Ok(winners)
        })
    }

    pub fn neural_scores(
        &self,
        descriptor: &hyphen_neural::WorkDescriptor,
        weights: &[i64],
        start: u64,
        stride: u64,
        candidates: u32,
    ) -> Result<Vec<u64>, String> {
        let request = Self::neural_request(descriptor, weights, start, stride, candidates)?;
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
    assert!(size_of::<NeuralWorkRequestV1>() == 17080);
    assert!(size_of::<NeuralSearchRequestV1>() == 17088);
    assert!(std::mem::size_of::<BackendApiV1>() == 72);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires an HYN3 CUDA plugin and device"]
    fn native_neural_windows_match_rust() {
        let dir =
            std::env::var_os("HYPHEN_TEST_ACCELERATOR_DIR").expect("set the test plugin directory");
        let device = Accelerator::load(Path::new(&dir), true).unwrap();
        for (epoch, start, stride, count) in [
            (0, 0, 1, 1),
            (7, u64::MAX - 17, 11, 8193),
            (300, 100, 4, NEURAL_WINDOW),
        ] {
            let descriptor = hyphen_neural::WorkDescriptor {
                epoch,
                spec: hyphen_neural::PROFILE_SPEC,
                threshold: 1 << 60,
                chain_seed: std::array::from_fn(|i| (i as u8).wrapping_add(epoch as u8)),
            };
            let mut weights = vec![0; descriptor.spec.weight_count()];
            descriptor.derive_weights(&mut weights).unwrap();
            let scores = device
                .neural_scores(&descriptor, &weights, start, stride, count)
                .unwrap();
            assert_eq!(scores.len(), count as usize);
            let winners = device
                .neural_winners(&descriptor, &weights, start, stride, count)
                .unwrap();
            let (mut input, mut hidden, mut output) = ([0; 16], [0; 64], [0; 16]);
            for (i, score) in scores.into_iter().enumerate() {
                let counter = start.wrapping_add((i as u64).wrapping_mul(stride));
                descriptor.candidate(counter, &mut input).unwrap();
                assert_eq!(
                    winners.binary_search(&(i as u32)).is_ok(),
                    hyphen_neural::block_work::qualifies(&descriptor, counter, score).unwrap(),
                    "epoch {epoch}, counter {counter}"
                );
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

    #[test]
    #[ignore = "requires an HYN3 CUDA plugin and device"]
    fn native_neural_threshold_is_exclusive() {
        // Published HYN3 height-one vector, using the complete 64-bit memory score.
        let dir =
            std::env::var_os("HYPHEN_TEST_ACCELERATOR_DIR").expect("set the test plugin directory");
        let device = Accelerator::load(Path::new(&dir), true).unwrap();
        let cfg = hyphen_core::ChainConfig::neural_devnet();
        let genesis = hyphen_core::build_genesis_block(&cfg).hash();
        let seed = hyphen_core::neural_work_seed(&genesis, 1, &[19; 32]);
        let mut descriptor = hyphen_neural::WorkDescriptor {
            epoch: 1,
            spec: hyphen_neural::PROFILE_SPEC,
            chain_seed: *seed.as_bytes(),
            threshold: 664_485_265_619_752_970,
        };
        let mut weights = vec![0; descriptor.spec.weight_count()];
        descriptor.derive_weights(&mut weights).unwrap();
        let (device_seed, device_weights) = device.prepare_job(&genesis, 1, &[19; 32]).unwrap();
        assert_eq!(device_seed, descriptor.chain_seed);
        assert_eq!(device_weights, weights);
        assert_eq!(
            device
                .work_commitment(&genesis, 1, 1 << 60, &[19; 32], 125)
                .unwrap(),
            hyphen_core::neural_work_commitment(&genesis, 1, 1 << 60, &[19; 32], &[125])
        );
        assert!(
            device
                .neural_winners(&descriptor, &weights, 125, 1, 1)
                .unwrap()
                .is_empty()
        );
        descriptor.threshold += 1;
        assert_eq!(
            device
                .neural_winners(&descriptor, &weights, 125, 1, 1)
                .unwrap(),
            [0]
        );
    }
}
