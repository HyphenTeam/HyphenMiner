#include "hyphen_accelerator.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

thread_local std::string last_error;

constexpr uint32_t kThreadsPerBlock = 256;
constexpr uint32_t kMaxChainLength = 6;

void set_error(const char *operation, hipError_t error) {
  last_error = std::string(operation) + ": " + hipGetErrorString(error);
}

void set_error(const char *message) { last_error = message; }

template <size_t N> void copy_text(uint8_t (&destination)[N], const char *source) {
  if (source == nullptr) {
    destination[0] = 0;
    return;
  }
  const size_t length = std::min(std::strlen(source), N - 1);
  std::memcpy(destination, source, length);
  destination[length] = 0;
}

struct DeviceBuffer {
  void *pointer = nullptr;

  ~DeviceBuffer() {
    if (pointer != nullptr) {
      hipFree(pointer);
    }
  }

  bool allocate(size_t bytes) {
    if (bytes == 0) {
      return true;
    }
    return hipMalloc(&pointer, bytes) == hipSuccess;
  }
};

const std::vector<uint16_t> &small_primes() {
  static const std::vector<uint16_t> primes = [] {
    std::vector<bool> composite(HYPHEN_CUNNINGHAM_SIEVE_LIMIT, false);
    std::vector<uint16_t> values;
    values.reserve(600);
    for (uint32_t candidate = 2; candidate < HYPHEN_CUNNINGHAM_SIEVE_LIMIT;
         ++candidate) {
      if (composite[candidate]) {
        continue;
      }
      values.push_back(static_cast<uint16_t>(candidate));
      for (uint64_t multiple = static_cast<uint64_t>(candidate) * candidate;
           multiple < HYPHEN_CUNNINGHAM_SIEVE_LIMIT; multiple += candidate) {
        composite[static_cast<size_t>(multiple)] = true;
      }
    }
    return values;
  }();
  return primes;
}

uint32_t mod_inverse(uint32_t value, uint32_t modulus) {
  int64_t previous = 0;
  int64_t current = 1;
  int64_t remainder = modulus;
  int64_t next = value;
  while (next != 0) {
    const int64_t quotient = remainder / next;
    const int64_t temporary = previous - quotient * current;
    previous = current;
    current = temporary;
    const int64_t remainder_next = remainder - quotient * next;
    remainder = next;
    next = remainder_next;
  }
  if (remainder != 1) {
    return 0;
  }
  if (previous < 0) {
    previous += modulus;
  }
  return static_cast<uint32_t>(previous);
}

__global__ void mark_composites_kernel(uint8_t *rejected, const uint16_t *starts,
                                       const uint16_t *steps, uint32_t prime_count,
                                       uint32_t length, uint32_t candidates) {
  const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t total = prime_count * length;
  if (thread >= total) {
    return;
  }
  const uint32_t step = steps[thread];
  if (step == 0) {
    return;
  }
  uint32_t index = starts[thread];
  if (step == 1) {
    for (; index < candidates; ++index) {
      rejected[index] = 1;
    }
    return;
  }
  for (; index < candidates; index += step) {
    rejected[index] = 1;
  }
}

__global__ void compact_survivors_kernel(const uint8_t *rejected, uint32_t candidates,
                                         uint64_t offset, uint64_t stride,
                                         uint64_t *counter, uint64_t *truncated,
                                         uint64_t *survivors, uint32_t capacity) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= candidates || rejected[index] != 0) {
    return;
  }
  const uint64_t slot = atomicAdd(counter, UINT64_C(1));
  if (slot < capacity) {
    survivors[slot] = offset + static_cast<uint64_t>(index) * stride;
  } else {
    *truncated = 1;
  }
}

int32_t enumerate_devices(HyphenDeviceInfoV1 *devices, uint32_t capacity,
                          uint32_t *count) {
  if (count == nullptr) {
    set_error("device count pointer is null");
    return 1;
  }
  int device_count = 0;
  const hipError_t status = hipGetDeviceCount(&device_count);
  if (status == hipErrorNoDevice) {
    *count = 0;
    return 0;
  }
  if (status != hipSuccess) {
    set_error("hipGetDeviceCount", status);
    return 2;
  }

  int runtime_version = 0;
  hipRuntimeGetVersion(&runtime_version);
  char runtime[64];
  std::snprintf(runtime, sizeof(runtime), "HIP %d.%d", runtime_version / 10000000,
                (runtime_version / 100000) % 100);

  uint32_t written = 0;
  for (int ordinal = 0;
       ordinal < device_count && written < capacity && devices != nullptr;
       ++ordinal) {
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, ordinal) != hipSuccess) {
      continue;
    }
    HyphenDeviceInfoV1 &info = devices[written];
    std::memset(&info, 0, sizeof(info));
    info.struct_size = sizeof(HyphenDeviceInfoV1);
    info.device_ordinal = static_cast<uint32_t>(ordinal);
    info.device_kind = HYPHEN_DEVICE_KIND_GPU;
    info.hardware_accelerated = 1;
    info.capability_mask = HYPHEN_CAP_CUNNINGHAM_SIEVE_V1;
    copy_text(info.backend, "hip");
    copy_text(info.vendor, "AMD");
    copy_text(info.name, properties.name);
    char identifier[64];
    std::snprintf(identifier, sizeof(identifier), "%04x:%02x:%02x.0",
                  properties.pciDomainID, properties.pciBusID,
                  properties.pciDeviceID);
    copy_text(info.stable_id, identifier);
    copy_text(info.runtime, runtime);
    ++written;
  }

  if (devices == nullptr) {
    *count = static_cast<uint32_t>(device_count);
  } else {
    *count = written;
  }
  return 0;
}

int32_t execute_cunningham_sieve(const HyphenExecuteRequestV1 *request,
                                 HyphenExecuteResultV1 *result) {
  if (request == nullptr || result == nullptr) {
    set_error("execute request or result is null");
    return 1;
  }
  if (request->struct_size != sizeof(HyphenExecuteRequestV1) ||
      result->struct_size != sizeof(HyphenExecuteResultV1)) {
    set_error("execute struct size mismatch");
    return 2;
  }
  if (request->input == nullptr ||
      request->input_len != sizeof(HyphenCunninghamSieveRequestV1)) {
    set_error("Cunningham sieve request has the wrong input image");
    return 3;
  }

  HyphenCunninghamSieveRequestV1 sieve{};
  std::memcpy(&sieve, request->input, sizeof(sieve));
  if (sieve.chain_length < 2 || sieve.chain_length > kMaxChainLength) {
    set_error("Cunningham chain length is outside the supported range");
    return 4;
  }
  if (sieve.stride == 0 || sieve.candidates == 0 ||
      sieve.candidates > HYPHEN_CUNNINGHAM_MAX_CANDIDATES) {
    set_error("Cunningham sieve candidate window is invalid");
    return 5;
  }
  if (sieve.max_survivors == 0) {
    set_error("Cunningham sieve survivor capacity must be positive");
    return 6;
  }
  if (static_cast<uint64_t>(sieve.candidates - 1) >
      (UINT64_MAX - sieve.offset) / sieve.stride) {
    set_error("Cunningham sieve window overflows u64");
    return 7;
  }

  const hipError_t device_status = hipSetDevice(static_cast<int>(request->device_ordinal));
  if (device_status != hipSuccess) {
    set_error("hipSetDevice", device_status);
    return 8;
  }

  const std::vector<uint16_t> &primes = small_primes();
  const uint32_t prime_count = static_cast<uint32_t>(primes.size());
  const uint32_t entry_count = prime_count * sieve.chain_length;
  std::vector<uint16_t> starts(entry_count, 0);
  std::vector<uint16_t> steps(entry_count, 0);

  for (uint32_t prime_index = 0; prime_index < prime_count; ++prime_index) {
    const uint32_t prime = primes[prime_index];
    const uint32_t base = prime_index * sieve.chain_length;
    const uint32_t offset_mod = static_cast<uint32_t>(sieve.offset % prime);

    if (prime == 2) {
      const uint32_t stride_mod = static_cast<uint32_t>(sieve.stride % 2);
      if (stride_mod == 0) {
        if (offset_mod == 0) {
          starts[base] = 0;
          steps[base] = 1;
        }
      } else {
        starts[base] = offset_mod == 0 ? 0 : 1;
        steps[base] = 2;
      }
      continue;
    }

    const uint32_t stride_mod = static_cast<uint32_t>(sieve.stride % prime);
    const uint32_t inverse_two = (prime + 1) / 2;
    uint32_t inverse_power = 1;

    if (stride_mod == 0) {
      for (uint32_t link = 0; link < sieve.chain_length; ++link) {
        const uint32_t root = (inverse_power + prime - 1) % prime;
        if (offset_mod == root) {
          starts[base + link] = 0;
          steps[base + link] = 1;
        }
        inverse_power =
            static_cast<uint32_t>(static_cast<uint64_t>(inverse_power) *
                                  inverse_two % prime);
      }
      continue;
    }

    const uint32_t inverse_stride = mod_inverse(stride_mod, prime);
    if (inverse_stride == 0) {
      set_error("failed to invert the candidate stride modulo a sieve prime");
      return 9;
    }
    for (uint32_t link = 0; link < sieve.chain_length; ++link) {
      const uint32_t root = (inverse_power + prime - 1) % prime;
      const uint32_t difference = (root + prime - offset_mod) % prime;
      starts[base + link] = static_cast<uint16_t>(
          static_cast<uint64_t>(difference) * inverse_stride % prime);
      steps[base + link] = static_cast<uint16_t>(prime);
      inverse_power = static_cast<uint32_t>(
          static_cast<uint64_t>(inverse_power) * inverse_two % prime);
    }
  }

  const uint32_t capacity = std::min(sieve.max_survivors, sieve.candidates);
  DeviceBuffer rejected;
  DeviceBuffer device_starts;
  DeviceBuffer device_steps;
  DeviceBuffer counter;
  DeviceBuffer truncated;
  DeviceBuffer survivors;

  if (!rejected.allocate(sieve.candidates) ||
      !device_starts.allocate(starts.size() * sizeof(uint16_t)) ||
      !device_steps.allocate(steps.size() * sizeof(uint16_t)) ||
      !counter.allocate(sizeof(uint64_t)) ||
      !truncated.allocate(sizeof(uint64_t)) ||
      !survivors.allocate(static_cast<size_t>(capacity) * sizeof(uint64_t))) {
    set_error("hipMalloc failed");
    return 10;
  }

  hipError_t status = hipMemset(rejected.pointer, 0, sieve.candidates);
  if (status == hipSuccess) {
    status = hipMemset(counter.pointer, 0, sizeof(uint64_t));
  }
  if (status == hipSuccess) {
    status = hipMemset(truncated.pointer, 0, sizeof(uint64_t));
  }
  if (status == hipSuccess) {
    status = hipMemcpy(device_starts.pointer, starts.data(),
                       starts.size() * sizeof(uint16_t), hipMemcpyHostToDevice);
  }
  if (status == hipSuccess) {
    status = hipMemcpy(device_steps.pointer, steps.data(),
                       steps.size() * sizeof(uint16_t), hipMemcpyHostToDevice);
  }
  if (status != hipSuccess) {
    set_error("preparing Cunningham sieve buffers", status);
    return 11;
  }

  const auto started = std::chrono::steady_clock::now();
  if (entry_count > 0) {
    const uint32_t blocks = (entry_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    mark_composites_kernel<<<blocks, kThreadsPerBlock>>>(
        static_cast<uint8_t *>(rejected.pointer),
        static_cast<const uint16_t *>(device_starts.pointer),
        static_cast<const uint16_t *>(device_steps.pointer), prime_count,
        sieve.chain_length, sieve.candidates);
  }
  const uint32_t compact_blocks =
      (sieve.candidates + kThreadsPerBlock - 1) / kThreadsPerBlock;
  compact_survivors_kernel<<<compact_blocks, kThreadsPerBlock>>>(
      static_cast<const uint8_t *>(rejected.pointer), sieve.candidates,
      sieve.offset, sieve.stride, static_cast<uint64_t *>(counter.pointer),
      static_cast<uint64_t *>(truncated.pointer),
      static_cast<uint64_t *>(survivors.pointer), capacity);
  status = hipDeviceSynchronize();
  if (status != hipSuccess) {
    set_error("Cunningham sieve kernel", status);
    return 12;
  }
  const auto finished = std::chrono::steady_clock::now();

  uint64_t survivor_count = 0;
  uint64_t was_truncated = 0;
  status = hipMemcpy(&survivor_count, counter.pointer, sizeof(uint64_t),
                     hipMemcpyDeviceToHost);
  if (status == hipSuccess) {
    status = hipMemcpy(&was_truncated, truncated.pointer, sizeof(uint64_t),
                       hipMemcpyDeviceToHost);
  }
  if (status != hipSuccess) {
    set_error("reading Cunningham sieve counters", status);
    return 13;
  }
  survivor_count = std::min<uint64_t>(survivor_count, capacity);

  uint8_t *output = static_cast<uint8_t *>(
      std::malloc(sizeof(HyphenCunninghamSieveHeaderV1) +
                  static_cast<size_t>(survivor_count) * sizeof(uint64_t)));
  if (output == nullptr) {
    set_error("allocating Cunningham sieve output");
    return 14;
  }
  HyphenCunninghamSieveHeaderV1 header{survivor_count, was_truncated};
  std::memcpy(output, &header, sizeof(header));
  if (survivor_count > 0) {
    status = hipMemcpy(output + sizeof(header), survivors.pointer,
                       static_cast<size_t>(survivor_count) * sizeof(uint64_t),
                       hipMemcpyDeviceToHost);
    if (status != hipSuccess) {
      std::free(output);
      set_error("copying Cunningham survivors", status);
      return 15;
    }
  }

  result->output = output;
  result->output_len =
      sizeof(HyphenCunninghamSieveHeaderV1) +
      static_cast<size_t>(survivor_count) * sizeof(uint64_t);
  result->operation_count = sieve.candidates;
  result->device_time_ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(finished - started)
          .count());
  return 0;
}

int32_t execute_request(const HyphenExecuteRequestV1 *request,
                        HyphenExecuteResultV1 *result) {
  if (request == nullptr || result == nullptr) {
    set_error("execute request or result is null");
    return 1;
  }
  if (request->kernel_id != HYPHEN_KERNEL_CUNNINGHAM_SIEVE_V1) {
    set_error("unsupported accelerator kernel");
    return 2;
  }
  return execute_cunningham_sieve(request, result);
}

void free_result(uint8_t *output, size_t /*output_len*/) { std::free(output); }

size_t get_last_error(uint8_t *buffer, size_t capacity) {
  const size_t required = last_error.size();
  if (buffer != nullptr && capacity > 0) {
    const size_t copied = std::min(required, capacity - 1);
    std::memcpy(buffer, last_error.data(), copied);
    buffer[copied] = 0;
  }
  return required;
}

} // namespace

extern "C" HYPHEN_BACKEND_EXPORT int32_t
hyphen_backend_get_api(uint32_t requested_abi,
                       const HyphenBackendApiV1 **api) {
  if (api == nullptr) {
    set_error("API output pointer is null");
    return 1;
  }
  if (requested_abi != HYPHEN_ACCELERATOR_ABI_VERSION) {
    set_error("unsupported accelerator ABI version");
    return 2;
  }
  static const HyphenBackendApiV1 backend = {
      HYPHEN_ACCELERATOR_ABI_VERSION,
      sizeof(HyphenBackendApiV1),
      {'h', 'i', 'p'},
      enumerate_devices,
      execute_request,
      free_result,
      get_last_error,
  };
  *api = &backend;
  return 0;
}
