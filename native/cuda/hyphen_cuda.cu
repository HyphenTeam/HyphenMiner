#include "hyphen_accelerator.h"

#include <cuda_runtime.h>

#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

namespace {
thread_local std::string last_error;

void set_error(const char *operation, cudaError_t error) {
  last_error = std::string(operation) + ": " + cudaGetErrorString(error);
}

void set_error(const char *message) { last_error = message; }

template <size_t N> void copy_text(uint8_t (&destination)[N], const char *text) {
  std::memset(destination, 0, N);
  if (text != nullptr) {
    std::snprintf(reinterpret_cast<char *>(destination), N, "%s", text);
  }
}

__global__ void diffusion_step(const int32_t *input, int32_t *output,
                               size_t count, uint32_t alpha_q12) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const size_t left_index = index == 0 ? count - 1 : index - 1;
  const size_t right_index = index + 1 == count ? 0 : index + 1;
  const int64_t alpha = static_cast<int64_t>(alpha_q12);
  const int64_t center_weight = INT64_C(4096) - INT64_C(2) * alpha;
  const int64_t numerator =
      center_weight * input[index] + alpha * input[left_index] +
      alpha * input[right_index];
  output[index] = static_cast<int32_t>(numerator / INT64_C(4096));
}

int32_t enumerate_devices(HyphenDeviceInfoV1 *devices, uint32_t capacity,
                          uint32_t *count) {
  if (count == nullptr) {
    set_error("count pointer is null");
    return 1;
  }
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);
  if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver) {
    cudaGetLastError();
    *count = 0;
    return 0;
  }
  if (error != cudaSuccess) {
    set_error("cudaGetDeviceCount", error);
    return 2;
  }
  *count = static_cast<uint32_t>(device_count);
  if (devices == nullptr || capacity == 0) {
    return 0;
  }
  if (capacity < *count) {
    set_error("device output capacity is too small");
    return 3;
  }

  int runtime_version = 0;
  cudaRuntimeGetVersion(&runtime_version);
  for (int ordinal = 0; ordinal < device_count; ++ordinal) {
    cudaDeviceProp properties{};
    error = cudaGetDeviceProperties(&properties, ordinal);
    if (error != cudaSuccess) {
      set_error("cudaGetDeviceProperties", error);
      return 4;
    }
    char pci_id[64]{};
    if (cudaDeviceGetPCIBusId(pci_id, sizeof(pci_id), ordinal) != cudaSuccess) {
      std::snprintf(pci_id, sizeof(pci_id), "cuda:%d", ordinal);
      cudaGetLastError();
    }
    char runtime[64]{};
    std::snprintf(runtime, sizeof(runtime), "CUDA %d.%d", runtime_version / 1000,
                  (runtime_version % 1000) / 10);

    HyphenDeviceInfoV1 &device = devices[ordinal];
    std::memset(&device, 0, sizeof(device));
    device.struct_size = sizeof(device);
    device.device_ordinal = static_cast<uint32_t>(ordinal);
    device.device_kind = HYPHEN_DEVICE_KIND_GPU;
    device.hardware_accelerated = 1;
    device.capability_mask = HYPHEN_CAP_DIFFUSION_Q12_V1;
    copy_text(device.backend, "nvidia-cuda");
    copy_text(device.vendor, "NVIDIA");
    copy_text(device.name, properties.name);
    copy_text(device.stable_id, pci_id);
    copy_text(device.runtime, runtime);
  }
  return 0;
}

int32_t execute(const HyphenExecuteRequestV1 *request,
                HyphenExecuteResultV1 *result) {
  if (request == nullptr || result == nullptr ||
      request->struct_size < sizeof(HyphenExecuteRequestV1) ||
      result->struct_size < sizeof(HyphenExecuteResultV1)) {
    set_error("invalid request or result structure");
    return 10;
  }
  result->output = nullptr;
  result->output_len = 0;
  result->operation_count = 0;
  result->device_time_ns = 0;
  if (request->kernel_id != HYPHEN_KERNEL_DIFFUSION_Q12_V1) {
    set_error("unsupported kernel ID");
    return 11;
  }
  if (request->input == nullptr || request->input_len % sizeof(int32_t) != 0) {
    set_error("input must be a non-null little-endian i32 array");
    return 12;
  }
  const size_t count = request->input_len / sizeof(int32_t);
  if (count < 3 || count > static_cast<size_t>(UINT32_MAX)) {
    set_error("input cell count must be in 3..=UINT32_MAX");
    return 13;
  }
  if (request->iterations == 0 || request->iterations > 1024 ||
      request->alpha_q12 > 2048) {
    set_error("iterations or alpha_q12 is outside the consensus profile");
    return 14;
  }
  const auto *host_input = reinterpret_cast<const int32_t *>(request->input);
  for (size_t index = 0; index < count; ++index) {
    if (host_input[index] < 0 || host_input[index] > 262143) {
      set_error("input cell is outside 0..=262143");
      return 15;
    }
  }
  if (count > std::numeric_limits<uint64_t>::max() / request->iterations / 6) {
    set_error("operation count overflow");
    return 16;
  }

  cudaError_t error = cudaSetDevice(static_cast<int>(request->device_ordinal));
  if (error != cudaSuccess) {
    set_error("cudaSetDevice", error);
    return 17;
  }
  int32_t *first = nullptr;
  int32_t *second = nullptr;
  cudaEvent_t started = nullptr;
  cudaEvent_t finished = nullptr;
  uint8_t *host_output = nullptr;
  const size_t bytes = count * sizeof(int32_t);
  int32_t status = 18;

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    error = (call);                                                              \
    if (error != cudaSuccess) {                                                  \
      set_error(#call, error);                                                   \
      goto cleanup;                                                              \
    }                                                                            \
  } while (0)

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&first), bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&second), bytes));
  CUDA_CHECK(cudaMemcpy(first, request->input, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventCreate(&started));
  CUDA_CHECK(cudaEventCreate(&finished));
  CUDA_CHECK(cudaEventRecord(started));

  {
    int32_t *current = first;
    int32_t *next = second;
    constexpr uint32_t block_size = 256;
    const uint32_t grid_size =
        static_cast<uint32_t>((count + block_size - 1) / block_size);
    for (uint32_t iteration = 0; iteration < request->iterations; ++iteration) {
      diffusion_step<<<grid_size, block_size>>>(current, next, count,
                                                 request->alpha_q12);
      CUDA_CHECK(cudaGetLastError());
      int32_t *temporary = current;
      current = next;
      next = temporary;
    }
    CUDA_CHECK(cudaEventRecord(finished));
    CUDA_CHECK(cudaEventSynchronize(finished));
    host_output = static_cast<uint8_t *>(std::malloc(bytes));
    if (host_output == nullptr) {
      set_error("cannot allocate host result buffer");
      status = 19;
      goto cleanup;
    }
    CUDA_CHECK(cudaMemcpy(host_output, current, bytes, cudaMemcpyDeviceToHost));
  }

  {
    float elapsed_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, started, finished));
    result->output = host_output;
    result->output_len = bytes;
    result->operation_count =
        static_cast<uint64_t>(count) * request->iterations * UINT64_C(6);
    result->device_time_ns = static_cast<uint64_t>(elapsed_ms * 1000000.0F);
    host_output = nullptr;
    status = 0;
  }

cleanup:
  std::free(host_output);
  if (finished != nullptr) {
    cudaEventDestroy(finished);
  }
  if (started != nullptr) {
    cudaEventDestroy(started);
  }
  cudaFree(second);
  cudaFree(first);
  return status;
#undef CUDA_CHECK
}

void free_result(uint8_t *output, size_t) { std::free(output); }

size_t get_last_error(uint8_t *buffer, size_t capacity) {
  if (buffer == nullptr || capacity == 0) {
    return last_error.size() + 1;
  }
  const size_t written = last_error.copy(reinterpret_cast<char *>(buffer), capacity - 1);
  buffer[written] = 0;
  return written + 1;
}

const HyphenBackendApiV1 api = {
    HYPHEN_ACCELERATOR_ABI_VERSION,
    sizeof(HyphenBackendApiV1),
    {'n', 'v', 'i', 'd', 'i', 'a', '-', 'c', 'u', 'd', 'a', 0},
    enumerate_devices,
    execute,
    free_result,
    get_last_error,
};
} // namespace

extern "C" HYPHEN_BACKEND_EXPORT int32_t
hyphen_backend_get_api(uint32_t requested_abi, const HyphenBackendApiV1 **out) {
  if (out == nullptr || requested_abi != HYPHEN_ACCELERATOR_ABI_VERSION) {
    return 1;
  }
  *out = &api;
  return 0;
}
