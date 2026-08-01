#ifndef HYPHEN_ACCELERATOR_H
#define HYPHEN_ACCELERATOR_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define HYPHEN_BACKEND_EXPORT __declspec(dllexport)
#else
#define HYPHEN_BACKEND_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define HYPHEN_ACCELERATOR_ABI_VERSION 2u
#define HYPHEN_KERNEL_DIFFUSION_2D_Q12_V1 2u
#define HYPHEN_CAP_DIFFUSION_2D_Q12_V1 (UINT64_C(1) << 1)
#define HYPHEN_POUW_GRID_SIDE 64u
#define HYPHEN_POUW_CELL_COUNT                                               \
  (HYPHEN_POUW_GRID_SIDE * HYPHEN_POUW_GRID_SIDE)

#define HYPHEN_DEVICE_KIND_GPU 1u
#define HYPHEN_DEVICE_KIND_NPU 2u
#define HYPHEN_DEVICE_KIND_DSP 3u

typedef struct HyphenDeviceInfoV1 {
  uint32_t struct_size;
  uint32_t device_ordinal;
  uint32_t device_kind;
  uint32_t hardware_accelerated;
  uint64_t capability_mask;
  uint8_t backend[32];
  uint8_t vendor[64];
  uint8_t name[128];
  uint8_t stable_id[64];
  uint8_t runtime[64];
} HyphenDeviceInfoV1;

typedef struct HyphenExecuteRequestV1 {
  uint32_t struct_size;
  uint32_t kernel_id;
  uint32_t device_ordinal;
  uint32_t iterations;
  uint32_t alpha_q12;
  uint32_t reserved;
  const uint8_t *input;
  size_t input_len;
} HyphenExecuteRequestV1;

typedef struct HyphenExecuteResultV1 {
  uint32_t struct_size;
  uint8_t *output;
  size_t output_len;
  uint64_t operation_count;
  uint64_t device_time_ns;
} HyphenExecuteResultV1;

typedef int32_t (*HyphenEnumerateDevicesV1)(HyphenDeviceInfoV1 *devices,
                                            uint32_t capacity,
                                            uint32_t *count);
typedef int32_t (*HyphenExecuteV1)(const HyphenExecuteRequestV1 *request,
                                   HyphenExecuteResultV1 *result);
typedef void (*HyphenFreeResultV1)(uint8_t *output, size_t output_len);
typedef size_t (*HyphenGetLastErrorV1)(uint8_t *buffer, size_t capacity);

typedef struct HyphenBackendApiV1 {
  uint32_t abi_version;
  uint32_t struct_size;
  uint8_t backend[32];
  HyphenEnumerateDevicesV1 enumerate_devices;
  HyphenExecuteV1 execute;
  HyphenFreeResultV1 free_result;
  HyphenGetLastErrorV1 get_last_error;
} HyphenBackendApiV1;

HYPHEN_BACKEND_EXPORT int32_t
hyphen_backend_get_api(uint32_t requested_abi,
                       const HyphenBackendApiV1 **api);

#ifdef __cplusplus
}
#endif

#endif
