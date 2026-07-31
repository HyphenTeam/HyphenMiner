#include "hyphen_accelerator.h"

#include <QnnInterface.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {
thread_local std::string last_error;
using GetProviders = Qnn_ErrorHandle_t (*)(const QnnInterface_t ***, uint32_t *);

template <size_t N> void copy_text(uint8_t (&destination)[N], const std::string &text) {
  std::memset(destination, 0, N);
  std::snprintf(reinterpret_cast<char *>(destination), N, "%s", text.c_str());
}

#if defined(_WIN32)
using LibraryHandle = HMODULE;
LibraryHandle open_library(const char *path) { return LoadLibraryA(path); }
void *load_symbol(LibraryHandle handle, const char *name) {
  return reinterpret_cast<void *>(GetProcAddress(handle, name));
}
void close_library(LibraryHandle handle) {
  if (handle != nullptr) {
    FreeLibrary(handle);
  }
}
std::string loader_error() { return "Windows error " + std::to_string(GetLastError()); }
#else
using LibraryHandle = void *;
LibraryHandle open_library(const char *path) { return dlopen(path, RTLD_NOW | RTLD_LOCAL); }
void *load_symbol(LibraryHandle handle, const char *name) { return dlsym(handle, name); }
void close_library(LibraryHandle handle) {
  if (handle != nullptr) {
    dlclose(handle);
  }
}
std::string loader_error() {
  const char *message = dlerror();
  return message == nullptr ? "unknown loader error" : message;
}
#endif

const QnnInterface_t *find_provider(const QnnInterface_t **providers, uint32_t count) {
  for (uint32_t index = 0; index < count; ++index) {
    const QnnInterface_t *provider = providers[index];
    if (provider != nullptr &&
        provider->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
        provider->apiVersion.coreApiVersion.minor >= QNN_API_VERSION_MINOR) {
      return provider;
    }
  }
  return nullptr;
}

int32_t enumerate_devices(HyphenDeviceInfoV1 *devices, uint32_t capacity,
                          uint32_t *count) {
  if (count == nullptr) {
    last_error = "count pointer is null";
    return 1;
  }
  *count = 0;
  const char *configured = std::getenv("HYPHEN_QNN_BACKEND_PATH");
#if defined(_WIN32)
  const char *path = configured == nullptr ? "QnnHtp.dll" : configured;
#else
  const char *path = configured == nullptr ? "libQnnHtp.so" : configured;
#endif
  LibraryHandle library = open_library(path);
  if (library == nullptr) {
    last_error = std::string("cannot load QNN HTP backend ") + path + ": " + loader_error();
    return 0;
  }
  auto get_providers =
      reinterpret_cast<GetProviders>(load_symbol(library, "QnnInterface_getProviders"));
  if (get_providers == nullptr) {
    last_error = "QNN HTP backend has no QnnInterface_getProviders export";
    close_library(library);
    return 0;
  }
  const QnnInterface_t **providers = nullptr;
  uint32_t provider_count = 0;
  Qnn_ErrorHandle_t error = get_providers(&providers, &provider_count);
  const QnnInterface_t *provider =
      error == QNN_SUCCESS ? find_provider(providers, provider_count) : nullptr;
  if (provider == nullptr) {
    last_error = "QNN HTP backend has no compatible core API provider";
    close_library(library);
    return 0;
  }
  const QNN_INTERFACE_VER_TYPE &qnn = provider->QNN_INTERFACE_VER_NAME;
  if (qnn.backendCreate == nullptr || qnn.backendFree == nullptr ||
      qnn.deviceCreate == nullptr || qnn.deviceFree == nullptr) {
    last_error = "QNN HTP provider lacks backend/device lifecycle functions";
    close_library(library);
    return 0;
  }
  Qnn_BackendHandle_t backend = nullptr;
  error = qnn.backendCreate(nullptr, nullptr, &backend);
  if (error != QNN_SUCCESS) {
    last_error = "QNN HTP backend initialization failed with code " +
                 std::to_string(static_cast<uint64_t>(error));
    close_library(library);
    return 0;
  }
  Qnn_DeviceHandle_t device_handle = nullptr;
  error = qnn.deviceCreate(nullptr, nullptr, &device_handle);
  if (error != QNN_SUCCESS) {
    last_error = "QNN HTP device is unavailable on this host (code " +
                 std::to_string(static_cast<uint64_t>(error)) + ")";
    qnn.backendFree(backend);
    close_library(library);
    return 0;
  }

  *count = 1;
  if (devices != nullptr && capacity > 0) {
    HyphenDeviceInfoV1 &device = devices[0];
    std::memset(&device, 0, sizeof(device));
    device.struct_size = sizeof(device);
    device.device_ordinal = 0;
    device.device_kind = HYPHEN_DEVICE_KIND_NPU;
    device.hardware_accelerated = 1;
    // A target-specific graph package is required before this backend can execute the
    // consensus profile. Detection alone must never enable scheduling.
    device.capability_mask = 0;
    copy_text(device.backend, "qualcomm-qnn");
    copy_text(device.vendor, "Qualcomm");
    copy_text(device.name, provider->providerName == nullptr ? "QNN HTP" : provider->providerName);
    copy_text(device.stable_id, "qnn-htp:0");
    copy_text(device.runtime,
              "QNN " + std::to_string(QNN_API_VERSION_MAJOR) + "." +
                  std::to_string(QNN_API_VERSION_MINOR));
  }
  qnn.deviceFree(device_handle);
  qnn.backendFree(backend);
  close_library(library);
  last_error =
      "QNN HTP detected, but the target-specific deterministic graph package is not installed";
  return 0;
}

int32_t execute(const HyphenExecuteRequestV1 *, HyphenExecuteResultV1 *) {
  last_error =
      "QNN execution is disabled until a target-specific deterministic graph package is installed";
  return 20;
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
    {'q', 'u', 'a', 'l', 'c', 'o', 'm', 'm', '-', 'q', 'n', 'n', 0},
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
