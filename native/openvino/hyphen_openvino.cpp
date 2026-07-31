#include "hyphen_accelerator.h"

#include <openvino/openvino.hpp>

#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;

void set_error(const std::string &message) { last_error = message; }

template <size_t N> void copy_text(uint8_t (&destination)[N], const std::string &text) {
  std::memset(destination, 0, N);
  std::snprintf(reinterpret_cast<char *>(destination), N, "%s", text.c_str());
}

bool is_supported_device(const std::string &name) {
  return name == "NPU" || name.rfind("NPU.", 0) == 0 || name == "GPU" ||
         name.rfind("GPU.", 0) == 0;
}

std::vector<std::string> compatible_devices(ov::Core &core) {
  std::vector<std::string> devices;
  for (const std::string &name : core.get_available_devices()) {
    if (is_supported_device(name)) {
      devices.push_back(name);
    }
  }
  return devices;
}

int32_t enumerate_devices(HyphenDeviceInfoV1 *output, uint32_t capacity,
                          uint32_t *count) {
  if (count == nullptr) {
    set_error("count pointer is null");
    return 1;
  }
  try {
    ov::Core core;
    const std::vector<std::string> devices = compatible_devices(core);
    *count = static_cast<uint32_t>(devices.size());
    if (output == nullptr || capacity == 0) {
      return 0;
    }
    if (capacity < *count) {
      set_error("device output capacity is too small");
      return 2;
    }
    const ov::Version version = ov::get_openvino_version();
    for (uint32_t ordinal = 0; ordinal < *count; ++ordinal) {
      const std::string &device_id = devices[ordinal];
      std::string full_name = device_id;
      try {
        full_name = core.get_property(device_id, ov::device::full_name);
      } catch (const std::exception &) {
        // A stable OpenVINO device identifier is still available.
      }
      std::string runtime = "OpenVINO ";
      runtime += version.buildNumber == nullptr ? "unknown" : version.buildNumber;
      HyphenDeviceInfoV1 &device = output[ordinal];
      std::memset(&device, 0, sizeof(device));
      device.struct_size = sizeof(device);
      device.device_ordinal = ordinal;
      device.device_kind = device_id.rfind("NPU", 0) == 0
                               ? HYPHEN_DEVICE_KIND_NPU
                               : HYPHEN_DEVICE_KIND_GPU;
      device.hardware_accelerated = 1;
      device.capability_mask = HYPHEN_CAP_DIFFUSION_Q12_V1;
      copy_text(device.backend, "intel-openvino");
      copy_text(device.vendor, "Intel");
      copy_text(device.name, full_name);
      copy_text(device.stable_id, device_id);
      copy_text(device.runtime, runtime);
    }
    return 0;
  } catch (const std::exception &error) {
    set_error(std::string("OpenVINO device enumeration failed: ") + error.what());
    return 3;
  }
}

std::shared_ptr<ov::Model> build_diffusion_model(size_t count,
                                                 uint32_t alpha_q12) {
  using namespace ov::opset13;
  auto input = std::make_shared<Parameter>(ov::element::i32, ov::Shape{count});
  std::vector<int32_t> left_indices(count);
  std::vector<int32_t> right_indices(count);
  for (size_t index = 0; index < count; ++index) {
    left_indices[index] = static_cast<int32_t>(index == 0 ? count - 1 : index - 1);
    right_indices[index] = static_cast<int32_t>(index + 1 == count ? 0 : index + 1);
  }
  auto axis = Constant::create(ov::element::i32, ov::Shape{}, {0});
  auto left_index =
      Constant::create(ov::element::i32, ov::Shape{count}, left_indices);
  auto right_index =
      Constant::create(ov::element::i32, ov::Shape{count}, right_indices);
  auto left = std::make_shared<Gather>(input, left_index, axis);
  auto right = std::make_shared<Gather>(input, right_index, axis);
  auto alpha = Constant::create(ov::element::i32, ov::Shape{},
                                {static_cast<int32_t>(alpha_q12)});
  auto center_weight = Constant::create(
      ov::element::i32, ov::Shape{}, {4096 - static_cast<int32_t>(2 * alpha_q12)});
  auto scale = Constant::create(ov::element::i32, ov::Shape{}, {4096});
  auto center_term = std::make_shared<Multiply>(input, center_weight);
  auto left_term = std::make_shared<Multiply>(left, alpha);
  auto right_term = std::make_shared<Multiply>(right, alpha);
  auto sum = std::make_shared<Add>(std::make_shared<Add>(center_term, left_term),
                                   right_term);
  auto output = std::make_shared<Divide>(sum, scale);
  return std::make_shared<ov::Model>(ov::OutputVector{output},
                                     ov::ParameterVector{input},
                                     "hyphen-diffusion-q12-v1");
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
  if (request->kernel_id != HYPHEN_KERNEL_DIFFUSION_Q12_V1 ||
      request->input == nullptr || request->input_len % sizeof(int32_t) != 0) {
    set_error("unsupported kernel or malformed input");
    return 11;
  }
  const size_t count = request->input_len / sizeof(int32_t);
  if (count < 3 || count > static_cast<size_t>(INT32_MAX) ||
      request->iterations == 0 || request->iterations > 1024 ||
      request->alpha_q12 > 2048) {
    set_error("request is outside the diffusion-q12-v1 profile");
    return 12;
  }
  const auto *input = reinterpret_cast<const int32_t *>(request->input);
  for (size_t index = 0; index < count; ++index) {
    if (input[index] < 0 || input[index] > 262143) {
      set_error("input cell is outside 0..=262143");
      return 13;
    }
  }
  if (count > std::numeric_limits<uint64_t>::max() / request->iterations / 6) {
    set_error("operation count overflow");
    return 14;
  }

  try {
    ov::Core core;
    const std::vector<std::string> devices = compatible_devices(core);
    if (request->device_ordinal >= devices.size()) {
      set_error("OpenVINO device ordinal does not exist");
      return 15;
    }
    const std::shared_ptr<ov::Model> model =
        build_diffusion_model(count, request->alpha_q12);
    ov::CompiledModel compiled =
        core.compile_model(model, devices[request->device_ordinal]);
    ov::InferRequest infer = compiled.create_infer_request();
    std::vector<int32_t> current(input, input + count);
    const auto started = std::chrono::steady_clock::now();
    for (uint32_t iteration = 0; iteration < request->iterations; ++iteration) {
      ov::Tensor input_tensor(ov::element::i32, ov::Shape{count}, current.data());
      infer.set_input_tensor(input_tensor);
      infer.infer();
      const ov::Tensor output_tensor = infer.get_output_tensor();
      const int32_t *values = output_tensor.data<const int32_t>();
      std::memcpy(current.data(), values, request->input_len);
    }
    const auto elapsed = std::chrono::steady_clock::now() - started;
    auto *output = static_cast<uint8_t *>(std::malloc(request->input_len));
    if (output == nullptr) {
      set_error("cannot allocate host result buffer");
      return 16;
    }
    std::memcpy(output, current.data(), request->input_len);
    result->output = output;
    result->output_len = request->input_len;
    result->operation_count =
        static_cast<uint64_t>(count) * request->iterations * UINT64_C(6);
    result->device_time_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
    return 0;
  } catch (const std::exception &error) {
    set_error(std::string("OpenVINO execution failed: ") + error.what());
    return 17;
  }
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
    {'i', 'n', 't', 'e', 'l', '-', 'o', 'p', 'e', 'n', 'v', 'i', 'n', 'o', 0},
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
