// Included by the CUDA backend after its device buffer and error helpers.
constexpr int64_t kNeuralOne = INT64_C(1) << 32;

__device__ uint64_t neural_mix(uint64_t value) {
  value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31);
}

__device__ int64_t neural_mul(int64_t left, int64_t right) {
  const uint64_t high = static_cast<uint64_t>(__mul64hi(left, right));
  const uint64_t low = static_cast<uint64_t>(left) * static_cast<uint64_t>(right);
  return static_cast<int64_t>((high << 32) | (low >> 32));
}

__device__ int64_t neural_tanh(int64_t value) {
  if (value >= 3 * kNeuralOne) return kNeuralOne;
  if (value <= -3 * kNeuralOne) return -kNeuralOne;
  const uint64_t magnitude = value < 0 ? -value : value;
  const uint64_t squared = neural_mul(value, value);
  const uint64_t factor = 27 * kNeuralOne + squared;
  const uint64_t denominator = 27 * kNeuralOne + 9 * squared;
  const uint64_t high = __umul64hi(magnitude, factor);
  const uint64_t low = magnitude * factor;
  // The rational activation is bounded by one. Compute its 33 quotient bits
  // by comparing full 128-bit products, then restore the sign for truncation.
  uint64_t quotient = 0;
  for (int bit = 32; bit >= 0; --bit) {
    const uint64_t trial = quotient | (UINT64_C(1) << bit);
    const uint64_t trial_high = __umul64hi(trial, denominator);
    if (trial_high < high || (trial_high == high && trial * denominator <= low)) {
      quotient = trial;
    }
  }
  return value < 0 ? -static_cast<int64_t>(quotient) : static_cast<int64_t>(quotient);
}

__global__ void score_neural_work(const HyphenNeuralWorkRequestV1 *work,
                                  uint64_t *scores) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= work->candidates) return;
  const uint64_t counter = work->start + index * work->stride;
  int64_t input[16];
  for (unsigned i = 0; i < 16; ++i) {
    const uint64_t noise = neural_mix(work->seed_words[i % 8] ^
        UINT64_C(0x48595048454e5731) ^ (counter * UINT64_C(0x9e3779b9)) ^ i);
    input[i] = static_cast<int64_t>(noise >> 32) * 2 - kNeuralOne;
  }
  int64_t hidden[64];
  for (unsigned row = 0; row < 64; ++row) {
    int64_t sum = work->weights[1024 + row];
    for (unsigned i = 0; i < 16; ++i) {
      sum += neural_mul(work->weights[row * 16 + i], input[i]);
    }
    hidden[row] = neural_tanh(sum);
  }
  uint64_t accumulator = UINT64_C(0x9e3779b97f4a7c15);
  for (unsigned row = 0; row < 16; ++row) {
    int64_t sum = work->weights[2112 + row];
    for (unsigned i = 0; i < 64; ++i) {
      sum += neural_mul(work->weights[1088 + row * 64 + i], hidden[i]);
    }
    accumulator = neural_mix(accumulator ^ static_cast<uint64_t>(neural_tanh(sum)));
  }
  scores[index] = neural_mix(accumulator);
}

int32_t execute_neural_work(const HyphenExecuteRequestV1 *request,
                            HyphenExecuteResultV1 *result) {
  if (request->struct_size != sizeof(*request) || result->struct_size != sizeof(*result) ||
      !request->input || request->input_len != sizeof(HyphenNeuralWorkRequestV1)) {
    set_error("invalid neural work request layout");
    return 1;
  }
  HyphenNeuralWorkRequestV1 work;
  std::memcpy(&work, request->input, sizeof(work));
  if (!work.stride || !work.candidates || work.candidates > 32768 || work.reserved) {
    set_error("invalid neural work window");
    return 2;
  }
  // This bound covers every profile-derived weight and keeps all dot products
  // inside the signed accumulator's range, including both bias terms.
  for (int64_t weight : work.weights) {
    if (weight < -kNeuralOne || weight > kNeuralOne) {
      set_error("neural weight outside the supported fixed-point range");
      return 3;
    }
  }
  const size_t bytes = sizeof(uint64_t) * work.candidates;
  DeviceBuffer device_work, scores;
  auto status = cudaSetDevice(static_cast<int>(request->device_ordinal));
  if (status != cudaSuccess) {
    set_error("cudaSetDevice", status);
    return 4;
  }
  if (!device_work.allocate(sizeof(work)) || !scores.allocate(bytes)) {
    set_error("allocating neural work buffers");
    return 5;
  }
  status = cudaMemcpy(device_work.pointer, &work, sizeof(work), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    set_error("uploading neural work", status);
    return 6;
  }
  const auto start = std::chrono::steady_clock::now();
  score_neural_work<<<(work.candidates + 127) / 128, 128>>>(
      static_cast<const HyphenNeuralWorkRequestV1 *>(device_work.pointer),
      static_cast<uint64_t *>(scores.pointer));
  status = cudaGetLastError();
  if (status == cudaSuccess) status = cudaStreamSynchronize(0);
  if (status != cudaSuccess) {
    set_error("scoring neural work", status);
    return 7;
  }
  const auto elapsed = std::chrono::steady_clock::now() - start;
  auto *output = static_cast<uint8_t *>(std::malloc(bytes));
  if (!output) {
    set_error("allocating neural scores");
    return 8;
  }
  status = cudaMemcpy(output, scores.pointer, bytes, cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    std::free(output);
    set_error("downloading neural scores", status);
    return 9;
  }
  result->output = output;
  result->output_len = bytes;
  result->operation_count = work.candidates;
  result->device_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
  return 0;
}
