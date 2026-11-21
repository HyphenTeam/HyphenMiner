// HYN3 uses BLAKE3 on messages of at most two compression blocks (one chunk).
// Sixteen cooperating lanes hold the sixteen message words. Four independent
// columns perform each G step and exchange rows for the diagonal half-round.
constexpr unsigned kCells = 32768;
constexpr unsigned kMaxFinalistsInFlight = 512;
constexpr unsigned kSearchThreads = 128;
constexpr unsigned kHashesPerBlock = kSearchThreads / 16;
constexpr unsigned kStepsPerLaunch =
    1024; // Bound kernel duration on display GPUs.
constexpr unsigned kChunkStart = 1, kChunkEnd = 2, kRoot = 8, kKeyed = 16;
constexpr unsigned kSingleKeyed = kChunkStart | kChunkEnd | kRoot | kKeyed;

struct Digest {
  uint32_t words[8];
};
struct Finalist {
  Digest gate;
  uint32_t index;
};
__device__ __constant__ uint32_t blake_iv[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19};
__device__ __constant__ uint8_t blake_permutation[16] = {
    2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};

__device__ __forceinline__ unsigned hash_mask() {
  return 0xffffu << (threadIdx.x & 16);
}
__device__ __forceinline__ uint32_t word_from(uint32_t value, unsigned lane) {
  return __shfl_sync(hash_mask(), value, lane, 16);
}
__device__ __forceinline__ void blake_g(uint32_t &a, uint32_t &b, uint32_t &c,
                                        uint32_t &d, uint32_t x, uint32_t y) {
  a += b + x;
  d ^= a;
  d = __funnelshift_r(d, d, 16);
  c += d;
  b ^= c;
  b = __funnelshift_r(b, b, 12);
  a += b + y;
  d ^= a;
  d = __funnelshift_r(d, d, 8);
  c += d;
  b ^= c;
  b = __funnelshift_r(b, b, 7);
}

__device__ __forceinline__ uint32_t compress_word(uint32_t cv, uint32_t message,
                                                  unsigned length,
                                                  unsigned flags) {
  const unsigned lane = threadIdx.x & 15, col = lane & 3;
  uint32_t a = word_from(cv, col), b = word_from(cv, col + 4);
  uint32_t c = blake_iv[col];
  uint32_t d = col == 2 ? length : col == 3 ? flags : 0;
  const unsigned perm = blake_permutation[lane];
#pragma unroll
  for (int round = 0; round < 7; ++round) {
    blake_g(a, b, c, d, word_from(message, 2 * col),
            word_from(message, 2 * col + 1));
    b = word_from(b, (col + 1) & 3);
    c = word_from(c, (col + 2) & 3);
    d = word_from(d, (col + 3) & 3);
    blake_g(a, b, c, d, word_from(message, 8 + 2 * col),
            word_from(message, 9 + 2 * col));
    b = word_from(b, (col + 3) & 3);
    c = word_from(c, (col + 2) & 3);
    d = word_from(d, (col + 1) & 3);
    if (round != 6)
      message = word_from(message, perm);
  }
  return lane < 4 ? a ^ c : lane < 8 ? b ^ d : 0;
}

// The terminating NUL is part of every consensus domain string.
template <size_t N>
__device__ uint32_t tagged_word(const char (&tag)[N], const uint8_t *prefix,
                                const uint8_t *tail, unsigned tail_len,
                                unsigned word) {
  uint32_t value = 0;
#pragma unroll
  for (unsigned i = 0; i < 4; ++i) {
    const unsigned pos = word * 4 + i;
    const uint32_t byte = pos < N                   ? uint8_t(tag[pos])
                          : pos < N + 32            ? prefix[pos - N]
                          : pos < N + 32 + tail_len ? tail[pos - N - 32]
                                                    : 0;
    value |= byte << (8 * i);
  }
  return value;
}

template <size_t N>
__device__ uint32_t tagged_digest(const char (&tag)[N], const uint8_t *prefix,
                                  const uint8_t *tail, unsigned tail_len) {
  const unsigned lane = threadIdx.x & 15;
  const uint32_t cv = compress_word(
      blake_iv[lane & 7], tagged_word(tag, prefix, tail, tail_len, lane), 64,
      kChunkStart);
  return compress_word(cv, tagged_word(tag, prefix, tail, tail_len, lane + 16),
                       N + 32 + tail_len - 64, kChunkEnd | kRoot);
}

__device__ uint64_t digest_low(uint32_t word) {
  const uint64_t lo = word_from(word, 0), hi = word_from(word, 1);
  return lo | (hi << 32);
}

__global__ void filter_neural_scores(const HyphenNeuralWorkRequestV1 *work,
                                     const uint64_t *scores,
                                     Finalist *finalists, uint32_t *count) {
  const unsigned index = blockIdx.x * kHashesPerBlock + threadIdx.x / 16;
  if (index >= work->candidates)
    return;
  const uint64_t suffix[] = {work->start + index * work->stride, scores[index]};
  const uint32_t gate =
      tagged_digest("Hyphen/neural-gate/v3",
                    reinterpret_cast<const uint8_t *>(work->seed_words),
                    reinterpret_cast<const uint8_t *>(suffix), sizeof(suffix));
  if (digest_low(gate) >= (UINT64_C(1) << 60))
    return;
  const unsigned lane = threadIdx.x & 15;
  unsigned slot = 0;
  if (lane == 0)
    slot = atomicAdd(count, 1u);
  slot = word_from(slot, 0);
  if (lane < 8)
    finalists[slot].gate.words[lane] = gate;
  if (lane == 0)
    finalists[slot].index = index;
}

// Each active finalist owns 1 MiB. The chain state crosses launch boundaries
// in global memory, so neither phase needs a single long watchdog-prone kernel.
template <bool read_phase>
__global__ void scratch_steps(const Finalist *finalists, unsigned count,
                              Digest *scratch, Digest *states, unsigned start) {
  const unsigned slot = blockIdx.x * kHashesPerBlock + threadIdx.x / 16;
  if (slot >= count)
    return;
  const unsigned lane = threadIdx.x & 15;
  const uint32_t key = lane < 8 ? finalists[slot].gate.words[lane] : 0;
  uint32_t state;
  if (!read_phase && start == 0) {
    constexpr char domain[] = "Hyphen/neural-memory/v3";
    uint32_t message = 0;
    for (unsigned i = 0; i < 4; ++i) {
      const unsigned pos = lane * 4 + i;
      if (pos < sizeof(domain))
        message |= uint32_t(domain[pos]) << (8 * i);
    }
    state = compress_word(key, message, sizeof(domain), kSingleKeyed);
  } else {
    state = lane < 8 ? states[slot].words[lane] : 0;
  }
  Digest *cells = scratch + size_t(slot) * kCells;
  for (unsigned i = 0; i < kStepsPerLaunch; ++i) {
    uint32_t message = state;
    if constexpr (read_phase) {
      const unsigned cell = word_from(state, 0) & (kCells - 1);
      if (lane >= 8)
        message = cells[cell].words[lane - 8];
    } else if (lane < 8) {
      cells[start + i].words[lane] = state;
    }
    state = compress_word(key, message, read_phase ? 64 : 32, kSingleKeyed);
  }
  if (lane < 8)
    states[slot].words[lane] = state;
}

__global__ void collect_winners(const Finalist *finalists, unsigned count,
                                const Digest *states, uint64_t threshold,
                                uint32_t *winners) {
  const unsigned slot = blockIdx.x * kHashesPerBlock + threadIdx.x / 16;
  if (slot >= count)
    return;
  const uint32_t word = tagged_digest(
      "Hyphen/neural-final/v3",
      reinterpret_cast<const uint8_t *>(&finalists[slot].gate),
      reinterpret_cast<const uint8_t *>(&states[slot]), sizeof(Digest));
  if (digest_low(word) < threshold && (threadIdx.x & 15) == 0) {
    const unsigned pos = atomicAdd(winners, 1u);
    winners[pos + 1] = finalists[slot].index;
  }
}

// Sleep the host controller while the display driver runs a batch of kernels.
struct SearchCompletion {
  cudaEvent_t event = nullptr;
  ~SearchCompletion() {
    if (event)
      cudaEventDestroy(event);
  }
  cudaError_t wait() {
    const auto status = cudaEventRecord(event);
    return status == cudaSuccess ? cudaEventSynchronize(event) : status;
  }
};

int32_t execute_neural_search(const HyphenExecuteRequestV1 *request,
                              HyphenExecuteResultV1 *result) {
  if (request->struct_size != sizeof(*request) ||
      result->struct_size != sizeof(*result) || !request->input ||
      request->input_len != sizeof(HyphenNeuralSearchRequestV1)) {
    set_error("invalid neural search layout");
    return 1;
  }
  HyphenNeuralSearchRequestV1 input;
  std::memcpy(&input, request->input, sizeof(input));
  const auto &work = input.work;
  constexpr unsigned max_window = 32768;
  if (!work.stride || !work.candidates || work.candidates > max_window ||
      work.reserved) {
    set_error("invalid neural search window");
    return 1;
  }
  for (auto weight : work.weights) {
    if (weight < -kNeuralOne || weight > kNeuralOne) {
      set_error("invalid neural weight");
      return 1;
    }
  }
  auto check = [](cudaError_t status, const char *name) {
    if (status != cudaSuccess)
      set_error(name, status);
    return status == cudaSuccess;
  };
  if (!check(cudaSetDevice(request->device_ordinal), "selecting CUDA device"))
    return 2;
  SearchCompletion completion;
  if (!check(cudaEventCreateWithFlags(&completion.event,
                                      cudaEventBlockingSync |
                                          cudaEventDisableTiming),
             "creating CUDA completion event"))
    return 2;
  DeviceBuffer job_buffer, score_buffer, finalist_buffer, count_buffer,
      winner_buffer;
  if (!job_buffer.allocate(sizeof(work)) ||
      !score_buffer.allocate(work.candidates * sizeof(uint64_t)) ||
      !finalist_buffer.allocate(work.candidates * sizeof(Finalist)) ||
      !count_buffer.allocate(sizeof(uint32_t)) ||
      !winner_buffer.allocate((work.candidates + 1) * sizeof(uint32_t))) {
    set_error("insufficient CUDA window memory");
    return 2;
  }
  auto *job = static_cast<HyphenNeuralWorkRequestV1 *>(job_buffer.pointer);
  auto *scores = static_cast<uint64_t *>(score_buffer.pointer);
  auto *finalists = static_cast<Finalist *>(finalist_buffer.pointer);
  auto *count = static_cast<uint32_t *>(count_buffer.pointer);
  auto *winners = static_cast<uint32_t *>(winner_buffer.pointer);
  const auto start = std::chrono::steady_clock::now();
  if (!check(cudaMemcpy(job, &work, sizeof(work), cudaMemcpyHostToDevice),
             "uploading CUDA job") ||
      !check(cudaMemset(count, 0, sizeof(*count)), "clearing finalist count") ||
      !check(cudaMemset(winners, 0, sizeof(*winners)), "clearing winner count"))
    return 2;
  score_neural_work<<<(work.candidates + 127) / 128, 128>>>(job, scores);
  if (!check(cudaGetLastError(), "launching neural scores"))
    return 2;
  filter_neural_scores<<<(work.candidates + kHashesPerBlock - 1) /
                             kHashesPerBlock,
                         kSearchThreads>>>(job, scores, finalists, count);
  unsigned total = 0;
  if (!check(cudaGetLastError(), "launching BLAKE3 gate") ||
      !check(completion.wait(), "waiting for BLAKE3 gate") ||
      !check(cudaMemcpy(&total, count, sizeof(total), cudaMemcpyDeviceToHost),
             "reading finalist count"))
    return 2;
  DeviceBuffer scratch_buffer, state_buffer;
  unsigned slots = 0;
  if (total) {
    size_t free_bytes, total_bytes;
    if (!check(cudaMemGetInfo(&free_bytes, &total_bytes),
               "querying CUDA memory"))
      return 2;
    constexpr size_t bytes_per_slot = sizeof(Digest) * kCells;
    slots = static_cast<unsigned>(
        std::min<size_t>(std::min(total, kMaxFinalistsInFlight),
                         free_bytes / (4 * bytes_per_slot)));
    while (slots && !scratch_buffer.allocate(slots * bytes_per_slot)) {
      cudaGetLastError();
      slots /= 2;
    }
    if (!slots || !state_buffer.allocate(slots * sizeof(Digest))) {
      set_error("insufficient CUDA scratchpad memory");
      return 2;
    }
  }
  auto *scratch = static_cast<Digest *>(scratch_buffer.pointer);
  auto *states = static_cast<Digest *>(state_buffer.pointer);
  for (unsigned offset = 0; offset < total; offset += slots) {
    const unsigned n = std::min(slots, total - offset),
                   blocks = (n + kHashesPerBlock - 1) / kHashesPerBlock;
    for (unsigned step = 0; step < kCells; step += kStepsPerLaunch) {
      scratch_steps<false><<<blocks, kSearchThreads>>>(finalists + offset, n,
                                                       scratch, states, step);
      if (!check(cudaGetLastError(), "launching BLAKE3 fill"))
        return 2;
    }
    for (unsigned step = 0; step < kCells; step += kStepsPerLaunch) {
      scratch_steps<true><<<blocks, kSearchThreads>>>(finalists + offset, n,
                                                      scratch, states, step);
      if (!check(cudaGetLastError(), "launching BLAKE3 traversal"))
        return 2;
    }
    collect_winners<<<blocks, kSearchThreads>>>(finalists + offset, n, states,
                                                input.threshold, winners);
    if (!check(cudaGetLastError(), "launching BLAKE3 threshold"))
      return 2;
  }
  unsigned found = 0;
  if (!check(completion.wait(), "waiting for BLAKE3 search") ||
      !check(cudaMemcpy(&found, winners, sizeof(found), cudaMemcpyDeviceToHost),
             "reading CUDA winner count"))
    return 2;
  if (found > work.candidates) {
    set_error("invalid CUDA winner count");
    return 2;
  }
  const size_t bytes = (size_t(found) + 1) * sizeof(uint32_t);
  auto *output = static_cast<uint8_t *>(std::malloc(bytes));
  if (!output) {
    set_error("allocating CUDA winner list");
    return 2;
  }
  if (!check(cudaMemcpy(output, winners, bytes, cudaMemcpyDeviceToHost),
             "reading CUDA winners")) {
    std::free(output);
    return 2;
  }
  result->output = output;
  result->output_len = bytes;
  result->operation_count = work.candidates;
  result->device_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - start)
                               .count();
  return 0;
}
