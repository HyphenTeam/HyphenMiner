// HYN3 job setup and winning-header work commitment. Both canonical
// transcripts fit in two BLAKE3 blocks. The caller supplies the domain bytes.
struct PreparedJob {
  Digest seed;
  int64_t weights[2128];
};
static_assert(sizeof(PreparedJob) == 32 + 2128 * 8);

__global__ void hash_work_transcript(HyphenNeuralTranscriptV1 input,
                                     Digest *digest) {
  const unsigned lane = threadIdx.x;
  const uint32_t cv =
      compress_word(blake_iv[lane & 7], input.words[lane], 64, kChunkStart);
  const uint32_t word = compress_word(cv, input.words[lane + 16],
                                      input.length - 64, kChunkEnd | kRoot);
  if (lane < 8)
    digest->words[lane] = word;
}

__device__ void chacha_quarter(uint32_t &a, uint32_t &b, uint32_t &c,
                               uint32_t &d) {
  a += b;
  d ^= a;
  d = __funnelshift_l(d, d, 16);
  c += d;
  b ^= c;
  b = __funnelshift_l(b, b, 12);
  a += b;
  d ^= a;
  d = __funnelshift_l(d, d, 8);
  c += d;
  b ^= c;
  b = __funnelshift_l(b, b, 7);
}

// One thread expands a 16-word ChaCha20 block. The fixed HYN3 16/64/16
// profile uses Q32.32 scales 1/4 and 1/8 and sets the last 16 biases to zero.
__global__ void derive_neural_weights(PreparedJob *job, uint64_t height) {
  const unsigned block = blockIdx.x * blockDim.x + threadIdx.x;
  if (block >= 133)
    return;
  if (block == 132) {
    for (unsigned i = 0; i < 16; ++i)
      job->weights[16 * block + i] = 0;
    return;
  }
  const uint64_t epoch = height ^ (UINT64_C(1) << 63);
  uint32_t initial[16] = {0x61707865, 0x3320646e, 0x79622d32, 0x6b206574};
  for (unsigned i = 0; i < 8; ++i)
    initial[4 + i] = job->seed.words[i];
  initial[12] = block;
  initial[13] = 1;
  initial[14] = uint32_t(epoch);
  initial[15] = uint32_t(epoch >> 32);
  uint32_t x[16];
  for (unsigned i = 0; i < 16; ++i)
    x[i] = initial[i];
  for (unsigned i = 0; i < 10; ++i) {
    for (unsigned c = 0; c < 4; ++c)
      chacha_quarter(x[c], x[4 + c], x[8 + c], x[12 + c]);
    for (unsigned c = 0; c < 4; ++c)
      chacha_quarter(x[c], x[4 + ((c + 1) & 3)], x[8 + ((c + 2) & 3)],
                     x[12 + ((c + 3) & 3)]);
  }
  for (unsigned i = 0; i < 16; ++i) {
    const unsigned index = block * 16 + i;
    const uint32_t word = x[i] + initial[i];
    const unsigned shift = index < 1088 ? 1 : 2;
    job->weights[index] = int64_t(word >> shift) - (kNeuralOne >> (shift + 1));
  }
}

int32_t execute_neural_transcript(const HyphenExecuteRequestV1 *request,
                                  HyphenExecuteResultV1 *result) {
  if (request->struct_size != sizeof(*request) ||
      result->struct_size != sizeof(*result) || !request->input ||
      request->input_len != sizeof(HyphenNeuralTranscriptV1)) {
    set_error("invalid neural transcript layout");
    return 1;
  }
  HyphenNeuralTranscriptV1 input;
  std::memcpy(&input, request->input, sizeof(input));
  if (input.length <= 64 || input.length > 128 || input.reserved) {
    set_error("invalid neural transcript length");
    return 1;
  }
  const bool prepare = request->kernel_id == HYPHEN_KERNEL_NEURAL_PREPARE_V1;
  const size_t bytes = prepare ? sizeof(PreparedJob) : sizeof(Digest);
  auto status = cudaSetDevice(request->device_ordinal);
  if (status != cudaSuccess) {
    set_error("selecting CUDA device", status);
    return 2;
  }
  DeviceBuffer buffer;
  if (!buffer.allocate(bytes)) {
    set_error("allocating CUDA transcript result");
    return 2;
  }
  hash_work_transcript<<<1, 16>>>(input, static_cast<Digest *>(buffer.pointer));
  status = cudaGetLastError();
  if (status == cudaSuccess && prepare) {
    derive_neural_weights<<<3, 64>>>(static_cast<PreparedJob *>(buffer.pointer),
                                     input.height);
    status = cudaGetLastError();
  }
  if (status != cudaSuccess) {
    set_error("launching CUDA job preparation", status);
    return 2;
  }
  auto *output = static_cast<uint8_t *>(std::malloc(bytes));
  if (!output) {
    set_error("allocating transcript output");
    return 2;
  }
  status = cudaMemcpy(output, buffer.pointer, bytes, cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    std::free(output);
    set_error("reading CUDA transcript result", status);
    return 2;
  }
  result->output = output;
  result->output_len = bytes;
  result->operation_count = 1;
  return 0;
}
