#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/core/tritonbackend.h"

namespace triton { namespace backend { namespace romav2_fast_sampler {

namespace {

struct TensorF32 {
  std::vector<float> values;
  std::vector<int64_t> shape;
};

struct TensorI64 {
  std::vector<int64_t> values;
  std::vector<int64_t> shape;
};

struct Candidate {
  float confidence;
  int dir;
  int64_t index;
};

struct SampledCandidate {
  float match[4];
  float confidence;
  float precision_a[4];
  float precision_b[4];
};

TRITONSERVER_Error*
Error(const TRITONSERVER_Error_Code code, const std::string& message)
{
  return TRITONSERVER_ErrorNew(code, message.c_str());
}

void
CheckCuda(cudaError_t status, const std::string& what)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(what + ": " + cudaGetErrorString(status));
  }
}

void
CheckTriton(TRITONSERVER_Error* error, const std::string& what)
{
  if (error != nullptr) {
    const std::string message = TRITONSERVER_ErrorMessage(error);
    TRITONSERVER_ErrorDelete(error);
    throw std::runtime_error(what + ": " + message);
  }
}

void
CopyToHost(
    const void* src, const size_t byte_size,
    const TRITONSERVER_MemoryType memory_type, void* dst)
{
  if (byte_size == 0) {
    return;
  }

  if (memory_type == TRITONSERVER_MEMORY_GPU) {
    CheckCuda(
        cudaMemcpy(dst, src, byte_size, cudaMemcpyDeviceToHost),
        "cudaMemcpyDeviceToHost");
  } else {
    std::memcpy(dst, src, byte_size);
  }
}

TensorF32
ReadFloatInput(TRITONBACKEND_Request* request, const char* input_name)
{
  TRITONBACKEND_Input* input;
  CheckTriton(
      TRITONBACKEND_RequestInput(request, input_name, &input),
      std::string("TRITONBACKEND_RequestInput ") + input_name);

  const char* name;
  TRITONSERVER_DataType datatype;
  const int64_t* shape;
  uint32_t dims_count;
  uint64_t byte_size;
  uint32_t buffer_count;
  CheckTriton(
      TRITONBACKEND_InputProperties(
          input, &name, &datatype, &shape, &dims_count, &byte_size,
          &buffer_count),
      std::string("TRITONBACKEND_InputProperties ") + input_name);

  if (datatype != TRITONSERVER_TYPE_FP32) {
    throw std::runtime_error(std::string(input_name) + " must be FP32");
  }

  TensorF32 result;
  result.shape.assign(shape, shape + dims_count);
  result.values.resize(byte_size / sizeof(float));

  size_t copied = 0;
  for (uint32_t idx = 0; idx < buffer_count; ++idx) {
    const void* buffer;
    size_t buffer_byte_size;
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;
    CheckTriton(
        TRITONBACKEND_InputBuffer(
            input, idx, &buffer, &buffer_byte_size, &memory_type,
            &memory_type_id),
        std::string("TRITONBACKEND_InputBuffer ") + input_name);
    CopyToHost(
        buffer, buffer_byte_size, memory_type,
        reinterpret_cast<char*>(result.values.data()) + copied);
    copied += buffer_byte_size;
  }

  return result;
}

TensorI64
ReadInt64Input(TRITONBACKEND_Request* request, const char* input_name)
{
  TRITONBACKEND_Input* input;
  CheckTriton(
      TRITONBACKEND_RequestInput(request, input_name, &input),
      std::string("TRITONBACKEND_RequestInput ") + input_name);

  const char* name;
  TRITONSERVER_DataType datatype;
  const int64_t* shape;
  uint32_t dims_count;
  uint64_t byte_size;
  uint32_t buffer_count;
  CheckTriton(
      TRITONBACKEND_InputProperties(
          input, &name, &datatype, &shape, &dims_count, &byte_size,
          &buffer_count),
      std::string("TRITONBACKEND_InputProperties ") + input_name);

  if (datatype != TRITONSERVER_TYPE_INT64) {
    throw std::runtime_error(std::string(input_name) + " must be INT64");
  }

  TensorI64 result;
  result.shape.assign(shape, shape + dims_count);
  result.values.resize(byte_size / sizeof(int64_t));

  size_t copied = 0;
  for (uint32_t idx = 0; idx < buffer_count; ++idx) {
    const void* buffer;
    size_t buffer_byte_size;
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;
    CheckTriton(
        TRITONBACKEND_InputBuffer(
            input, idx, &buffer, &buffer_byte_size, &memory_type,
            &memory_type_id),
        std::string("TRITONBACKEND_InputBuffer ") + input_name);
    CopyToHost(
        buffer, buffer_byte_size, memory_type,
        reinterpret_cast<char*>(result.values.data()) + copied);
    copied += buffer_byte_size;
  }

  return result;
}

void
RequireShape(
    const TensorF32& tensor, const std::string& name, const int64_t channels)
{
  if (tensor.shape.size() < 4 || tensor.shape[0] != 1 ||
      tensor.shape[tensor.shape.size() - 1] != channels) {
    throw std::runtime_error(
        name + " expected shape [1,H,W," + std::to_string(channels) +
        "]");
  }
}

float
GridX(const int64_t x, const int64_t width)
{
  return -1.0f + (2.0f * (static_cast<float>(x) + 0.5f) /
                  static_cast<float>(width));
}

float
GridY(const int64_t y, const int64_t height)
{
  return -1.0f + (2.0f * (static_cast<float>(y) + 0.5f) /
                  static_cast<float>(height));
}

void
CopyPrecisionAt(const TensorF32& precision, const int64_t index, float* out)
{
  const float* ptr = precision.values.data() + (index * 4);
  std::copy(ptr, ptr + 4, out);
}

void
SamplePrecisionBilinear(
    const TensorF32& precision, const float nx, const float ny,
    const int64_t height, const int64_t width, float* out)
{
  std::fill(out, out + 4, 0.0f);

  const float x = ((nx + 1.0f) * static_cast<float>(width) - 1.0f) / 2.0f;
  const float y = ((ny + 1.0f) * static_cast<float>(height) - 1.0f) / 2.0f;
  const int64_t x0 = static_cast<int64_t>(std::floor(x));
  const int64_t y0 = static_cast<int64_t>(std::floor(y));
  const float wx = x - static_cast<float>(x0);
  const float wy = y - static_cast<float>(y0);

  const int64_t xs[2] = {x0, x0 + 1};
  const int64_t ys[2] = {y0, y0 + 1};
  const float xw[2] = {1.0f - wx, wx};
  const float yw[2] = {1.0f - wy, wy};

  for (int yy = 0; yy < 2; ++yy) {
    if (ys[yy] < 0 || ys[yy] >= height) {
      continue;
    }
    for (int xx = 0; xx < 2; ++xx) {
      if (xs[xx] < 0 || xs[xx] >= width) {
        continue;
      }
      const float weight = xw[xx] * yw[yy];
      const int64_t index = (ys[yy] * width + xs[xx]) * 4;
      for (int k = 0; k < 4; ++k) {
        out[k] += precision.values[index + k] * weight;
      }
    }
  }
}

std::vector<int64_t>
WeightedSampleWithoutReplacement(
    const std::vector<float>& weights, const int64_t count,
    std::mt19937_64& rng)
{
  if (count <= 0 || weights.empty()) {
    return {};
  }

  const int64_t target_count =
      std::min<int64_t>(count, static_cast<int64_t>(weights.size()));
  std::vector<int64_t> positive;
  std::vector<int64_t> zero_or_invalid;
  positive.reserve(weights.size());
  zero_or_invalid.reserve(weights.size());

  for (int64_t idx = 0; idx < static_cast<int64_t>(weights.size()); ++idx) {
    if (std::isfinite(weights[idx]) && weights[idx] > 0.0f) {
      positive.push_back(idx);
    } else {
      zero_or_invalid.push_back(idx);
    }
  }

  std::vector<int64_t> sampled;
  sampled.reserve(target_count);

  if (positive.empty()) {
    std::vector<int64_t> uniform(weights.size());
    for (int64_t idx = 0; idx < static_cast<int64_t>(weights.size()); ++idx) {
      uniform[idx] = idx;
    }
    std::shuffle(uniform.begin(), uniform.end(), rng);
    uniform.resize(target_count);
    return uniform;
  }

  struct KeyedIndex {
    double key;
    int64_t index;
  };
  std::vector<KeyedIndex> keyed;
  keyed.reserve(positive.size());
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  for (const int64_t idx : positive) {
    double u = uniform(rng);
    while (u <= 0.0) {
      u = uniform(rng);
    }
    keyed.push_back({std::log(u) / static_cast<double>(weights[idx]), idx});
  }
  std::sort(keyed.begin(), keyed.end(), [](const auto& a, const auto& b) {
    return a.key > b.key;
  });

  const int64_t positive_count =
      std::min<int64_t>(target_count, static_cast<int64_t>(keyed.size()));
  for (int64_t idx = 0; idx < positive_count; ++idx) {
    sampled.push_back(keyed[idx].index);
  }

  if (static_cast<int64_t>(sampled.size()) < target_count) {
    std::shuffle(zero_or_invalid.begin(), zero_or_invalid.end(), rng);
    const int64_t remaining = target_count - sampled.size();
    sampled.insert(
        sampled.end(), zero_or_invalid.begin(),
        zero_or_invalid.begin() + std::min<int64_t>(
                                      remaining, zero_or_invalid.size()));
  }

  return sampled;
}

std::vector<Candidate>
BuildCandidates(
    const TensorF32& warp_ab, const TensorF32& overlap_ab,
    const TensorF32& warp_ba, const TensorF32& overlap_ba,
    const int64_t height, const int64_t width)
{
  const int64_t pixels = height * width;
  const float boundary = 1.0f - (1.0f / static_cast<float>(height));
  std::vector<Candidate> candidates;
  candidates.reserve(pixels * 2);

  auto maybe_push = [&](const float confidence, const int dir, const int64_t index) {
    const TensorF32& warp = (dir == 0) ? warp_ab : warp_ba;
    const int64_t y = index / width;
    const int64_t x = index % width;
    const float grid_x = GridX(x, width);
    const float grid_y = GridY(y, height);
    const float warp_x = warp.values[index * 2];
    const float warp_y = warp.values[index * 2 + 1];
    const float max_abs = std::max(
        {std::fabs(grid_x), std::fabs(grid_y), std::fabs(warp_x),
         std::fabs(warp_y)});
    const float masked_confidence =
        (std::isfinite(confidence) && max_abs <= boundary) ? confidence : 0.0f;
    candidates.push_back({masked_confidence, dir, index});
  };

  for (int64_t idx = 0; idx < pixels; ++idx) {
    maybe_push(overlap_ab.values[idx], 0, idx);
    maybe_push(overlap_ba.values[idx], 1, idx);
  }
  return candidates;
}

SampledCandidate
MaterializeCandidate(
    const Candidate& candidate, const TensorF32& warp_ab,
    const TensorF32& precision_ab, const TensorF32& warp_ba,
    const TensorF32& precision_ba, const int64_t height, const int64_t width)
{
  SampledCandidate out{};
  const int64_t index = candidate.index;
  const int64_t y = index / width;
  const int64_t x = index % width;
  const float gx = GridX(x, width);
  const float gy = GridY(y, height);
  out.confidence = candidate.confidence;

  if (candidate.dir == 0) {
    const float wx = warp_ab.values[index * 2];
    const float wy = warp_ab.values[index * 2 + 1];
    out.match[0] = gx;
    out.match[1] = gy;
    out.match[2] = wx;
    out.match[3] = wy;
    SamplePrecisionBilinear(precision_ba, wx, wy, height, width, out.precision_a);
    CopyPrecisionAt(precision_ab, index, out.precision_b);
  } else {
    const float wx = warp_ba.values[index * 2];
    const float wy = warp_ba.values[index * 2 + 1];
    out.match[0] = wx;
    out.match[1] = wy;
    out.match[2] = gx;
    out.match[3] = gy;
    CopyPrecisionAt(precision_ba, index, out.precision_a);
    SamplePrecisionBilinear(precision_ab, wx, wy, height, width, out.precision_b);
  }
  return out;
}

std::vector<float>
KdeWeights(const std::vector<SampledCandidate>& sampled)
{
  constexpr float stddev = 0.1f;
  constexpr float denom = 2.0f * stddev * stddev;
  std::vector<float> weights(sampled.size(), 0.0f);

  for (size_t i = 0; i < sampled.size(); ++i) {
    float density = 0.0f;
    for (size_t j = 0; j < sampled.size(); ++j) {
      float sq_dist = 0.0f;
      for (int k = 0; k < 4; ++k) {
        const float diff = sampled[i].match[k] - sampled[j].match[k];
        sq_dist += diff * diff;
      }
      density += std::exp(-sq_dist / denom);
    }
    float p = 1.0f / (density + 1.0f);
    if (density < 10.0f) {
      p = 1e-7f;
    }
    weights[i] = p;
  }

  return weights;
}

std::vector<SampledCandidate>
SampleRomaCandidates(
    const TensorF32& warp_ab, const TensorF32& overlap_ab,
    const TensorF32& precision_ab, const TensorF32& warp_ba,
    const TensorF32& overlap_ba, const TensorF32& precision_ba,
    const int64_t height, const int64_t width, const int64_t num_corresp,
    const int64_t seed)
{
  if (num_corresp <= 0) {
    return {};
  }

  std::random_device random_device;
  std::mt19937_64 rng(
      seed >= 0 ? static_cast<uint64_t>(seed) : random_device());

  const std::vector<Candidate> candidates =
      BuildCandidates(warp_ab, overlap_ab, warp_ba, overlap_ba, height, width);
  std::vector<float> confidence;
  confidence.reserve(candidates.size());
  for (const Candidate& candidate : candidates) {
    confidence.push_back(candidate.confidence);
  }

  const int64_t first_count =
      std::min<int64_t>(4 * num_corresp, static_cast<int64_t>(candidates.size()));
  const std::vector<int64_t> first_indices =
      WeightedSampleWithoutReplacement(confidence, first_count, rng);

  std::vector<SampledCandidate> expanded;
  expanded.reserve(first_indices.size());
  for (const int64_t idx : first_indices) {
    expanded.push_back(MaterializeCandidate(
        candidates[idx], warp_ab, precision_ab, warp_ba, precision_ba, height,
        width));
  }

  const std::vector<float> balance_weights = KdeWeights(expanded);
  const int64_t final_count =
      std::min<int64_t>(num_corresp, static_cast<int64_t>(expanded.size()));
  const std::vector<int64_t> balanced_indices =
      WeightedSampleWithoutReplacement(balance_weights, final_count, rng);

  std::vector<SampledCandidate> final;
  final.reserve(balanced_indices.size());
  for (const int64_t idx : balanced_indices) {
    final.push_back(expanded[idx]);
  }
  return final;
}

void
WriteOutputs(
    TRITONBACKEND_Response* response,
    const std::vector<SampledCandidate>& selected)
{
  const int64_t count = static_cast<int64_t>(selected.size());

  TRITONBACKEND_Output* matches_output;
  const int64_t matches_shape[2] = {count, 4};
  CheckTriton(
      TRITONBACKEND_ResponseOutput(
          response, &matches_output, "sampled_matches", TRITONSERVER_TYPE_FP32,
          matches_shape, 2),
      "TRITONBACKEND_ResponseOutput sampled_matches");

  TRITONBACKEND_Output* confidence_output;
  const int64_t confidence_shape[1] = {count};
  CheckTriton(
      TRITONBACKEND_ResponseOutput(
          response, &confidence_output, "sampled_confidence",
          TRITONSERVER_TYPE_FP32, confidence_shape, 1),
      "TRITONBACKEND_ResponseOutput sampled_confidence");

  TRITONBACKEND_Output* precision_a_output;
  const int64_t precision_shape[3] = {count, 2, 2};
  CheckTriton(
      TRITONBACKEND_ResponseOutput(
          response, &precision_a_output, "sampled_precision_A",
          TRITONSERVER_TYPE_FP32, precision_shape, 3),
      "TRITONBACKEND_ResponseOutput sampled_precision_A");

  TRITONBACKEND_Output* precision_b_output;
  CheckTriton(
      TRITONBACKEND_ResponseOutput(
          response, &precision_b_output, "sampled_precision_B",
          TRITONSERVER_TYPE_FP32, precision_shape, 3),
      "TRITONBACKEND_ResponseOutput sampled_precision_B");

  float* matches;
  float* confidence;
  float* precision_a;
  float* precision_b;
  TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
  int64_t memory_type_id = 0;
  CheckTriton(
      TRITONBACKEND_OutputBuffer(
          matches_output, reinterpret_cast<void**>(&matches),
          count * 4 * sizeof(float), &memory_type, &memory_type_id),
      "TRITONBACKEND_OutputBuffer sampled_matches");
  memory_type = TRITONSERVER_MEMORY_CPU;
  memory_type_id = 0;
  CheckTriton(
      TRITONBACKEND_OutputBuffer(
          confidence_output, reinterpret_cast<void**>(&confidence),
          count * sizeof(float), &memory_type, &memory_type_id),
      "TRITONBACKEND_OutputBuffer sampled_confidence");
  memory_type = TRITONSERVER_MEMORY_CPU;
  memory_type_id = 0;
  CheckTriton(
      TRITONBACKEND_OutputBuffer(
          precision_a_output, reinterpret_cast<void**>(&precision_a),
          count * 4 * sizeof(float), &memory_type, &memory_type_id),
      "TRITONBACKEND_OutputBuffer sampled_precision_A");
  memory_type = TRITONSERVER_MEMORY_CPU;
  memory_type_id = 0;
  CheckTriton(
      TRITONBACKEND_OutputBuffer(
          precision_b_output, reinterpret_cast<void**>(&precision_b),
          count * 4 * sizeof(float), &memory_type, &memory_type_id),
      "TRITONBACKEND_OutputBuffer sampled_precision_B");

  for (int64_t out_idx = 0; out_idx < count; ++out_idx) {
    const auto& candidate = selected[out_idx];
    confidence[out_idx] = candidate.confidence;
    std::copy(candidate.match, candidate.match + 4, matches + out_idx * 4);
    std::copy(
        candidate.precision_a, candidate.precision_a + 4,
        precision_a + out_idx * 4);
    std::copy(
        candidate.precision_b, candidate.precision_b + 4,
        precision_b + out_idx * 4);
  }
}

void
ProcessRequest(TRITONBACKEND_Request* request, TRITONBACKEND_Response* response)
{
  TensorF32 warp_ab = ReadFloatInput(request, "warp_AB");
  TensorF32 overlap_ab = ReadFloatInput(request, "overlap_AB");
  TensorF32 precision_ab = ReadFloatInput(request, "precision_AB");
  TensorF32 warp_ba = ReadFloatInput(request, "warp_BA");
  TensorF32 overlap_ba = ReadFloatInput(request, "overlap_BA");
  TensorF32 precision_ba = ReadFloatInput(request, "precision_BA");
  TensorI64 num_corresp = ReadInt64Input(request, "num_corresp");
  TensorI64 seed = ReadInt64Input(request, "seed");

  RequireShape(warp_ab, "warp_AB", 2);
  RequireShape(overlap_ab, "overlap_AB", 1);
  RequireShape(precision_ab, "precision_AB", 2);
  RequireShape(warp_ba, "warp_BA", 2);
  RequireShape(overlap_ba, "overlap_BA", 1);
  RequireShape(precision_ba, "precision_BA", 2);

  const int64_t height = warp_ab.shape[1];
  const int64_t width = warp_ab.shape[2];
  if (warp_ba.shape[1] != height || warp_ba.shape[2] != width ||
      overlap_ab.shape[1] != height || overlap_ab.shape[2] != width ||
      overlap_ba.shape[1] != height || overlap_ba.shape[2] != width ||
      precision_ab.shape[1] != height || precision_ab.shape[2] != width ||
      precision_ba.shape[1] != height || precision_ba.shape[2] != width) {
    throw std::runtime_error("all dense tensors must share H/W");
  }

  const int64_t requested =
      num_corresp.values.empty() ? 0 : std::max<int64_t>(0, num_corresp.values[0]);
  const int64_t sample_seed = seed.values.empty() ? -1 : seed.values[0];
  std::vector<SampledCandidate> selected = SampleRomaCandidates(
      warp_ab, overlap_ab, precision_ab, warp_ba, overlap_ba, precision_ba,
      height, width, requested, sample_seed);
  WriteOutputs(response, selected);
}

}  // namespace

class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state)
  {
    try {
      *state = new ModelState(triton_model);
    }
    catch (const BackendModelException& ex) {
      RETURN_IF_ERROR(ex.err_);
    }
    return nullptr;
  }

 private:
  explicit ModelState(TRITONBACKEND_Model* triton_model)
      : BackendModel(triton_model)
  {
  }
};

class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state)
  {
    try {
      *state = new ModelInstanceState(model_state, triton_model_instance);
    }
    catch (const BackendModelInstanceException& ex) {
      RETURN_IF_ERROR(ex.err_);
    }
    return nullptr;
  }

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance)
      : BackendModelInstance(model_state, triton_model_instance)
  {
  }
};

extern "C" {

TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* name;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &name));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());
  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  ModelState* state = nullptr;
  RETURN_IF_ERROR(ModelState::Create(model, &state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(state)));
  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  delete reinterpret_cast<ModelState*>(vstate);
  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  auto* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  ModelInstanceState* instance_state = nullptr;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));
  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  delete reinterpret_cast<ModelInstanceState*>(vstate);
  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    TRITONBACKEND_Response* response = nullptr;
    TRITONSERVER_Error* response_error =
        TRITONBACKEND_ResponseNew(&response, request);

    if (response_error == nullptr) {
      try {
        ProcessRequest(request, response);
      }
      catch (const std::exception& ex) {
        response_error = Error(TRITONSERVER_ERROR_INTERNAL, ex.what());
      }
    }

    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSend(
            response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, response_error),
        "failed to send response");
    if (response_error != nullptr) {
      TRITONSERVER_ErrorDelete(response_error);
    }
    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(
            request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }
  return nullptr;
}

}  // extern "C"

}}}  // namespace triton::backend::romav2_fast_sampler
