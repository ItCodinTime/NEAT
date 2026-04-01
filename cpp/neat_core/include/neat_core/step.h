#pragma once

#include <cstddef>

namespace neat_core {

struct StepStats {
  float grad_norm;
  float update_norm;
  float nce_norm;
  float conflict_ratio;
};

StepStats cpu_step_inplace(
    float* param,
    const float* grad,
    float* momentum,
    float* nce,
    std::size_t n,
    float learning_rate,
    float alpha,
    float beta,
    float eps,
    float weight_decay,
    float nce_clip_ratio,
    const char* nce_mode,
    bool decouple_weight_decay);

}  // namespace neat_core
