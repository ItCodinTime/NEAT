#include "neat_core/step.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

namespace neat_core {

namespace {

float dot(const float* a, const float* b, std::size_t n) {
  float value = 0.0f;
  for (std::size_t i = 0; i < n; ++i) {
    value += a[i] * b[i];
  }
  return value;
}

float l2_norm(const float* a, std::size_t n, float eps = 0.0f) {
  return std::sqrt(std::max(dot(a, a, n) + eps, 0.0f));
}

}  // namespace

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
    bool decouple_weight_decay) {
  const std::string mode = nce_mode == nullptr ? "projection" : nce_mode;
  const float grad_norm = l2_norm(grad, n, 0.0f);
  const float momentum_norm = l2_norm(momentum, n, 0.0f);
  float conflict_ratio = 0.0f;

  if (grad_norm > eps && momentum_norm > eps) {
    const float cosine = dot(grad, momentum, n) / ((grad_norm * momentum_norm) + eps);
    conflict_ratio = std::max(0.0f, -cosine);
  }

  if (mode == "off") {
    std::fill(nce, nce + n, 0.0f);
  } else {
    if (mode == "cosine") {
      for (std::size_t i = 0; i < n; ++i) {
        nce[i] = -alpha * conflict_ratio * grad[i];
      }
    } else {
      const float denom = dot(momentum, momentum, n) + eps;
      const float scale = denom <= eps ? 0.0f : dot(grad, momentum, n) / denom;
      for (std::size_t i = 0; i < n; ++i) {
        nce[i] = -alpha * conflict_ratio * scale * momentum[i];
      }
    }

    const float nce_norm = l2_norm(nce, n, 0.0f);
    const float clip_limit = nce_clip_ratio * grad_norm;
    if (nce_norm > clip_limit && nce_norm > eps) {
      const float clip_scale = clip_limit / nce_norm;
      for (std::size_t i = 0; i < n; ++i) {
        nce[i] *= clip_scale;
      }
    }
  }

  for (std::size_t i = 0; i < n; ++i) {
    const float update_direction = grad[i] + nce[i];
    momentum[i] = (beta * momentum[i]) + ((1.0f - beta) * update_direction);
  }

  if (decouple_weight_decay && weight_decay > 0.0f) {
    const float decay_scale = 1.0f - (learning_rate * weight_decay);
    for (std::size_t i = 0; i < n; ++i) {
      param[i] = (param[i] * decay_scale) - (learning_rate * momentum[i]);
    }
  } else if (weight_decay > 0.0f) {
    for (std::size_t i = 0; i < n; ++i) {
      const float effective_grad = momentum[i] + (weight_decay * param[i]);
      param[i] -= learning_rate * effective_grad;
    }
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      param[i] -= learning_rate * momentum[i];
    }
  }

  StepStats stats;
  stats.grad_norm = grad_norm;
  stats.update_norm = l2_norm(momentum, n, 0.0f);
  stats.nce_norm = l2_norm(nce, n, 0.0f);
  stats.conflict_ratio = conflict_ratio;
  return stats;
}

}  // namespace neat_core
