#include "neat_core/step.h"

#include <stdexcept>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

void validate_array(const py::array_t<float, py::array::c_style>& array, const char* name) {
  if (array.ndim() != 1) {
    throw std::invalid_argument(std::string(name) + " must be a 1D float32 array");
  }
}

}  // namespace

PYBIND11_MODULE(_neat_core, m) {
  m.doc() = "Native CPU kernels for the NEAT optimizer";

  m.def(
      "cpu_step_inplace",
      [](py::array_t<float, py::array::c_style> param,
         py::array_t<float, py::array::c_style | py::array::forcecast> grad,
         py::array_t<float, py::array::c_style> momentum,
         py::array_t<float, py::array::c_style> nce,
         float learning_rate,
         float alpha,
         float beta,
         float eps,
         float weight_decay,
         float nce_clip_ratio,
         const std::string& nce_mode,
         bool decouple_weight_decay) {
        validate_array(param, "param");
        validate_array(grad, "grad");
        validate_array(momentum, "momentum");
        validate_array(nce, "nce");

        if (param.size() != grad.size() || param.size() != momentum.size() ||
            param.size() != nce.size()) {
          throw std::invalid_argument("param, grad, momentum, and nce must match in size");
        }

        auto stats = neat_core::cpu_step_inplace(
            static_cast<float*>(param.mutable_data()),
            static_cast<const float*>(grad.data()),
            static_cast<float*>(momentum.mutable_data()),
            static_cast<float*>(nce.mutable_data()),
            static_cast<std::size_t>(param.size()),
            learning_rate,
            alpha,
            beta,
            eps,
            weight_decay,
            nce_clip_ratio,
            nce_mode.c_str(),
            decouple_weight_decay);

        py::dict out;
        out["grad_norm"] = stats.grad_norm;
        out["update_norm"] = stats.update_norm;
        out["nce_norm"] = stats.nce_norm;
        out["conflict_ratio"] = stats.conflict_ratio;
        return out;
      },
      py::arg("param"),
      py::arg("grad"),
      py::arg("momentum"),
      py::arg("nce"),
      py::arg("learning_rate"),
      py::arg("alpha"),
      py::arg("beta"),
      py::arg("eps"),
      py::arg("weight_decay"),
      py::arg("nce_clip_ratio"),
      py::arg("nce_mode"),
      py::arg("decouple_weight_decay"));
}
