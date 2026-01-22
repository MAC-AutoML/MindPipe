#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "quantization/gemm_cuda.h"
#include "quantization/gemv_cuda.h"
#include "quantization_new/gemm/gemm_cuda.h"
#include "quantization_new/gemv/gemv_cuda.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gemm_forward_cuda", &gemm_forward_cuda, "Quantized GEMM kernel.");
    m.def("gemv_forward_cuda", &gemv_forward_cuda, "Quantized GEMV kernel.");
    m.def("gemm_forward_cuda_new", &gemm_forward_cuda_new, "New quantized GEMM kernel.");
    m.def("gemv_forward_cuda_new", &gemv_forward_cuda_new, "New quantized GEMV kernel.");
}
