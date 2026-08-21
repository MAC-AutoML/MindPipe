#include <torch/extension.h>

#include <torch_npu/csrc/core/NPUStorageImpl.h>
#include <torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include <aclnnop/aclnn_grouped_matmul_swiglu_quant_weight_nz.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

extern "C" {
int InitHugeMemThreadLocal(void* ptr, bool enable);
void UnInitHugeMemThreadLocal(void* ptr, bool enable);
void ReleaseHugeMem(void* ptr, bool enable);
}

namespace {

struct AclTensorDeleter {
    void operator()(aclTensor* tensor) const {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
        }
    }
};

using AclTensorPtr = std::unique_ptr<aclTensor, AclTensorDeleter>;

struct ChunkCall {
    AclTensorPtr weight;
    AclTensorPtr weight_scale;
    AclTensorPtr output;
    AclTensorPtr output_scale;
    AclTensorPtr output_offset;
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
};

struct LaunchState {
    at::Tensor x;
    std::vector<at::Tensor> weights;
    at::Tensor group_list;
    std::vector<at::Tensor> weight_scales;
    at::Tensor x_scale;
    at::Tensor quant_output;
    at::Tensor scale_output;
    at::Tensor offset_output;
    at::Tensor workspace;
    AclTensorPtr x_acl;
    AclTensorPtr x_scale_acl;
    AclTensorPtr group_list_acl;
    std::vector<ChunkCall> calls;
    std::atomic<bool> huge_mem_released{false};
};

struct HugeMemProducerScope {
    explicit HugeMemProducerScope(std::shared_ptr<LaunchState> state)
        : state_(std::move(state)) {
        InitHugeMemThreadLocal(nullptr, false);
    }

    ~HugeMemProducerScope() {
        if (!handed_off_ && !state_->huge_mem_released.exchange(true)) {
            ReleaseHugeMem(nullptr, false);
        }
        UnInitHugeMemThreadLocal(nullptr, false);
    }

    void hand_off() {
        handed_off_ = true;
    }

  private:
    std::shared_ptr<LaunchState> state_;
    bool handed_off_ = false;
};

struct HugeMemReleaseScope {
    explicit HugeMemReleaseScope(std::shared_ptr<LaunchState> state)
        : state_(std::move(state)) {}

    ~HugeMemReleaseScope() {
        if (!state_->huge_mem_released.exchange(true)) {
            ReleaseHugeMem(nullptr, false);
        }
    }

  private:
    std::shared_ptr<LaunchState> state_;
};

void check_status(aclnnStatus status, const char* operation) {
    TORCH_CHECK(status == 0, operation, " failed with aclnnStatus ", status);
}

void check_npu_tensor(const at::Tensor& tensor,
                      at::ScalarType dtype,
                      const char* name) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
                name, " must be an NPU tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has unexpected dtype");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

aclDataType acl_dtype(at::ScalarType dtype) {
    switch (dtype) {
        case at::kChar:
            return ACL_INT8;
        case at::kFloat:
            return ACL_FLOAT;
        case at::kLong:
            return ACL_INT64;
        default:
            TORCH_CHECK(false, "unsupported dtype for aclTensor: ", dtype);
    }
}

AclTensorPtr make_acl_tensor(const at::Tensor& tensor) {
    std::vector<int64_t> view_dims(tensor.sizes().begin(), tensor.sizes().end());
    std::vector<int64_t> strides(tensor.strides().begin(), tensor.strides().end());
    std::vector<int64_t> storage_dims;
    aclFormat format = ACL_FORMAT_ND;

    auto* storage_impl = static_cast<torch_npu::NPUStorageImpl*>(
        tensor.storage().unsafeGetStorageImpl());
    const auto& desc = storage_impl->npu_desc_;
    if (desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ) {
        storage_dims.assign(desc.storage_sizes_.begin(),
                            desc.storage_sizes_.end());
        format = desc.npu_format_;
    } else {
        storage_dims.push_back(
            static_cast<int64_t>(tensor.storage().nbytes() / tensor.itemsize()));
        if (tensor.dim() == 3) {
            format = ACL_FORMAT_NCL;
        } else if (tensor.dim() == 4) {
            format = ACL_FORMAT_NCHW;
        } else if (tensor.dim() == 5) {
            format = ACL_FORMAT_NCDHW;
        }
    }

    aclTensor* result = aclCreateTensor(
        view_dims.data(), view_dims.size(), acl_dtype(tensor.scalar_type()),
        strides.data(), tensor.storage_offset(), format, storage_dims.data(),
        storage_dims.size(), const_cast<void*>(tensor.storage().data()));
    TORCH_CHECK(result != nullptr, "aclCreateTensor returned null");
    return AclTensorPtr(result);
}

std::tuple<at::Tensor, at::Tensor> grouped_swiglu_loop_out(
    const at::Tensor& x,
    const std::vector<at::Tensor>& weights,
    const at::Tensor& group_list,
    const std::vector<at::Tensor>& weight_scales,
    const at::Tensor& x_scale) {
    check_npu_tensor(x, at::kChar, "x");
    check_npu_tensor(group_list, at::kLong, "group_list");
    check_npu_tensor(x_scale, at::kFloat, "x_scale");
    TORCH_CHECK(x.dim() == 2, "x must have shape [tokens, hidden]");
    TORCH_CHECK(group_list.dim() == 1 && group_list.numel() == 1,
                "group_list must have one cumulative boundary");
    TORCH_CHECK(x_scale.dim() == 1 && x_scale.size(0) == x.size(0),
                "x_scale must have one value per token");
    TORCH_CHECK(!weights.empty(), "weights must not be empty");
    TORCH_CHECK(weights.size() == weight_scales.size(),
                "weights and weight_scales must have equal length");

    const int64_t chunks = static_cast<int64_t>(weights.size());
    int64_t output_width = -1;
    for (int64_t index = 0; index < chunks; ++index) {
        const auto& weight = weights[index];
        const auto& scale = weight_scales[index];
        check_npu_tensor(weight, at::kChar, "weight");
        check_npu_tensor(scale, at::kFloat, "weight_scale");
        TORCH_CHECK(weight.dim() == 3 && weight.size(0) == 1,
                    "each weight must have shape [1, hidden, 2 * chunk]");
        TORCH_CHECK(weight.size(1) == x.size(1),
                    "weight hidden dimension does not match x");
        TORCH_CHECK(weight.size(2) % 2 == 0,
                    "weight output width must be divisible by two");
        TORCH_CHECK(scale.dim() == 2 && scale.size(0) == 1 &&
                        scale.size(1) == weight.size(2),
                    "each weight_scale must have shape [1, 2 * chunk]");
        const auto* storage_impl = static_cast<torch_npu::NPUStorageImpl*>(
            weight.storage().unsafeGetStorageImpl());
        TORCH_CHECK(storage_impl->npu_desc_.npu_format_ ==
                        ACL_FORMAT_FRACTAL_NZ,
                    "weight must use ACL_FORMAT_FRACTAL_NZ");
        const int64_t current_width = weight.size(2) / 2;
        if (output_width < 0) {
            output_width = current_width;
        }
        TORCH_CHECK(current_width == output_width,
                    "all chunk output widths must match");
    }

    auto quant_output = at::empty(
        {chunks, x.size(0), output_width}, x.options().dtype(at::kChar));
    auto scale_output = at::empty(
        {chunks, x.size(0)}, x.options().dtype(at::kFloat));
    auto offset_output = at::empty({chunks}, x.options().dtype(at::kFloat));

    auto state = std::make_shared<LaunchState>();
    state->x = x;
    state->weights = weights;
    state->group_list = group_list;
    state->weight_scales = weight_scales;
    state->x_scale = x_scale;
    state->quant_output = quant_output;
    state->scale_output = scale_output;
    state->offset_output = offset_output;
    state->x_acl = make_acl_tensor(state->x);
    state->x_scale_acl = make_acl_tensor(state->x_scale);
    state->group_list_acl = make_acl_tensor(state->group_list);
    state->calls.reserve(chunks);

    const aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    HugeMemProducerScope huge_mem_producer(state);
    uint64_t workspace_capacity = 0;

    for (int64_t index = 0; index < chunks; ++index) {
        state->calls.emplace_back();
        auto& call = state->calls.back();
        call.weight = make_acl_tensor(state->weights[index]);
        call.weight_scale = make_acl_tensor(state->weight_scales[index]);
        call.output = make_acl_tensor(state->quant_output.select(0, index));
        call.output_scale = make_acl_tensor(state->scale_output.select(0, index));
        call.output_offset = make_acl_tensor(state->offset_output.select(0, index));

        check_status(
            aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize(
                state->x_acl.get(), call.weight.get(), nullptr, nullptr,
                call.weight_scale.get(), state->x_scale_acl.get(),
                state->group_list_acl.get(), call.output.get(),
                call.output_scale.get(), call.output_offset.get(),
                &call.workspace_size, &call.executor),
            "aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize");
        TORCH_CHECK(call.executor != nullptr,
                    "GetWorkspaceSize returned a null executor");
        workspace_capacity = std::max(workspace_capacity, call.workspace_size);
    }

    if (workspace_capacity > 0) {
        state->workspace =
            at_npu::native::allocate_workspace(workspace_capacity, stream);
    }

    auto acl_call = [state, stream]() -> int {
        HugeMemReleaseScope release_huge_mem(state);
        void* workspace_ptr = state->workspace.defined()
            ? const_cast<void*>(state->workspace.storage().data())
            : nullptr;
        for (const auto& call : state->calls) {
            check_status(aclnnGroupedMatmulSwigluQuantWeightNZ(
                             workspace_ptr, call.workspace_size,
                             call.executor, stream),
                         "aclnnGroupedMatmulSwigluQuantWeightNZ");
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2(
        "aclnnGroupedMatmulSwigluQuantWeightNZ_loop_out", acl_call);
    huge_mem_producer.hand_off();

    return {quant_output, scale_output};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("forward", &grouped_swiglu_loop_out,
               "Four grouped SwiGLU quant calls writing preallocated outputs");
}
