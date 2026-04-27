import gc
import torch
import torch.nn as nn

from transformers.models.bloom.modeling_bloom import BloomBlock, BloomGelu
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.activations import GELUActivation, GELUTanh
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2DecoderLayer
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer
try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLRMSNorm
except ImportError:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2RMSNorm as Qwen2_5_VLRMSNorm
try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
except ImportError:
    Qwen3_5DecoderLayer = tuple()  # type: ignore[assignment]
try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
except ImportError:
    Qwen3_5RMSNorm = tuple()  # type: ignore[assignment]
try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNormGated
except ImportError:
    Qwen3_5RMSNormGated = tuple()  # type: ignore[assignment]

from .qmodule import ScaledActivation
from .forward_utils import forward_in_chunks
from ..utils.module import get_op_by_name, get_op_name, set_op_by_name
from ..utils.device import resolve_device

__all__ = ["auto_scale_block", "apply_scale"]


def _is_llama_family_decoder(module) -> bool:
    return (
        hasattr(module, "self_attn")
        and hasattr(module, "mlp")
        and hasattr(module, "input_layernorm")
        and hasattr(module, "post_attention_layernorm")
        and hasattr(module.self_attn, "q_proj")
        and hasattr(module.self_attn, "k_proj")
        and hasattr(module.self_attn, "v_proj")
        and hasattr(module.self_attn, "o_proj")
        and hasattr(module.mlp, "gate_proj")
        and hasattr(module.mlp, "up_proj")
        and hasattr(module.mlp, "down_proj")
    )


def _is_norm_like(module) -> bool:
    if isinstance(
        module,
        (nn.LayerNorm, LlamaRMSNorm, Qwen2RMSNorm, Qwen2_5_VLRMSNorm, Qwen3_5RMSNormGated),
    ):
        return True
    return hasattr(module, "weight") and module.__class__.__name__.endswith("RMSNorm")


def _is_qwen2_vl_vision_block(module) -> bool:
    return (
        hasattr(module, "norm1")
        and hasattr(module, "norm2")
        and hasattr(module, "attn")
        and hasattr(module, "mlp")
        and hasattr(module.attn, "qkv")
        and hasattr(module.attn, "proj")
        and (
            (hasattr(module.mlp, "fc1") and hasattr(module.mlp, "fc2"))
            or (
                hasattr(module.mlp, "linear_fc1")
                and hasattr(module.mlp, "linear_fc2")
            )
            or (
                hasattr(module.mlp, "gate_proj")
                and hasattr(module.mlp, "up_proj")
                and hasattr(module.mlp, "down_proj")
            )
        )
    )


def _is_qwen2_vl_patch_merger(module) -> bool:
    mlp = getattr(module, "mlp", None)
    return (
        hasattr(module, "ln_q")
        and isinstance(mlp, nn.Sequential)
        and len(mlp) >= 3
        and isinstance(mlp[0], nn.Linear)
        and isinstance(mlp[1], (nn.GELU, GELUActivation))
        and isinstance(mlp[2], nn.Linear)
    )


def _is_qwen3_vl_patch_merger(module) -> bool:
    return (
        hasattr(module, "norm")
        and hasattr(module, "linear_fc1")
        and hasattr(module, "act_fn")
        and hasattr(module, "linear_fc2")
        and isinstance(module.linear_fc1, nn.Linear)
        and isinstance(module.linear_fc2, nn.Linear)
    )


def _is_minicpmv_resampler(module) -> bool:
    attn = getattr(module, "attn", None)
    return (
        hasattr(module, "ln_q")
        and hasattr(module, "ln_kv")
        and hasattr(module, "ln_post")
        and hasattr(module, "kv_proj")
        and hasattr(module, "proj_fc")
        and attn is not None
        and hasattr(attn, "q_proj")
        and hasattr(attn, "k_proj")
        and hasattr(attn, "v_proj")
        and hasattr(attn, "out_proj")
    )


def _is_minicpmv_timm_vision_block(module) -> bool:
    attn = getattr(module, "attn", None)
    mlp = getattr(module, "mlp", None)
    return (
        hasattr(module, "norm1")
        and hasattr(module, "norm2")
        and attn is not None
        and mlp is not None
        and hasattr(attn, "q_proj")
        and hasattr(attn, "k_proj")
        and hasattr(attn, "v_proj")
        and hasattr(attn, "proj")
        and hasattr(mlp, "fc1")
        and hasattr(mlp, "fc2")
    )


def _expand_channel_scales(scales: torch.Tensor, target_dim: int) -> torch.Tensor:
    scales = scales.view(-1)
    if scales.numel() == target_dim:
        return scales
    if scales.numel() <= 0 or target_dim % scales.numel() != 0:
        raise ValueError(
            f"Cannot expand AWQ scales from {scales.numel()} to target_dim={target_dim}."
        )
    return scales.repeat(target_dim // scales.numel())


@torch.no_grad()
def get_weight_scale(weight, q_group_size=-1):
    org_shape = weight.shape
    if q_group_size > 0:
        weight = weight.view(-1, q_group_size)
    scale = weight.abs() / weight.abs().amax(dim=1, keepdim=True)
    scale = scale.view(org_shape)
    scale = scale.mean(0)
    return scale


@torch.no_grad()
def get_act_scale(x):
    return x.abs().view(-1, x.shape[-1]).mean(0)


@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device).to(ln.weight.dtype)

    if isinstance(ln, Qwen3_5RMSNorm):
        # Qwen3.5 RMSNorm applies (1 + weight) instead of weight directly.
        # AWQ needs to divide the effective gain by `scales`, not the raw parameter.
        effective_weight = (ln.weight.float() + 1.0).div_(scales.float()).sub_(1.0)
        ln.weight.copy_(effective_weight.to(ln.weight.dtype))
    else:
        ln.weight.div_(scales)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc_scales = _expand_channel_scales(scales, fc.weight.shape[1]).to(fc.weight.device).to(fc.weight.dtype)
        fc.weight.mul_(fc_scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)
    # assert fc1.out_features == fc2.in_features

    scales = scales.to(fc1.weight.device).to(fc1.weight.dtype)

    # fc1.weight.div_(scales.view(-1, 1))
    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_gelu_fc(gelu, fc, scales):
    assert isinstance(gelu, (nn.GELU, BloomGelu, GELUActivation, GELUTanh))
    assert isinstance(fc, nn.Linear)

    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device).to(fc.weight.dtype))

    for p in fc.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def auto_scale_block(
    model,
    module,
    module_kwargs,
    w_bit,
    q_config,
    input_feat,
    qwen3_5_quantize_linear_attn=True,
    sample_inputs=None,
    replay_samples_by_name=None,
):
    from .quantizer import pseudo_quantize_tensor

    # firstly, get the weight quantize function
    if w_bit is not None:

        def w_quantize_func(p):
            return pseudo_quantize_tensor(
                p,
                n_bit=w_bit,
                **q_config,
            ).detach()

    else:

        def w_quantize_func(p):
            return p

    module_kwargs = dict(module_kwargs or {})
    if "use_cache" in module_kwargs:
        module_kwargs.pop("use_cache")

    def _move_nested_to_device(value, device):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, tuple):
            return tuple(_move_nested_to_device(item, device) for item in value)
        if isinstance(value, list):
            return [_move_nested_to_device(item, device) for item in value]
        if isinstance(value, dict):
            return {key: _move_nested_to_device(item, device) for key, item in value.items()}
        return value

    def _run_block(block, x, kwargs):
        with torch.no_grad():
            output = block(x, **kwargs)
        if isinstance(output, tuple):
            return output[0]
        return output

    def _run_block_in_chunks(block, x, kwargs):
        try:
            return forward_in_chunks(model, block, x, kwargs)
        except NotImplementedError:
            if kwargs:
                raise
            outputs = []
            for chunk in torch.split(x, 1, dim=0):
                outputs.append(_run_block(block, chunk, {}))
            return torch.cat(outputs, dim=0)

    # find the best scale ratio
    def _search_module_scale(block, linears2scale: list, x, kwargs={}):
        # w: co, ci
        # x: n, ci
        x = x.to(next(block.parameters()).device)
        with torch.no_grad():
            org_out = _run_block_in_chunks(block, x, kwargs)

        x_max = get_act_scale(x)

        best_error = float("inf")
        best_ratio = -1
        best_scales = None

        n_grid = 20
        history = []

        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid
            scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            for fc in linears2scale:
                fc_scales = _expand_channel_scales(scales, fc.weight.shape[1]).to(fc.weight.device)
                fc.weight.mul_(fc_scales.view(1, -1))
                fc.weight.data = w_quantize_func(fc.weight.data) / (
                    fc_scales.view(1, -1).to(fc.weight.device)
                )
            out = _run_block_in_chunks(block, x, kwargs)

            loss = (
                (org_out - out).float().pow(2).mean().item()
            )  # float prevents overflow
            history.append(loss)
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scales = scales
            block.load_state_dict(org_sd)
        if best_ratio == -1:
            print(history)
            raise Exception
        # print(best_ratio)
        best_scales = best_scales.view(-1)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()

    def _search_module_scale_samples(block, linears2scale: list, replay_samples):
        device = next(block.parameters()).device
        flattened_inputs = [
            sample_input.detach().view(-1, sample_input.shape[-1])
            for sample_input, _sample_kwargs in replay_samples
        ]
        x_max = get_act_scale(torch.cat(flattened_inputs, dim=0))

        best_error = float("inf")
        best_ratio = -1
        best_scales = None

        n_grid = 20
        history = []

        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
        org_outputs = []
        for sample_input, sample_kwargs in replay_samples:
            sample_input_device = sample_input.to(device)
            sample_kwargs_device = _move_nested_to_device(sample_kwargs, device)
            org_outputs.append(_run_block(block, sample_input_device, sample_kwargs_device).detach().cpu())

        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid
            scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            for fc in linears2scale:
                fc_scales = _expand_channel_scales(scales, fc.weight.shape[1]).to(fc.weight.device)
                fc.weight.mul_(fc_scales.view(1, -1))
                fc.weight.data = (
                    w_quantize_func(fc.weight.data)
                    / fc_scales.view(1, -1).to(fc.weight.device).to(fc.weight.dtype)
                ).to(fc.weight.dtype)

            loss = 0.0
            for (sample_input, sample_kwargs), org_out in zip(replay_samples, org_outputs):
                sample_input_device = sample_input.to(device)
                sample_kwargs_device = _move_nested_to_device(sample_kwargs, device)
                out = _run_block(block, sample_input_device, sample_kwargs_device)
                loss += (org_out.to(out.device) - out).float().pow(2).mean().item()
            loss /= max(1, len(replay_samples))
            history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scales = scales
            block.load_state_dict(org_sd)

        if best_ratio == -1:
            print(history)
            raise Exception
        best_scales = best_scales.view(-1)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()

    def _auto_get_scale(prev_op, layers, inp, module2inspect=None, kwargs={}, replay_samples=None):
        # module2inspect: if given, we will check the output diff of this module instead of layers
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        if replay_samples is not None:
            scales = _search_module_scale_samples(module2inspect, layers, replay_samples)
        else:
            scales = _search_module_scale(module2inspect, layers, inp, kwargs)
        scales = scales.detach().cpu()
        # prev_op_name, [layer_name], scale
        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, m) for m in layers]),
            scales,
        )

    scales_list = []  # return the searched scales

    if isinstance(module, OPTDecoderLayer):
        # attention input
        scales_list.append(
            _auto_get_scale(
                prev_op=module.self_attn_layer_norm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )
        # attn out
        scales_list.append(
            _auto_get_scale(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.out_proj],
                inp=input_feat["self_attn.out_proj"],
            )
        )
        # fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.final_layer_norm,
                layers=[module.fc1],
                inp=input_feat["fc1"],
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.fc1,
                layers=[module.fc2],
                inp=input_feat["fc2"],
            )
        )

    elif (
        isinstance(module, (LlamaDecoderLayer, Qwen2DecoderLayer, Qwen2_5_VLDecoderLayer))
        or _is_llama_family_decoder(module)
    ) and not isinstance(module, Qwen3_5DecoderLayer):
        # attention input
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )
        # attn out
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                )
            )
        # fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
            )
        )
    elif isinstance(module, Qwen3_5DecoderLayer):
        if getattr(module, "layer_type", None) == "linear_attention" and qwen3_5_quantize_linear_attn:
            token_mixer_input_names = [
                "linear_attn.in_proj_qkv",
                "linear_attn.in_proj_z",
                "linear_attn.in_proj_b",
                "linear_attn.in_proj_a",
            ]
            token_mixer_reference_input = next(
                (name for name in token_mixer_input_names if name in input_feat),
                None,
            )
            token_mixer_input_layers = [
                getattr(module.linear_attn, name.rsplit(".", 1)[-1])
                for name in token_mixer_input_names
                if name in input_feat
            ]
            if token_mixer_input_layers and token_mixer_reference_input is not None:
                scales_list.append(
                    _auto_get_scale(
                        prev_op=module.input_layernorm,
                        layers=token_mixer_input_layers,
                        inp=input_feat[token_mixer_reference_input],
                        module2inspect=module.linear_attn,
                        kwargs=module_kwargs,
                    )
                )
            # linear_attn.out_proj follows a head-shared gated RMSNorm whose
            # parameter dimension is head_v_dim instead of value_dim, so the
            # standard AWQ norm->fc scale transfer does not apply directly.
            # Keep out_proj on direct pseudo quantization only for now.

        # Keep the remaining Qwen3.5 attention projections in higher precision
        # for stability unless explicitly enabled above.
        if "mlp.gate_proj" in input_feat:
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.post_attention_layernorm,
                    layers=[module.mlp.gate_proj, module.mlp.up_proj],
                    inp=input_feat["mlp.gate_proj"],
                    module2inspect=module.mlp,
                )
            )
        if "mlp.down_proj" in input_feat:
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.mlp.up_proj,
                    layers=[module.mlp.down_proj],
                    inp=input_feat["mlp.down_proj"],
                )
            )

    elif _is_qwen2_vl_vision_block(module):
        replay_samples_by_name = replay_samples_by_name or {}
        attn_replay_samples = replay_samples_by_name.get("attn")
        mlp_replay_samples = replay_samples_by_name.get("mlp")
        if mlp_replay_samples is None and sample_inputs is not None:
            mlp_replay_samples = [(sample_input, {}) for sample_input, _sample_kwargs in sample_inputs]
        if "attn.qkv" in input_feat:
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.norm1,
                    layers=[module.attn.qkv],
                    inp=input_feat["attn.qkv"],
                    module2inspect=module.attn,
                    kwargs=module_kwargs,
                    replay_samples=attn_replay_samples if attn_replay_samples is not None else sample_inputs,
                )
            )
        if "mlp.fc1" in input_feat:
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.norm2,
                    layers=[module.mlp.fc1],
                    inp=input_feat["mlp.fc1"],
                    module2inspect=module.mlp,
                    kwargs={},
                    replay_samples=mlp_replay_samples,
                )
            )
        elif "mlp.gate_proj" in input_feat:
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.norm2,
                    layers=[module.mlp.gate_proj, module.mlp.up_proj],
                    inp=input_feat["mlp.gate_proj"],
                    module2inspect=module.mlp,
                    kwargs={},
                    replay_samples=mlp_replay_samples,
                )
            )
        elif "mlp.linear_fc1" in input_feat:
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.norm2,
                    layers=[module.mlp.linear_fc1],
                    inp=input_feat["mlp.linear_fc1"],
                    module2inspect=module.mlp,
                    kwargs={},
                    replay_samples=mlp_replay_samples,
                )
            )
            if "mlp.linear_fc2" in input_feat:
                scales_list.append(
                    _auto_get_scale(
                        prev_op=module.mlp.act_fn,
                        layers=[module.mlp.linear_fc2],
                        inp=input_feat["mlp.linear_fc2"],
                    )
                )

    elif _is_minicpmv_timm_vision_block(module):
        replay_samples_by_name = replay_samples_by_name or {}
        attn_replay_samples = replay_samples_by_name.get("attn")
        mlp_replay_samples = replay_samples_by_name.get("mlp")
        if "attn.q_proj" in input_feat:
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.norm1,
                    layers=[module.attn.q_proj, module.attn.k_proj, module.attn.v_proj],
                    inp=input_feat["attn.q_proj"],
                    module2inspect=module.attn,
                    kwargs=module_kwargs,
                    replay_samples=attn_replay_samples if attn_replay_samples is not None else sample_inputs,
                )
            )
        if "attn.proj" in input_feat:
            attn_proj_prev = getattr(module.attn, "norm", None)
            if _is_norm_like(attn_proj_prev):
                scales_list.append(
                    _auto_get_scale(
                        prev_op=attn_proj_prev,
                        layers=[module.attn.proj],
                        inp=input_feat["attn.proj"],
                        module2inspect=module.attn.proj,
                    )
                )
            elif (
                getattr(module.attn, "v_proj", None) is not None
                and getattr(module.attn.v_proj, "weight", None) is not None
                and getattr(module.attn.proj, "weight", None) is not None
                and module.attn.v_proj.weight.shape == module.attn.proj.weight.shape
            ):
                scales_list.append(
                    _auto_get_scale(
                        prev_op=module.attn.v_proj,
                        layers=[module.attn.proj],
                        inp=input_feat["attn.proj"],
                    )
                )
        if "mlp.fc1" in input_feat:
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.norm2,
                    layers=[module.mlp.fc1],
                    inp=input_feat["mlp.fc1"],
                    module2inspect=module.mlp,
                    kwargs={},
                    replay_samples=mlp_replay_samples,
                )
            )
        if "mlp.fc2" in input_feat:
            mlp_fc2_prev = getattr(module.mlp, "norm", None)
            if not _is_norm_like(mlp_fc2_prev):
                mlp_fc2_prev = getattr(module.mlp, "act", None)
            if _is_norm_like(mlp_fc2_prev) or isinstance(
                mlp_fc2_prev,
                (nn.GELU, BloomGelu, GELUActivation, GELUTanh, nn.SiLU),
            ):
                scales_list.append(
                    _auto_get_scale(
                        prev_op=mlp_fc2_prev,
                        layers=[module.mlp.fc2],
                        inp=input_feat["mlp.fc2"],
                    )
                )

    elif _is_qwen2_vl_patch_merger(module):
        if "mlp.0" in input_feat:
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.ln_q,
                    layers=[module.mlp[0]],
                    inp=input_feat["mlp.0"],
                    module2inspect=module,
                    kwargs=module_kwargs,
                    replay_samples=sample_inputs,
                )
            )
        if "mlp.2" in input_feat:
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.mlp[1],
                    layers=[module.mlp[2]],
                    inp=input_feat["mlp.2"],
                )
            )

    elif _is_qwen3_vl_patch_merger(module):
        replay_samples_by_name = replay_samples_by_name or {}
        norm_replay_samples = replay_samples_by_name.get("norm")
        if "linear_fc1" in input_feat:
            if getattr(module, "use_postshuffle_norm", False):
                scales_list.append(
                    _auto_get_scale(
                        prev_op=module.norm,
                        layers=[module.linear_fc1],
                        inp=input_feat["linear_fc1"],
                        module2inspect=module.linear_fc1,
                        replay_samples=norm_replay_samples,
                    )
                )
            else:
                scales_list.append(
                    _auto_get_scale(
                        prev_op=module.norm,
                        layers=[module.linear_fc1],
                        inp=input_feat["linear_fc1"],
                        module2inspect=module,
                        kwargs=module_kwargs,
                        replay_samples=sample_inputs,
                    )
                )
        if "linear_fc2" in input_feat:
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.act_fn,
                    layers=[module.linear_fc2],
                    inp=input_feat["linear_fc2"],
                )
            )

    elif _is_minicpmv_resampler(module):
        if (
            hasattr(module.attn, "v_proj")
            and hasattr(module.attn, "out_proj")
            and "attn.out_proj" in input_feat
            and getattr(module.attn.v_proj, "weight", None) is not None
            and getattr(module.attn.out_proj, "weight", None) is not None
            and module.attn.v_proj.weight.shape == module.attn.out_proj.weight.shape
        ):
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.attn.v_proj,
                    layers=[module.attn.out_proj],
                    inp=input_feat["attn.out_proj"],
                )
            )
        if "proj_fc" in input_feat:
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.ln_post,
                    layers=[module.proj_fc],
                    inp=input_feat["proj_fc"],
                    module2inspect=module.proj_fc,
                )
            )

    elif isinstance(module, BloomBlock):
        # attention input
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm,
                layers=[module.self_attention.query_key_value],
                inp=input_feat["self_attention.query_key_value"],
                module2inspect=module,
                kwargs=module_kwargs,
            )
        )
        # attn out
        # Please refer to https://github.com/mit-han-lab/llm-awq/issues/2#issuecomment-1606297469
        """
        scales_list.append(_auto_get_scale(
            prev_op=module.self_attention.query_key_value,
            layers=[module.self_attention.dense],
            inp=input_feat['self_attention.dense'],
        ))
        """
        # fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.dense_h_to_4h],
                inp=input_feat["mlp.dense_h_to_4h"],
                module2inspect=module,
                kwargs=module_kwargs,
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.gelu_impl,
                layers=[module.mlp.dense_4h_to_h],
                inp=input_feat["mlp.dense_4h_to_h"],
            )
        )
    elif "mpt" in str(module.__class__).lower():
        # attention input
        scales_list.append(
            _auto_get_scale(
                prev_op=module.norm_1,
                layers=[module.attn.Wqkv],
                inp=input_feat["attn.Wqkv"],
                module2inspect=module.attn,
                kwargs=module_kwargs,
            )
        )

        # attn out
        scales_list.append(
            _auto_get_scale(
                prev_op=module.attn.Wqkv,
                layers=[module.attn.out_proj],
                inp=input_feat["attn.out_proj"],
            )
        )
        # fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.norm_2,
                layers=[module.ffn.up_proj],
                inp=input_feat["ffn.up_proj"],
                module2inspect=module.ffn,
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.ffn.act,
                layers=[module.ffn.down_proj],
                inp=input_feat["ffn.down_proj"],
            )
        )

    elif "falcon" in str(module.__class__).lower():
        # attn out
        # Haotian: TBD: need to handle repeated scales for MQ
        """
        scales_list.append(_auto_get_scale(
            prev_op=module.self_attention.query_key_value,
            layers=[module.self_attention.dense],
            inp=input_feat['self_attention.dense'],
        ))
        """
        # fc1, as long as it is scaled, everything is screwed up
        if "falcon-7b" in str(module.__class__).lower():
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.input_layernorm,
                    layers=[
                        module.mlp.dense_h_to_4h,
                        module.self_attention.query_key_value,
                    ],
                    inp=input_feat["self_attention.query_key_value"],
                    module2inspect=module,
                    kwargs=module_kwargs,
                )
            )
        elif "falcon-40b" in str(module.__class__).lower():
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.ln_attn,
                    layers=[module.self_attention.query_key_value],
                    inp=input_feat["self_attention.query_key_value"],
                    module2inspect=module,
                    kwargs=module_kwargs,
                )
            )
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.ln_mlp,
                    layers=[module.mlp.dense_h_to_4h],
                    inp=input_feat["mlp.dense_h_to_4h"],
                    module2inspect=module,
                    kwargs=module_kwargs,
                )
            )
        else:
            raise NotImplementedError(
                "Unknown Falcon architecture, currently only falcon-7b and falcon-40b are supported"
            )
        # fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.act,
                layers=[module.mlp.dense_4h_to_h],
                inp=input_feat["mlp.dense_4h_to_h"],
            )
        )
    elif "bigcode" in str(module.__class__).lower():
        scales_list.append(
            _auto_get_scale(
                prev_op=module.ln_1,
                layers=[module.attn.c_attn],
                inp=input_feat["attn.c_attn"],
                module2inspect=module.attn,
                kwargs=module_kwargs,
            )
        )
        # fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.ln_2,
                layers=[module.mlp.c_fc],
                inp=input_feat["mlp.c_fc"],
                module2inspect=module.mlp,
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.act,
                layers=[module.mlp.c_proj],
                inp=input_feat["mlp.c_proj"],
            )
        )
    elif "neox" in str(module.__class__).lower():
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm,
                layers=[module.attention.query_key_value],
                inp=input_feat["attention.query_key_value"],
                module2inspect=module.attention,
                kwargs=module_kwargs,
            )
        )
        # fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.dense_h_to_4h],
                inp=input_feat["mlp.dense_h_to_4h"],
                module2inspect=module.mlp,
            )
        )
        # fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.act,
                layers=[module.mlp.dense_4h_to_h],
                inp=input_feat["mlp.dense_4h_to_h"],
            )
        )
    else:
        raise NotImplementedError(f"{type(module)} not supported yet!")

    return scales_list


def apply_scale(module, scales_list, input_feat_dict=None, device=None):
    for prev_op_name, layer_names, scales in scales_list:
        prev_op = get_op_by_name(module, prev_op_name)
        layers = [get_op_by_name(module, name) for name in layer_names]

        # device_map 模式下不手动移动模块权重
        # scales 对齐到实际操作设备：norm/linear 用 weight.device，GELU/SiLU 用第一个 linear 的设备
        if isinstance(prev_op, (nn.GELU, BloomGelu, GELUActivation, nn.SiLU)):
            scales = scales.to(layers[0].weight.device)
        else:
            scales = scales.to(prev_op.weight.device)

        if isinstance(prev_op, nn.Linear):
            assert len(layers) == 1
            scale_fc_fc(prev_op, layers[0], scales)
        elif _is_norm_like(prev_op):
            scale_ln_fcs(prev_op, layers, scales)
        elif isinstance(prev_op, (nn.GELU, BloomGelu, GELUActivation, GELUTanh, nn.SiLU)):
            new_module = ScaledActivation(prev_op, scales)
            set_op_by_name(module, prev_op_name, new_module)
            scale_gelu_fc(prev_op, layers[0], scales)
        else:
            raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")

        # apply the scaling to input feat if given; prepare it for clipping
        if input_feat_dict is not None:
            for layer_name, layer in zip(layer_names, layers):
                inp = input_feat_dict[layer_name]
                target_dim = getattr(layer, "in_features", inp.shape[-1])
                layer_scales = _expand_channel_scales(scales, target_dim)
                inp.div_(layer_scales.view(1, -1).to(inp.device).to(inp.dtype))
