"""ShortGPT structured pruning: layer-wise pruning via weight zeroing or layer removal."""

from __future__ import annotations

import logging
from pathlib import Path

import torch.nn as nn

from ...base import BasePruningMethod
from ....common.datasets import get_calibration_and_evaluation_data
from ....common.modeling import find_linear_layers
from ....common.modeling import get_text_backbone
from ....common.runtime import prepend_python_path

logger = logging.getLogger(__name__)


def _update_layer_indices(module: nn.Module, new_idx: int):
    """递归更新模块树中所有名为 layer_idx 的属性。"""
    if hasattr(module, 'layer_idx'):
        try:
            module.layer_idx = new_idx
        except (AttributeError, TypeError):
            pass  # property 没有 setter 或只读属性
    for child in module.children():
        _update_layer_indices(child, new_idx)


class ShortGPTMethod(BasePruningMethod):
    name = "shortgpt"
    npu_ready = True
    default_calibration_dataset = "pg19"

    def apply_pruning(self, model, tokenizer_bundle, args) -> dict:
        tokenizer = tokenizer_bundle.tokenizer
        device = args.device
        sparsity_ratio = args.sparsity_ratio

        # 1. 加载校准数据
        calibration_batches, _ = get_calibration_and_evaluation_data(
            tokenizer=tokenizer,
            dataset_name=args.calibration_dataset,
            sequence_length=args.sequence_length,
            sample_count=args.calibration_samples,
            seed=args.seed,
            data_path=args.data_path,
        )
        logger.info("ShortGPT: loaded %d calibration batches from %s",
                     len(calibration_batches), args.calibration_dataset)

        # 2. 计算层重要性
        model.eval()

        source_root = Path(__file__).resolve().parent / "source"
        with prepend_python_path(source_root):
            from lib.shortgpt import compute_layer_importances

            importances = compute_layer_importances(
                model=model,
                calibration_batches=calibration_batches,
                device=device,
            )

        n_layers = len(importances)
        logger.info("ShortGPT: layer importances = %s",
                     [f"{v:.4f}" for v in importances])

        # 3. 确定要剪枝的层（重要性最低的层）
        n_prune = int(n_layers * sparsity_ratio)
        ranked = sorted(range(n_layers), key=lambda i: importances[i])
        layers_to_prune = sorted(ranked[:n_prune])
        logger.info("ShortGPT: pruning %d/%d layers (sparsity_ratio=%.2f): %s",
                     n_prune, n_layers, sparsity_ratio, layers_to_prune)

        backbone = get_text_backbone(model)

        if not args.pseudo_pruning:
            # ---- 真剪枝：物理删除层 ----
            # 记录原始参数量
            original_params = sum(p.numel() for p in model.parameters())

            # 构建保留的层列表
            kept_indices = sorted(set(range(n_layers)) - set(layers_to_prune))
            kept_layers = [backbone.layers[i] for i in kept_indices]

            # 替换 ModuleList
            backbone.root.layers = nn.ModuleList(kept_layers)

            # 更新模型配置
            backbone.decoder_config.num_hidden_layers = len(kept_layers)

            # 递归更新层内所有 layer_idx 属性（RoPE 等位置编码依赖此值）
            for new_idx, layer in enumerate(backbone.layers):
                _update_layer_indices(layer, new_idx)

            # 计算剪枝后参数缩减率
            pruned_params = sum(p.numel() for p in model.parameters())
            param_reduction = 1.0 - (pruned_params / max(original_params, 1))

            logger.info("ShortGPT: removed %d layers, remaining %d layers, "
                        "param reduction=%.2f%%",
                        n_prune, len(kept_layers), param_reduction * 100)

            return {
                "source_root": str(source_root),
                "target_sparsity_ratio": sparsity_ratio,
                "observed_sparsity_ratio": param_reduction,
                "structure_pattern": "layer-removal",
                "n_prune_layers": n_prune,
                "total_layers": n_layers,
                "remaining_layers": len(kept_layers),
                "pruned_layer_indices": layers_to_prune,
                "layer_importances": importances,
                "pseudo_pruning": False,
            }

        # ---- 伪剪枝：将被剪枝层的权重置零 ----
        zeroed_params = 0
        for layer_idx in layers_to_prune:
            block = backbone.layers[layer_idx]
            for linear in find_linear_layers(block).values():
                linear.weight.data.zero_()
                zeroed_params += linear.weight.data.numel()
                if linear.bias is not None:
                    linear.bias.data.zero_()
                    zeroed_params += linear.bias.data.numel()

        logger.info("ShortGPT: zeroed %d parameters across %d layers",
                     zeroed_params, len(layers_to_prune))

        # 计算稀疏度
        total_params = 0
        zero_params = 0
        for block in backbone.layers:
            for linear in find_linear_layers(block).values():
                w = linear.weight.data
                zero_params += int((w == 0).sum().item())
                total_params += w.numel()
        observed_sparsity = zero_params / max(total_params, 1)

        return {
            "source_root": str(source_root),
            "target_sparsity_ratio": sparsity_ratio,
            "observed_sparsity_ratio": observed_sparsity,
            "structure_pattern": "layer-removal",
            "n_prune_layers": n_prune,
            "total_layers": n_layers,
            "pruned_layer_indices": layers_to_prune,
            "layer_importances": importances,
            "pseudo_pruning": True,
        }
