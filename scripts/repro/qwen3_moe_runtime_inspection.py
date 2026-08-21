"""Worker callbacks and validators for Qwen3 MoE single-storage W8A8."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import os
import re
from typing import Any


EXPECTED_METHOD = "AscendW8A8DynamicFusedMoEMethod"
AUDIT_ENV = "MINDPIPE_QWEN3_MOE_SINGLE_STORAGE_AUDIT"
FORBIDDEN_REPLICATION_ENV = (
    "MINDPIPE_QWEN3_MOE_REPLICATED_LOCAL_EXPERTS",
    "MINDPIPE_QWEN3_MOE_REPLICATED_PARALLEL_HALVES",
    "MINDPIPE_QWEN3_MOE_REPLICATED_SINGLE_PASS",
    "MINDPIPE_QWEN3_MOE_REPLICATED_STAGE_AWARE",
)
MAIN_WEIGHT_NAMES = ("w13_weight", "w2_weight")
QUANT_METADATA_NAMES = (
    "w13_weight_scale",
    "w13_weight_scale_fp32",
    "w13_weight_offset",
    "w2_weight_scale",
    "w2_weight_scale_fp32",
    "w2_weight_offset",
)
_EXPERT_MODULE = re.compile(
    r"(?:^|.*\.)layers\.(?P<layer>\d+)\.mlp\.experts$"
)
_EXPERT_SOURCE = re.compile(
    r"(?:^|.*\.)layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)\."
    r"(?P<kind>weight|weight_scale|weight_offset)$"
)


@dataclass(frozen=True)
class InspectionSpec:
    expected_layers: int = 48
    global_num_experts: int = 128
    top_k: int = 8
    hidden_size: int = 2048
    intermediate_size: int = 768
    world_size: int = 2
    expected_quant_method: str = EXPECTED_METHOD

    @property
    def local_num_experts(self) -> int:
        if self.global_num_experts % self.world_size:
            raise ValueError(
                f"global_num_experts={self.global_num_experts} is not divisible "
                f"by world_size={self.world_size}"
            )
        return self.global_num_experts // self.world_size


DEFAULT_SPEC = InspectionSpec()


def _class_name(value: Any) -> str | None:
    return value.__class__.__name__ if value is not None else None


def _storage_info(tensor: Any) -> dict[str, Any]:
    import torch

    if not isinstance(tensor, torch.Tensor):
        return {"present": False}
    result: dict[str, Any] = {
        "present": True,
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
        "element_size": int(tensor.element_size()),
        "tensor_data_ptr": int(tensor.data_ptr()),
        "storage_offset": int(tensor.storage_offset()),
        "is_contiguous": bool(tensor.is_contiguous()),
    }
    storage = tensor.untyped_storage()
    result.update({
        "base_data_ptr": int(storage.data_ptr()),
        "storage_nbytes": int(storage.nbytes()),
        "logical_nbytes": int(tensor.numel() * tensor.element_size()),
    })
    if tensor.dtype.is_floating_point:
        result["finite"] = bool(torch.isfinite(tensor).all().item())
    return result


def _direct_tensors(module: Any) -> dict[str, dict[str, Any]]:
    import torch

    values: dict[str, tuple[torch.Tensor, set[str]]] = {}

    def add(name: str, tensor: Any, source: str) -> None:
        if not isinstance(tensor, torch.Tensor):
            return
        current = values.get(name)
        if current is None:
            values[name] = (tensor, {source})
        else:
            current[1].add(source)

    for name, tensor in module.named_parameters(recurse=False):
        add(name, tensor, "parameter")
    for name, tensor in module.named_buffers(recurse=False):
        add(name, tensor, "buffer")
    for name, tensor in vars(module).items():
        add(name, tensor, "attribute")

    result: dict[str, dict[str, Any]] = {}
    for name, (tensor, sources) in sorted(values.items()):
        result[name] = {
            **_storage_info(tensor),
            "registrations": sorted(sources),
        }
    return result


def _expected_shapes(spec: InspectionSpec) -> dict[str, list[int]]:
    local = spec.local_num_experts
    return {
        "w13_weight": [local, spec.hidden_size, 2 * spec.intermediate_size],
        "w2_weight": [local, spec.intermediate_size, spec.hidden_size],
        "w13_weight_scale": [local, 2 * spec.intermediate_size],
        "w13_weight_scale_fp32": [local, 2 * spec.intermediate_size],
        "w13_weight_offset": [local, 2 * spec.intermediate_size],
        "w2_weight_scale": [local, spec.hidden_size],
        "w2_weight_scale_fp32": [local, spec.hidden_size],
        "w2_weight_offset": [local, spec.hidden_size],
    }


def _local_expert_ids(module: Any, spec: InspectionSpec) -> tuple[list[int], list[str]]:
    failures: list[str] = []
    expert_map = getattr(module, "expert_map", None)
    if expert_map is None:
        failures.append("expert_map is absent; EP2 sharding is not proven")
        return [], failures
    try:
        values = [int(value) for value in expert_map.detach().cpu().tolist()]
    except Exception as exc:
        failures.append(f"cannot read expert_map: {type(exc).__name__}: {exc}")
        return [], failures
    if len(values) != spec.global_num_experts:
        failures.append(
            f"expert_map length={len(values)}, expected {spec.global_num_experts}"
        )
    local_ids = [index for index, local_id in enumerate(values) if local_id != -1]
    mapped_local_ids = sorted(local_id for local_id in values if local_id != -1)
    if len(local_ids) != spec.local_num_experts:
        failures.append(
            f"expert_map local count={len(local_ids)}, expected {spec.local_num_experts}"
        )
    expected_local = list(range(spec.local_num_experts))
    if mapped_local_ids != expected_local:
        failures.append(
            f"expert_map physical ids={mapped_local_ids}, expected {expected_local}"
        )
    return local_ids, failures


def _inspect_load_audit(
    model: Any,
    local_ids_by_layer: dict[int, list[int]],
    spec: InspectionSpec,
    require_load_audit: bool,
) -> dict[str, Any]:
    states = []
    for module_name, module in model.named_modules():
        state = getattr(module, "_mindpipe_qwen3_expert_load_audit", None)
        if state is not None:
            states.append((module_name, state))
    failures: list[str] = []
    if not states:
        if require_load_audit:
            failures.append(
                f"load audit is absent; start model loading with {AUDIT_ENV}=1"
            )
        return {
            "present": False,
            "enabled_env": os.getenv(AUDIT_ENV),
            "record_count": 0,
            "loaded_count": 0,
            "skipped_nonlocal_count": 0,
            "duplicate_source_count": 0,
            "failures": failures,
        }
    if len(states) != 1:
        failures.append(f"found {len(states)} load audit states, expected exactly one")
    module_name, state = states[0]
    records = state.get("records") if isinstance(state, dict) else None
    if not isinstance(records, dict):
        failures.append("load audit records are missing or not a mapping")
        records = {}

    observed: Counter[tuple[int, int, str, str]] = Counter()
    loaded_count = 0
    skipped_count = 0
    duplicate_count = 0
    unmatched_sources: list[str] = []
    status_mismatches: list[str] = []
    for source_name, record in records.items():
        match = _EXPERT_SOURCE.match(source_name)
        if match is None:
            unmatched_sources.append(source_name)
            continue
        key = (
            int(match.group("layer")),
            int(match.group("expert")),
            match.group("projection"),
            match.group("kind"),
        )
        count = record.get("count") if isinstance(record, dict) else None
        if count != 1:
            duplicate_count += 1
            failures.append(f"{source_name}: load count={count!r}, expected 1")
        observed[key] += int(count) if isinstance(count, int) else 0
        if record.get("conflicts"):
            failures.append(f"{source_name}: conflicting audit targets")
        expected_status = (
            "loaded"
            if key[1] in set(local_ids_by_layer.get(key[0], []))
            else "skipped_nonlocal"
        )
        status = record.get("status")
        if status != expected_status:
            status_mismatches.append(
                f"{source_name}: status={status!r}, expected {expected_status!r}"
            )
        if status == "loaded":
            loaded_count += 1
        elif status == "skipped_nonlocal":
            skipped_count += 1

    expected = {
        (layer, expert, projection, kind)
        for layer in range(spec.expected_layers)
        for expert in range(spec.global_num_experts)
        for projection in ("gate_proj", "up_proj", "down_proj")
        for kind in ("weight", "weight_scale", "weight_offset")
    }
    missing = sorted(expected - set(observed))
    unexpected = sorted(set(observed) - expected)
    if unmatched_sources:
        failures.append(f"unmatched expert source tensors: {unmatched_sources[:8]}")
    failures.extend(status_mismatches[:16])
    if len(status_mismatches) > 16:
        failures.append(
            f"{len(status_mismatches) - 16} additional load status mismatches"
        )
    if missing:
        failures.append(
            f"missing {len(missing)} expert source load records; first={missing[:4]}"
        )
    if unexpected:
        failures.append(
            f"unexpected {len(unexpected)} expert source load records; first={unexpected[:4]}"
        )
    expected_loaded = (
        spec.expected_layers * spec.local_num_experts * 3 * 3
    )
    expected_skipped = (
        spec.expected_layers
        * (spec.global_num_experts - spec.local_num_experts)
        * 3
        * 3
    )
    if loaded_count != expected_loaded:
        failures.append(
            f"loaded expert source count={loaded_count}, expected {expected_loaded}"
        )
    if skipped_count != expected_skipped:
        failures.append(
            f"skipped nonlocal source count={skipped_count}, expected {expected_skipped}"
        )
    return {
        "present": True,
        "state_module": module_name,
        "enabled_env": os.getenv(AUDIT_ENV),
        "record_count": len(records),
        "loaded_count": loaded_count,
        "skipped_nonlocal_count": skipped_count,
        "duplicate_source_count": duplicate_count,
        "expected_record_count": len(expected),
        "expected_loaded_count": expected_loaded,
        "expected_skipped_nonlocal_count": expected_skipped,
        "failures": failures,
    }


def _memory_snapshot() -> dict[str, Any]:
    try:
        import torch_npu

        device = torch_npu.npu.current_device()
        return {
            "available": True,
            "device": int(device),
            "memory_allocated": int(torch_npu.npu.memory_allocated(device)),
            "memory_reserved": int(torch_npu.npu.memory_reserved(device)),
            "max_memory_allocated": int(
                torch_npu.npu.max_memory_allocated(device)
            ),
            "max_memory_reserved": int(torch_npu.npu.max_memory_reserved(device)),
        }
    except Exception as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def inspect_model(
    model: Any,
    spec: InspectionSpec = DEFAULT_SPEC,
    *,
    require_load_audit: bool = True,
) -> dict[str, Any]:
    """Inspect one fully loaded worker without executing inference."""
    expected_shapes = _expected_shapes(spec)
    layers: dict[str, Any] = {}
    failures: list[str] = []
    layer_indices: list[int] = []
    local_ids_by_layer: dict[int, list[int]] = {}
    canonical_weight_storages: dict[tuple[str, int, int], list[str]] = {}
    main_weight_unique_bytes = 0
    quant_metadata_storage_bytes: dict[tuple[str, int, int], int] = {}

    for name, module in model.named_modules():
        match = _EXPERT_MODULE.match(name)
        if match is None:
            continue
        layer_index = int(match.group("layer"))
        layer_indices.append(layer_index)
        layer_failures: list[str] = []
        wrapper_method = getattr(module, "quant_method", None)
        inner_method = getattr(wrapper_method, "quant_method", wrapper_method)
        method_name = _class_name(inner_method)
        if method_name != spec.expected_quant_method:
            layer_failures.append(
                f"quant_method={method_name!r}, expected {spec.expected_quant_method!r}"
            )

        local_num_experts = int(getattr(module, "local_num_experts", -1))
        if local_num_experts != spec.local_num_experts:
            layer_failures.append(
                f"local_num_experts={local_num_experts}, expected {spec.local_num_experts}"
            )
        if int(getattr(module, "global_num_experts", -1)) != spec.global_num_experts:
            layer_failures.append(
                f"global_num_experts={getattr(module, 'global_num_experts', None)!r}, "
                f"expected {spec.global_num_experts}"
            )
        if int(getattr(module, "ep_size", -1)) != spec.world_size:
            layer_failures.append(
                f"ep_size={getattr(module, 'ep_size', None)!r}, expected {spec.world_size}"
            )
        if int(getattr(module, "tp_size", -1)) != 1:
            layer_failures.append(
                f"internal tp_size={getattr(module, 'tp_size', None)!r}, expected 1 under EP2"
            )
        redundant = int(getattr(module, "global_redundant_expert_num", 0))
        if redundant != 0:
            layer_failures.append(
                f"global_redundant_expert_num={redundant}, expected 0"
            )

        local_ids, map_failures = _local_expert_ids(module, spec)
        local_ids_by_layer[layer_index] = local_ids
        layer_failures.extend(map_failures)
        tensors = _direct_tensors(module)
        canonical_infos = {
            tensor_name: _storage_info(getattr(module, tensor_name, None))
            for tensor_name in (*MAIN_WEIGHT_NAMES, *QUANT_METADATA_NAMES)
        }
        for tensor_name, expected_shape in expected_shapes.items():
            info = canonical_infos[tensor_name]
            if not info["present"]:
                layer_failures.append(f"missing {tensor_name}")
                continue
            if info["shape"] != expected_shape:
                layer_failures.append(
                    f"{tensor_name} shape={info['shape']}, expected {expected_shape}"
                )
            if tensor_name in MAIN_WEIGHT_NAMES and info["dtype"] != "torch.int8":
                layer_failures.append(
                    f"{tensor_name} dtype={info['dtype']!r}, expected 'torch.int8'"
                )
            if tensor_name in QUANT_METADATA_NAMES and not info.get("finite", False):
                layer_failures.append(f"{tensor_name} is not finite")

        for tensor_name in MAIN_WEIGHT_NAMES:
            info = canonical_infos[tensor_name]
            if not info["present"]:
                continue
            key = (info["device"], info["base_data_ptr"], info["storage_nbytes"])
            canonical_weight_storages.setdefault(key, []).append(
                f"{name}.{tensor_name}"
            )
        for tensor_name in QUANT_METADATA_NAMES:
            info = canonical_infos[tensor_name]
            if not info["present"]:
                continue
            key = (info["device"], info["base_data_ptr"], info["storage_nbytes"])
            quant_metadata_storage_bytes[key] = info["storage_nbytes"]

        canonical_names = set(MAIN_WEIGHT_NAMES) | set(QUANT_METADATA_NAMES)
        extra_weight_like = []
        minimum_main_numel = min(
            spec.local_num_experts * spec.hidden_size * 2 * spec.intermediate_size,
            spec.local_num_experts * spec.intermediate_size * spec.hidden_size,
        )
        for tensor_name, info in tensors.items():
            if tensor_name in canonical_names:
                continue
            if (
                info["dtype"] == "torch.int8"
                and len(info["shape"]) >= 3
                and info["numel"] >= minimum_main_numel
            ):
                extra_weight_like.append(tensor_name)
        if extra_weight_like:
            layer_failures.append(
                f"extra weight-like tensors detected: {extra_weight_like}"
            )

        entry = {
            "layer_index": layer_index,
            "module_class": _class_name(module),
            "quant_method_wrapper": _class_name(wrapper_method),
            "quant_method": method_name,
            "parallel": {
                "local_num_experts": local_num_experts,
                "global_num_experts": int(
                    getattr(module, "global_num_experts", -1)
                ),
                "ep_size": int(getattr(module, "ep_size", -1)),
                "ep_rank": int(getattr(module, "ep_rank", -1)),
                "tp_size": int(getattr(module, "tp_size", -1)),
                "tp_rank": int(getattr(module, "tp_rank", -1)),
                "global_redundant_expert_num": redundant,
                "local_global_expert_ids": local_ids,
            },
            "canonical_tensors": canonical_infos,
            "direct_tensors": tensors,
            "extra_weight_like_tensors": extra_weight_like,
            "passed": not layer_failures,
            "failures": layer_failures,
        }
        layers[name] = entry
        failures.extend(f"{name}: {failure}" for failure in layer_failures)

    expected_indices = set(range(spec.expected_layers))
    actual_indices = set(layer_indices)
    if len(layers) != spec.expected_layers:
        failures.append(
            f"model: MoE layer count={len(layers)}, expected {spec.expected_layers}"
        )
    if len(layer_indices) != len(actual_indices):
        failures.append("model: duplicate MoE layer indices detected")
    missing_indices = sorted(expected_indices - actual_indices)
    unexpected_indices = sorted(actual_indices - expected_indices)
    if missing_indices:
        failures.append(f"model: missing layer indices {missing_indices}")
    if unexpected_indices:
        failures.append(f"model: unexpected layer indices {unexpected_indices}")

    aliased_main_weights = {
        f"{key[0]}:{key[1]}:{key[2]}": names
        for key, names in canonical_weight_storages.items()
        if len(names) != 1
    }
    if aliased_main_weights:
        failures.append(
            f"model: canonical main weights share base storage: {aliased_main_weights}"
        )
    expected_storage_count = spec.expected_layers * len(MAIN_WEIGHT_NAMES)
    if len(canonical_weight_storages) != expected_storage_count:
        failures.append(
            f"model: unique main weight storage count={len(canonical_weight_storages)}, "
            f"expected {expected_storage_count}"
        )
    main_weight_unique_bytes = sum(key[2] for key in canonical_weight_storages)
    expected_main_bytes = (
        spec.expected_layers
        * spec.local_num_experts
        * (
            2 * spec.intermediate_size * spec.hidden_size
            + spec.hidden_size * spec.intermediate_size
        )
    )
    if main_weight_unique_bytes != expected_main_bytes:
        failures.append(
            f"model: main weight unique bytes={main_weight_unique_bytes}, "
            f"expected {expected_main_bytes}"
        )

    forbidden_env = {
        name: os.getenv(name)
        for name in FORBIDDEN_REPLICATION_ENV
        if os.getenv(name) not in (None, "", "0")
    }
    if forbidden_env:
        failures.append(f"model: forbidden replication environment: {forbidden_env}")
    load_audit = _inspect_load_audit(
        model,
        local_ids_by_layer,
        spec,
        require_load_audit=require_load_audit,
    )
    failures.extend(f"load_audit: {failure}" for failure in load_audit["failures"])

    ep_ranks = {
        entry["parallel"]["ep_rank"]
        for entry in layers.values()
        if entry["parallel"]["ep_rank"] >= 0
    }
    if len(ep_ranks) != 1:
        failures.append(f"model: worker exposes EP ranks {sorted(ep_ranks)}")
    ordered_layers = dict(
        sorted(layers.items(), key=lambda item: item[1]["layer_index"])
    )
    return {
        "schema_version": 2,
        "model_class": _class_name(model),
        "spec": asdict(spec),
        "moe_layer_count": len(layers),
        "ep_rank": next(iter(ep_ranks)) if len(ep_ranks) == 1 else None,
        "local_global_expert_ids": (
            local_ids_by_layer.get(0, []) if local_ids_by_layer else []
        ),
        "layers": ordered_layers,
        "storage_summary": {
            "unique_main_weight_storage_count": len(canonical_weight_storages),
            "expected_main_weight_storage_count": expected_storage_count,
            "main_weight_unique_bytes": main_weight_unique_bytes,
            "expected_main_weight_unique_bytes": expected_main_bytes,
            "quant_metadata_unique_bytes": sum(
                quant_metadata_storage_bytes.values()
            ),
            "aliased_main_weights": aliased_main_weights,
        },
        "load_audit": load_audit,
        "forbidden_replication_env": forbidden_env,
        "memory": _memory_snapshot(),
        "passed": not failures,
        "failures": failures,
    }


def validate_worker_reports(
    reports: Any,
    *,
    spec: InspectionSpec = DEFAULT_SPEC,
) -> list[str]:
    worker_reports = reports if isinstance(reports, list) else [reports]
    failures: list[str] = []
    if len(worker_reports) != spec.world_size:
        failures.append(
            f"worker count={len(worker_reports)}, expected {spec.world_size}"
        )
    ranks: list[int] = []
    expert_sets: list[set[int]] = []
    for index, report in enumerate(worker_reports):
        if not isinstance(report, dict):
            failures.append(
                f"worker {index} returned {type(report).__name__}, expected dict"
            )
            continue
        if not report.get("passed", False):
            for failure in report.get("failures", ["worker report did not pass"]):
                failures.append(f"worker {index}: {failure}")
        rank = report.get("ep_rank")
        if not isinstance(rank, int):
            failures.append(f"worker {index}: invalid ep_rank={rank!r}")
        else:
            ranks.append(rank)
        ids = report.get("local_global_expert_ids")
        if not isinstance(ids, list) or any(type(value) is not int for value in ids):
            failures.append(f"worker {index}: invalid local expert ids={ids!r}")
        else:
            expert_sets.append(set(ids))
            if len(ids) != spec.local_num_experts:
                failures.append(
                    f"worker {index}: local expert count={len(ids)}, "
                    f"expected {spec.local_num_experts}"
                )
    if sorted(ranks) != list(range(spec.world_size)):
        failures.append(
            f"EP ranks={sorted(ranks)}, expected {list(range(spec.world_size))}"
        )
    if len(expert_sets) == spec.world_size:
        for left in range(len(expert_sets)):
            for right in range(left + 1, len(expert_sets)):
                overlap = sorted(expert_sets[left] & expert_sets[right])
                if overlap:
                    failures.append(
                        f"workers {left}/{right} share expert ids {overlap}"
                    )
        union = set().union(*expert_sets)
        expected = set(range(spec.global_num_experts))
        if union != expected:
            failures.append(
                f"worker expert union missing={sorted(expected - union)}, "
                f"unexpected={sorted(union - expected)}"
            )
    return failures
