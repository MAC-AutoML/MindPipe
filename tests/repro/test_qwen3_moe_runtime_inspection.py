from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "repro"
    / "qwen3_moe_runtime_inspection.py"
)
SPEC = importlib.util.spec_from_file_location(
    "qwen3_moe_runtime_inspection", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
INSPECTION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = INSPECTION
SPEC.loader.exec_module(INSPECTION)


SMALL_SPEC = INSPECTION.InspectionSpec(
    expected_layers=2,
    global_num_experts=4,
    top_k=2,
    hidden_size=4,
    intermediate_size=2,
    world_size=2,
)


class AscendW8A8DynamicFusedMoEMethod:
    pass


class Wrapper:
    def __init__(self) -> None:
        self.quant_method = AscendW8A8DynamicFusedMoEMethod()


class FakeExperts(nn.Module):
    def __init__(self, rank: int, *, replicated: bool = False) -> None:
        super().__init__()
        local = 4 if replicated else SMALL_SPEC.local_num_experts
        self.quant_method = Wrapper()
        self.local_num_experts = local
        self.global_num_experts = SMALL_SPEC.global_num_experts
        self.ep_size = SMALL_SPEC.world_size
        self.ep_rank = rank
        self.tp_size = 1
        self.tp_rank = 0
        self.global_redundant_expert_num = 0
        if replicated:
            expert_map = torch.arange(4, dtype=torch.int32)
        elif rank == 0:
            expert_map = torch.tensor([0, 1, -1, -1], dtype=torch.int32)
        else:
            expert_map = torch.tensor([-1, -1, 0, 1], dtype=torch.int32)
        self.register_buffer("expert_map", expert_map)
        self.w13_weight = nn.Parameter(
            torch.zeros(local, 4, 4, dtype=torch.int8), requires_grad=False
        )
        self.w2_weight = nn.Parameter(
            torch.zeros(local, 2, 4, dtype=torch.int8), requires_grad=False
        )
        for name, shape in (
            ("w13_weight_scale", (local, 4)),
            ("w13_weight_scale_fp32", (local, 4)),
            ("w13_weight_offset", (local, 4)),
            ("w2_weight_scale", (local, 4)),
            ("w2_weight_scale_fp32", (local, 4)),
            ("w2_weight_offset", (local, 4)),
        ):
            dtype = torch.float32 if name.endswith("fp32") else torch.bfloat16
            tensor = torch.ones(shape, dtype=dtype)
            if name.endswith("offset"):
                tensor.zero_()
            if name.endswith("fp32"):
                setattr(self, name, tensor)
            else:
                self.register_parameter(
                    name, nn.Parameter(tensor, requires_grad=False)
                )


class FakeLayer(nn.Module):
    def __init__(self, rank: int, *, replicated: bool = False) -> None:
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.experts = FakeExperts(rank, replicated=replicated)


class FakeModel(nn.Module):
    def __init__(
        self,
        rank: int,
        *,
        replicated: bool = False,
        with_load_audit: bool = True,
    ) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [
                FakeLayer(rank, replicated=replicated)
                for _ in range(SMALL_SPEC.expected_layers)
            ]
        )
        if with_load_audit:
            records = {}
            local_ids = {0, 1} if rank == 0 else {2, 3}
            for layer in range(SMALL_SPEC.expected_layers):
                for expert in range(SMALL_SPEC.global_num_experts):
                    for projection in ("gate_proj", "up_proj", "down_proj"):
                        for kind in ("weight", "weight_scale", "weight_offset"):
                            source = (
                                f"model.layers.{layer}.mlp.experts.{expert}."
                                f"{projection}.{kind}"
                            )
                            records[source] = {
                                "target_name": "packed",
                                "expert_id": expert,
                                "shard_id": {
                                    "gate_proj": "w1",
                                    "down_proj": "w2",
                                    "up_proj": "w3",
                                }[projection],
                                "status": (
                                    "loaded"
                                    if expert in local_ids
                                    else "skipped_nonlocal"
                                ),
                                "count": 1,
                                "conflicts": [],
                            }
            self._mindpipe_qwen3_expert_load_audit = {
                "schema_version": 1,
                "enabled": True,
                "records": records,
            }


@pytest.fixture(autouse=True)
def clean_replication_env(monkeypatch: pytest.MonkeyPatch):
    for name in INSPECTION.FORBIDDEN_REPLICATION_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(INSPECTION.AUDIT_ENV, "1")


def test_single_storage_ep2_worker_passes() -> None:
    report = INSPECTION.inspect_model(FakeModel(0), SMALL_SPEC)

    assert report["passed"] is True
    assert report["failures"] == []
    assert report["storage_summary"] == {
        "unique_main_weight_storage_count": 4,
        "expected_main_weight_storage_count": 4,
        "main_weight_unique_bytes": 96,
        "expected_main_weight_unique_bytes": 96,
        "quant_metadata_unique_bytes": 256,
        "aliased_main_weights": {},
    }
    assert report["load_audit"]["record_count"] == 72
    assert report["load_audit"]["loaded_count"] == 36
    assert report["load_audit"]["skipped_nonlocal_count"] == 36


def test_replication_and_forbidden_env_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINDPIPE_QWEN3_MOE_REPLICATED_LOCAL_EXPERTS", "1")

    report = INSPECTION.inspect_model(
        FakeModel(0, replicated=True), SMALL_SPEC
    )

    assert report["passed"] is False
    failures = "\n".join(report["failures"])
    assert "local_num_experts=4, expected 2" in failures
    assert "forbidden replication environment" in failures


def test_missing_or_duplicate_load_audit_is_rejected() -> None:
    missing = INSPECTION.inspect_model(
        FakeModel(0, with_load_audit=False), SMALL_SPEC
    )
    assert missing["passed"] is False
    assert any("load audit is absent" in item for item in missing["failures"])

    model = FakeModel(0)
    record = next(
        iter(model._mindpipe_qwen3_expert_load_audit["records"].values())
    )
    record["count"] = 2
    duplicate = INSPECTION.inspect_model(model, SMALL_SPEC)
    assert duplicate["passed"] is False
    assert duplicate["load_audit"]["duplicate_source_count"] == 1
    assert any("load count=2" in item for item in duplicate["failures"])


def test_extra_weight_like_storage_is_rejected() -> None:
    model = FakeModel(0)
    experts = model.model.layers[0].mlp.experts
    experts.register_buffer(
        "w13_weight_reordered", experts.w13_weight.detach().clone()
    )

    report = INSPECTION.inspect_model(model, SMALL_SPEC)

    assert report["passed"] is False
    assert any(
        "extra weight-like tensors detected" in item
        for item in report["failures"]
    )


def test_cross_worker_validator_requires_disjoint_complete_experts() -> None:
    reports = [
        INSPECTION.inspect_model(FakeModel(rank), SMALL_SPEC)
        for rank in range(2)
    ]
    assert INSPECTION.validate_worker_reports(reports, spec=SMALL_SPEC) == []

    overlap = [reports[0], reports[0]]
    failures = INSPECTION.validate_worker_reports(overlap, spec=SMALL_SPEC)
    assert any("EP ranks=[0, 0]" in item for item in failures)
    assert any("share expert ids" in item for item in failures)


def test_worker_validator_fails_closed_on_bad_report() -> None:
    failures = INSPECTION.validate_worker_reports(
        [{"passed": False, "failures": ["bad"]}], spec=SMALL_SPEC
    )
    assert "worker count=1, expected 2" in failures
    assert "worker 0: bad" in failures
