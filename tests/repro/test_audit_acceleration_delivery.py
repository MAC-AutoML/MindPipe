import subprocess
import sys
from pathlib import Path

import pytest

from scripts.repro import audit_acceleration_delivery as audit


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _init_repo(repo: Path) -> None:
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "MindPipe Test")
    _git(repo, "config", "user.email", "mindpipe-test@example.invalid")
    (repo / "README.md").write_text("baseline\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-qm", "baseline")


def _run_main(repo: Path, monkeypatch, base_ref: str = "HEAD") -> int:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_acceleration_delivery.py",
            "--repo-root",
            str(repo),
            "--base-ref",
            base_ref,
        ],
    )
    return audit.main()


def _local_model_assignment() -> str:
    local_home = "/home/" + "ma-user"
    return f"MODEL = '{local_home}/private'\n"


def test_audit_path_accepts_source(tmp_path: Path):
    source = tmp_path / "scripts" / "tool.py"
    source.parent.mkdir(parents=True)
    source.write_text("print('ok')\n", encoding="utf-8")

    size, errors = audit.audit_path(tmp_path, Path("scripts/tool.py"), 1024)

    assert size == len("print('ok')\n")
    assert errors == []


def test_audit_path_rejects_model_and_local_path(tmp_path: Path):
    model = tmp_path / "my_results" / "model.safetensors"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"/home/" + b"ma-user/private-model")

    _, errors = audit.audit_path(
        tmp_path, Path("my_results/model.safetensors"), 1024
    )

    assert any("artifact directory" in error for error in errors)
    assert any("artifact suffix" in error for error in errors)
    assert any("hard-coded local" in error for error in errors)


def test_audit_path_rejects_binary_and_secret(tmp_path: Path):
    payload = tmp_path / "unsafe.txt"
    payload.write_bytes(b"api_" + b"key=" + b"sk-" + b"abcdefghijklmnopqrstuvwxyz\0")

    _, errors = audit.audit_path(tmp_path, Path("unsafe.txt"), 1024)

    assert any("binary content" in error for error in errors)


def test_audit_path_enforces_file_size(tmp_path: Path):
    payload = tmp_path / "large.py"
    payload.write_bytes(b"x" * 17)

    _, errors = audit.audit_path(tmp_path, Path("large.py"), 16)

    assert errors == ["large.py: 17 bytes exceeds per-file limit 16"]


@pytest.mark.parametrize(
    "directory",
    [
        "checkpoints",
        "logs",
        "models",
        "new_results",
        "outputs",
        "results",
        "tokenizer",
        "tokenizers",
    ],
)
def test_audit_path_rejects_delivery_artifact_directories(
    tmp_path: Path, directory: str
):
    relative = Path(directory) / "evidence.json"
    artifact = tmp_path / relative
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n", encoding="utf-8")

    _, errors = audit.audit_path(tmp_path, relative, 1024)

    assert errors == [f"{relative}: forbidden artifact directory"]


@pytest.mark.parametrize("suffix", [".npy", ".npz", ".gguf", ".h5", ".model"])
def test_audit_path_rejects_parameter_artifact_suffixes(
    tmp_path: Path, suffix: str
):
    relative = Path(f"weights{suffix}")
    (tmp_path / relative).write_text("parameter data\n", encoding="utf-8")

    _, errors = audit.audit_path(tmp_path, relative, 1024)

    assert errors == [f"{relative}: forbidden artifact suffix"]


@pytest.mark.parametrize(
    "name",
    [
        "merges.txt",
        "vocab.json",
        "added_tokens.json",
        "chat_template.json",
        "chat_template.jinja",
        "tokenizer.model",
        "spiece.model",
        "sentencepiece.bpe.model",
    ],
)
def test_audit_path_rejects_tokenizer_artifact_names(tmp_path: Path, name: str):
    artifact = tmp_path / name
    artifact.write_text("tokenizer data\n", encoding="utf-8")

    _, errors = audit.audit_path(tmp_path, Path(name), 1024)

    assert f"{name}: forbidden model/tokenizer artifact" in errors
    if name.endswith(".model"):
        assert f"{name}: forbidden artifact suffix" in errors


def test_main_rejects_staged_forbidden_file_deleted_from_worktree(
    tmp_path: Path, monkeypatch, capsys
):
    _init_repo(tmp_path)
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"staged model payload")
    _git(tmp_path, "add", "model.safetensors")
    model.unlink()

    assert _run_main(tmp_path, monkeypatch) == 1
    captured = capsys.readouterr()
    assert "model.safetensors: forbidden artifact suffix" in captured.err


def test_main_audits_index_and_later_worktree_edit(
    tmp_path: Path, monkeypatch, capsys
):
    _init_repo(tmp_path)
    source = tmp_path / "tool.py"
    source.write_text("print('staged and safe')\n", encoding="utf-8")
    _git(tmp_path, "add", "tool.py")
    source.write_text(_local_model_assignment(), encoding="utf-8")

    assert _run_main(tmp_path, monkeypatch) == 1
    captured = capsys.readouterr()
    assert "tool.py: hard-coded local home path" in captured.err


def test_main_audits_unsafe_head_when_worktree_version_is_safe(
    tmp_path: Path, monkeypatch, capsys
):
    _init_repo(tmp_path)
    base_ref = _git(tmp_path, "rev-parse", "HEAD")
    source = tmp_path / "tool.py"
    source.write_text(_local_model_assignment(), encoding="utf-8")
    _git(tmp_path, "add", "tool.py")
    _git(tmp_path, "commit", "-qm", "add unsafe delivery source")
    source.write_text("print('safe worktree version')\n", encoding="utf-8")

    assert _run_main(tmp_path, monkeypatch, base_ref) == 1
    captured = capsys.readouterr()
    assert "tool.py: hard-coded local home path" in captured.err


def test_main_audits_untracked_file_from_worktree(
    tmp_path: Path, monkeypatch, capsys
):
    _init_repo(tmp_path)
    (tmp_path / "leak.py").write_text(_local_model_assignment(), encoding="utf-8")

    assert _run_main(tmp_path, monkeypatch) == 1
    captured = capsys.readouterr()
    assert "leak.py: hard-coded local home path" in captured.err


@pytest.mark.parametrize("source", ["worktree", "index", "head"])
@pytest.mark.parametrize(
    ("relative", "error_fragment"),
    [
        (Path("results/evidence.json"), "forbidden artifact directory"),
        (Path("weights.npy"), "forbidden artifact suffix"),
    ],
)
def test_main_rejects_new_blacklist_from_every_git_source(
    tmp_path: Path,
    monkeypatch,
    capsys,
    source: str,
    relative: Path,
    error_fragment: str,
):
    _init_repo(tmp_path)
    base_ref = _git(tmp_path, "rev-parse", "HEAD")
    artifact = tmp_path / relative
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("delivery artifact\n", encoding="utf-8")

    if source == "index":
        _git(tmp_path, "add", "-f", relative.as_posix())
        artifact.unlink()
    elif source == "head":
        _git(tmp_path, "add", "-f", relative.as_posix())
        _git(tmp_path, "commit", "-qm", "add forbidden delivery artifact")
        artifact.unlink()

    assert _run_main(tmp_path, monkeypatch, base_ref) == 1
    captured = capsys.readouterr()
    assert f"{relative}: {error_fragment}" in captured.err


def test_main_skips_staged_deletion(tmp_path: Path, monkeypatch, capsys):
    _init_repo(tmp_path)
    artifact = tmp_path / "old-model.safetensors"
    artifact.write_bytes(b"old artifact")
    _git(tmp_path, "add", "old-model.safetensors")
    _git(tmp_path, "commit", "-qm", "add old artifact")
    _git(tmp_path, "rm", "-q", "old-model.safetensors")

    assert _run_main(tmp_path, monkeypatch) == 0
    captured = capsys.readouterr()
    assert "delivery_audit=PASS" in captured.out
    assert "total_bytes=0" in captured.out
    assert captured.err == ""
