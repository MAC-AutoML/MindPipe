# Mindpipe Modify Plan

## Goals

- Rename the working copy from `algorithm-workflow` semantics to `mindpipe`.
- Keep feature parity with the current project.
- Keep a single top-level `main.py` entrypoint.
- Separate algorithm implementation, workflow orchestration, and evaluation logic.
- Preserve per-method `method.py` and vendored `source/` directories.

## Target Structure

```text
mindpipe/
├── main.py
├── README.md
├── algorithm/
│   ├── common/
│   ├── quantization/
│   └── pruning/
├── workflow/
│   ├── schema.py
│   ├── builder.py
│   └── executor.py
├── evaluation/
│   ├── ppl.py
│   ├── lm_eval.py
│   └── runner.py
├── scripts/
└── results/
```

## Steps

1. Move `algorithm_v1/common`, `algorithm_v1/quantization`, and `algorithm_v1/pruning` under `algorithm/`.
2. Split `algorithm_v1/workflow` into `workflow/schema.py`, `workflow/builder.py`, and `workflow/executor.py`.
3. Move PPL and lm-eval code from `algorithm_v1/common/evaluation.py` into `evaluation/`.
4. Collapse task dispatch into the root `main.py`; remove extra `main.py` entry modules.
5. Fix imports, default output paths, and script references from `algorithm-workflow` to `mindpipe`.
6. Run lightweight validation for CLI help and import sanity.
