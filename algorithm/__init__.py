"""Algorithm package bootstrap."""

from __future__ import annotations

import os

# InternVL slow tokenizer relies on the Python protobuf backend. This must be
# configured before remote tokenizer modules are imported, otherwise the process
# can silently fall back to an incompatible fast-tokenizer path.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
