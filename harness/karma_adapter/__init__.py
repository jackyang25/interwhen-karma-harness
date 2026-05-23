"""KARMA-compatible model adapters.

This package exposes:
- `SonnetAdapter` (apparatus validation) — Anthropic Tools API + MedAI MCP.
- `Qwen3Adapter` + `VLLMServer` — talks to a `vllm serve` subprocess over
  OpenAI-compatible HTTP using the structured tools API.

Both adapters give their respective model its native, fine-tuned tool-use
channel, matching the methodology of EkaCare's published comparison.

These adapters do not yet subclass KARMA's BaseModel — that integration is
deferred until we know the full surface area (verifier, monitors, etc.) so we
can wrap once instead of refactoring repeatedly. The eval loop in
harness.runner is KARMA-shaped (dataset → model → scorer) and can be lifted
into KARMA's CLI later without rewriting the adapters.
"""
from harness.karma_adapter.sonnet import SonnetAdapter, SonnetResponse

# Qwen3Adapter / VLLMServer are imported lazily. The module itself doesn't
# require vllm or transformers (it talks to vllm as a subprocess via HTTP),
# but it does need `openai` and `httpx`, which aren't installed on every
# cluster image. Lazy import lets `from harness.karma_adapter import
# SonnetAdapter` work even on clusters that haven't pip-installed the GPU-side
# deps yet.
def __getattr__(name: str):
    if name in {"Qwen3Adapter", "Qwen3Response", "VLLMServer"}:
        from harness.karma_adapter import qwen3
        return getattr(qwen3, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SonnetAdapter",
    "SonnetResponse",
    "Qwen3Adapter",
    "Qwen3Response",
    "VLLMServer",
]
