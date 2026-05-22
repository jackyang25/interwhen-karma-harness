"""KARMA-compatible model adapters.

For now this package exposes the Sonnet adapter (apparatus validation) and the
MedAI MCP transport wrapper. The vLLM/Qwen3 adapter is added once the
apparatus-validation gate is passed.

These adapters do not yet subclass KARMA's BaseModel — that integration is
deferred until we know the full surface area (verifier, monitors, etc.) so we
can wrap once instead of refactoring repeatedly. The eval loop in
harness.runner is KARMA-shaped (dataset → model → scorer) and can be lifted
into KARMA's CLI later without rewriting the adapters.
"""
from harness.karma_adapter.sonnet import SonnetAdapter, SonnetResponse

# Qwen3Adapter is imported lazily — it pulls in vllm + transformers which only
# exist on GPU clusters. Plain `from harness.karma_adapter import Qwen3Adapter`
# works on a GPU box; it'll raise ImportError on a CPU/Serverless cluster.
def __getattr__(name: str):
    if name in {"Qwen3Adapter", "Qwen3Response"}:
        from harness.karma_adapter import qwen3
        return getattr(qwen3, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["SonnetAdapter", "SonnetResponse", "Qwen3Adapter", "Qwen3Response"]
