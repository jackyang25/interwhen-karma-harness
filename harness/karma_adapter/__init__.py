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

__all__ = ["SonnetAdapter", "SonnetResponse"]
