"""Custom KARMA model adapter.

Subclasses KARMA's BaseModel and registers via @register_model_meta. Wraps a
vLLM-served Qwen3 reasoning model with:
- text-mode MCP tool-calling (tool descriptions in system prompt; tool calls
  parsed from the reasoning trace; results injected back into the stream),
- the interwhen monitor + semantic verifier,
- the fact extractor invoked per-vignette.

KARMA invokes the adapter through its standard model interface. All
verification logic is encapsulated inside; KARMA itself is not modified.
"""
