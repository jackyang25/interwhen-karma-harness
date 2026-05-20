"""Runtime patches for upstream KARMA.

Applied on import so the framework reads EKA_API_TOKEN without needing a fork.
Delete this file once KARMA upstream supports Bearer auth for MCP clients
natively, then remove the call site in harness/__init__.py.

Why this exists: upstream KARMA's MCP client calls FastMCPClient(server_url)
with no auth argument. The MedAI server requires an Authorization header. This
patch wraps FastMCPClient.__init__ to inject the Bearer token from the env var
when no explicit auth is provided.

The patch is keyed on FastMCPClient's stable public API surface, not KARMA's
internals, so it is robust to KARMA refactors.
"""
import functools
import os


_PATCHED = False


def apply_patches() -> None:
    """Add EKA_API_TOKEN Bearer auth to KARMA's MCP client. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return

    try:
        from fastmcp import Client as _FastMCPClient
    except ImportError:
        # fastmcp not installed — nothing to patch. Caller will get an
        # ImportError elsewhere with a clearer message.
        return

    _original_init = _FastMCPClient.__init__

    @functools.wraps(_original_init)
    def patched_init(self, *args, auth=None, **kwargs):
        if auth is None:
            token = os.environ.get("EKA_API_TOKEN")
            if token:
                auth = token  # FastMCP accepts a raw string as a Bearer token
        return _original_init(self, *args, auth=auth, **kwargs)

    _FastMCPClient.__init__ = patched_init
    _PATCHED = True
