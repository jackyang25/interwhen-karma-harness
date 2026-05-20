"""Experimental condition configurations.

One module per condition (A, B, C, D', E) — each builds the appropriate
adapter configuration (system prompt, tool access, verification wrapper,
post-hoc check) so the run scripts can iterate over conditions cleanly.

Prompts and other locked configuration values are read from /conf, not
hard-coded here. See TESTING.md Section 6 for the condition definitions.
"""
