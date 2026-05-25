"""Evaluation harness: clinical-tool-call verification on top of KARMA + interwhen.

Contains the full experimental apparatus — fact extractor, semantic verifier,
interwhen-style monitor, KARMA-compatible model adapters, runner, and analysis.
See TESTING.md for the experimental design this package implements.
"""

# Apply upstream-KARMA patches on import so EKA_API_TOKEN works without a fork.
# See harness/_patches.py for what this does and when to remove it.
from . import _patches as _patches

_patches.apply_patches()
