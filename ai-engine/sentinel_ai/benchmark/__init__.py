"""Measurement (spec §32, §33).

Not part of the engine. Nothing in `sentinel_ai/` imports this package, and it imports
the engine the way an operator would — through `main.compose` — so what it measures is
the real pipeline rather than a reconstruction of it that could drift from one.

The rule this package exists to serve: **do not claim performance without measuring
it.** Every number in the project's documentation about latency, throughput, VRAM or
camera count comes from here.
"""
