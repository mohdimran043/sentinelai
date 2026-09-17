"""Pure temporal behaviour detectors (spec §6, §7).

Each module here is a state machine over the cheap per-frame signals the pipeline
already computes, expressed as `(state, observation) -> (state, candidates)` with no
clock read and no I/O — the same bet `domain/policy/escalation.py` makes, for the same
payoff: "the person stayed down for seven seconds" becomes a test that runs in
microseconds on a CPU, instead of a test nobody writes.

A detector here never calls a vision-language model. It raises a *candidate*, which
the escalation gate then rules on exactly as it rules on the six existing triggers, so
the gate stays the one place that decides whether GPU is spent (ADR 7).
"""
