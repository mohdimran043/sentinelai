# Plan — SentinelAI situational awareness

Turning the existing escalation-gate engine into a multi-camera situational-awareness
platform: per-camera AI capabilities, pluggable behaviour detectors, temporal
fall/collapse detection, optional face-based person authorization, a real alert engine,
and an operator console built for a command centre rather than an admin panel.

## What this plan is not

It is **not** a rebuild. The escalation gate, the ports, the VRAM planner, the admission
gate, the clip pipeline and the publish/spool/dead-letter path all stay exactly as they
are. Every decision in [docs/decisions.md](../../decisions.md) stands unless this plan
names it and says why.

Two of those decisions constrain this work directly and are honoured rather than undone:

* **ADR 7 — the escalation gate is the product.** Behaviour detectors added here run on
  the cheap per-frame signals the gate already computes. None of them calls a VLM. They
  *raise* the escalation reason that the gate then rules on, so the gate stays the one
  place that decides whether GPU gets spent.
* **ADR 10 — welfare concerns are an opinion, not a detection.** Temporal fall detection
  is a second source of judgement, so per that ADR's own closing paragraph it gets its
  own `basis` value (`temporal_pose_vlm`) rather than widening `single_frame_vlm`.
  `Confidence` still has no `certain`, and the wording stays "possible collapse".

## Layering rule for everything below

Temporal behaviour classification is a **pure function of cheap signals over time**.
That is the same bet `domain/policy/escalation.py` already makes, and it is why this
plan puts every state machine in `domain/` and every model in `adapters/`:

```
domain/behaviour/     pure state machines — fall, loiter, abandonment, tamper, crossing
domain/policy/        pure rules          — authorization confirmation, alert aggregation
ports/                new seams           — PoseEstimator, FaceDetector, FaceEmbedder,
                                            FaceIndex, BehaviourDetector
adapters/             the models          — YOLO-pose, SCRFD, ArcFace, encrypted index
```

A fall state machine that reads no clock and holds no pixels can be tested exhaustively
on CPU in milliseconds, which is the only way the "person stayed down for 7 seconds"
case gets covered at all. `tests/test_architecture.py` enforces the purity; new pure
modules are added to its scan automatically because it walks the packages.

---

## Phase A — Per-camera AI capabilities

**Why first.** Everything after this is optional per camera, and §13's "automatically
avoid loading unnecessary models for disabled capabilities" is only possible if the
composition root can ask, before building anything, which models any camera actually
wants.

* `domain/capabilities.py` — `Capability` StrEnum, `CameraCapabilities` frozen set-like
  value, and `required_model_keys(capabilities) -> frozenset[str]`, pure.
* `cameras.json` grows a `capabilities` list; absent means the current behaviour
  (scene description + anomaly detection), so every existing file keeps loading.
* `CameraProfile.vlm_enabled` is **derived** from `SCENE_DESCRIPTION` at load time
  rather than becoming a second source of truth. The gate is untouched.
* `main.build_models` builds only what the union of camera capabilities requires.
* `GET /cameras` reports each camera's capabilities; `PATCH /cameras/{id}` edits them.

**Done when:** a camera with `[]` capabilities loads no VLM at all, and the suite proves
it by composing an engine and asserting the registry has no VLM key.

## Phase B — Behaviour detector framework

* `ports/behaviour.py` — `BehaviourDetector`, synchronous and pure-friendly, mirroring
  `Tracker`'s reasoning (cheap CPU association work, no async).
* `domain/behaviour/` — one module per detector, each a `(state, observation) -> (state,
  candidates)` function. Registry is a tuple, like `ALL_TRIGGERS`.
* `CandidateEvent{kind, confidence, track_ids, detail, first_seen, last_seen}`.
* The runner feeds the same `SceneState` it already builds; detectors add no decode, no
  inference, and no second detector pass.
* A candidate raises an escalation with its own reason, and the gate rules on it exactly
  as it rules on the existing seven.

**Detectors in this phase:** loitering (dwell is already computed), abandoned object,
camera tamper/obstruction, line crossing. Each is pure and separately tested.

## Phase C — Fall / collapse detection

* `ports/pose.py` — `PoseEstimator`, optional, returning keypoints per person track.
* `adapters/pose/yolo11_pose.py` — YOLO11n-pose, loaded only when a camera enables
  `FALL_DETECTION`.
* `domain/behaviour/fall.py` — the state machine:
  `UPRIGHT → DESCENDING → DOWN → SETTLED(candidate)`, driven by centroid vertical
  velocity, bbox aspect inversion and stillness, refined by torso angle when pose is
  available and degrading honestly to geometry when it is not.
* A settled candidate escalates with `reason=fall_suspected`; the VLM confirms and the
  resulting welfare concern carries `basis="temporal_pose_vlm"`.
* Schema: `basis` widens from `const` to an enum of two values, additively.

**Done when:** a synthetic track that stands, drops and stays down produces exactly one
candidate; one that sits down slowly produces none; one that drops and gets straight up
produces none. All on CPU, no model weights.

## Phase D — Person authorization

* `domain/identity.py` — `AuthorizedPerson`, `FaceQuality`, thresholds as configuration.
* `ports/face.py` — `FaceDetector`, `FaceEmbedder`, `FaceIndex`.
* `domain/policy/authorization.py` — **pure** temporal confirmation (§11): N consistent
  observations of one tracked person, each above a quality floor, each below the match
  threshold, spanning a minimum duration, before `unauthorized` is ever concluded.
* Encrypted-at-rest embedding store; reference images optional and deletable (§12).
* Per-camera enablement, per-camera authorized roster, audit trail.
* API: `/authorized-persons`, `/authorized-persons/{id}/images`,
  `/cameras/{id}/authorization`.

## Phase E — Alert engine

* `domain/alert.py` + `domain/policy/alerting.py` — pure aggregation: dedup window,
  occurrence counting, first/last seen, severity, state machine
  (`active → acknowledged → resolved`).
* §17's worked example is the acceptance test: one unauthorized person seen 17 times in
  20 seconds is **one** alert with `occurrences=17`, not 100 alerts.
* VLM queue becomes priority-ordered (§16); drop-on-full drops the *lowest* priority.

## Phase F — Console redesign

Command-centre shell, live camera wall, alert rail, investigation view, people &
authorization, camera capability editor, AI performance page. Built against the real
endpoints above — nothing mocked (§40).

## Phase G — Benchmarks

`sentinel_ai/benchmark/` — a real harness that runs 1→N cameras off looped files and
reports detector/pose/face/VLM latency, FPS, VRAM, GPU and CPU utilisation. Numbers in
the final report come from this and from nowhere else (§33).

---

## Test obligations, every phase

Nothing in `domain/` may read a clock or import an adapter; the fitness test enforces it.
Every pure rule gets exhaustive CPU tests. Every adapter gets a port-contract test. GPU
tests are marked `gpu` and deselected in CI, as now.
