# Resident Welfare Monitoring

**Date:** 2026-08-02
**Status:** Approved for planning
**Predecessor:** [Camera record editing](2026-08-02-camera-edit-console-design.md)

## 1. Goal

Turn a perimeter-security pipeline into one whose subject is a person's wellbeing: notice that
someone has fainted, is being harmed, is fighting, or has taken something they should not have;
record what the model saw in a form software can route on; and get a notification with a short
clip to a human who is not looking at the console.

**Done means:** a collapse on a watched camera produces an event carrying machine-readable
welfare concern, and a notification reaches a configured channel with a clip, within a bounded and
stated number of seconds — on a deployment where nothing about that is silently disabled.

### 1.1 What the user asked for, verbatim

> "alert must also include person fainting, person breaking, person fighting anything that might
> harm him this system is for his welbeing any unusual activity that are available in the person
> must be available. let say if he took some unknown medicine it must documented by llm also a
> notifcation must be sent to end user via some signal or any other system we can decide that
> later"

> "whenever we get a event 3 seconds clip needs to be sent to end user also we dont have to save
> full video once we process video and generated text via llm it must send notification along with
> 3 seccond clip"

> "This system is just a brain and eye. brain has the llms and eye is the web. and it save the
> notification clip and allows them to configure duration and for each camera what kind of
> notification is needed."

## 2. What already exists — measured, not assumed

Building any of this without knowing the following would produce duplicate mechanisms.

| Capability | State today | Where |
|---|---|---|
| Collapse / altercation / distress in the VLM prompt | **Already asked for**, weights `threat_value` up | `adapters/vision/qwen25vl.py` |
| A forced look independent of motion | **Already exists** — `periodic_summary` fires every `summary_interval_seconds` | `domain/policy/triggers.py:117` |
| Default forced-look interval | **45 s** | `domain/camera_profile.py:39` |
| Admission control | token bucket capacity 2, refill 10 s, cooldown 5 s | `domain/camera_profile.py:41-44` |
| Full-video recording | **Does not exist.** Bounded in-memory pre-roll; a clip is written only when an event fires | `adapters/storage/minio_clips.py` |
| Clip length | 3 s pre-roll + 5 s post-roll ≈ 8 s, global, env-only | `config.py:96-97` |
| Outbound notification to a person | **Does not exist at all.** Publishers are RabbitMQ, in-memory, dead-letter | `adapters/publishers/` |
| Per-camera config surface | `label`, `zone` only, via `PATCH /cameras/{id}` | `adapters/config/camera_file.py` |

### 2.1 A correction that changes the design

An earlier reading of `qwen25vl.py`'s docstring — "a fall that begins and ends between two
escalations is never looked at by anything" — is true of the *motion-triggered* path only.
`periodic_summary` guarantees a look every `summary_interval_seconds` regardless of motion.

So the worst case for a person who collapses and then lies still is **bounded at ~45 s**, not
unbounded. Continuous watching is therefore mostly a **tuning and admission-control** problem, not
a new subsystem. This design does not add a second sweep mechanism; it makes the existing one
per-camera configurable and stops welfare escalations being dropped by a bucket sized for
perimeter security.

## 3. Inherited constraints

- **The event carries no model identity** (spec §3.3). Nothing added here may leak it.
- **No authentication exists.** Any write surface added here ships gated the same way
  `SENTINEL_ENABLE_CAMERA_WRITES` is, and notification targets are configuration, not API input.
- **A notification channel is undecided by the user** ("we can decide that later"). The design
  must therefore be channel-agnostic: a port plus adapters, never a hardcoded provider.
- **8 GB card.** Anything that raises inference frequency must be costed and must not starve real
  escalations.
- `domain/` and `ports/` stay pure — no I/O, no clock reads. Enforced by
  `tests/test_architecture.py`.
- **No structured field may read as a detector output.** `qwen25vl.py` argues at length against a
  `fall_detected: true` field precisely because a Phase 1C consumer would read it as a trained
  classifier's verdict. §5 honours that while still delivering routability.

## 4. Stated assumptions

Proceeding under these rather than blocking. Each is cheap to reverse.

1. **"3 second clip" means the notification clip's length is configurable, defaulting to today's
   3 s + 5 s.** The user's own product model says the system "allows them to configure duration",
   which supersedes guessing a fixed 3 s. Shortening the post-roll shortens the alert delay and
   costs the evidence that answers "did they get up?" — that is an operator's tradeoff, so it is
   exposed rather than decided here.
2. **The first shipped channel is a generic webhook.** It is the only choice that reaches Signal
   (via signal-cli-rest), ntfy, Slack, Home Assistant and a custom endpoint without committing to
   any of them. Naming a provider now would be deciding what the user deferred.
3. **The clip travels as a URL, not bytes.** MinIO has no presigning in this codebase today, so
   the notification carries `clip_uri` and the webhook body says plainly that the URL requires
   credentials the recipient may not have. Attaching bytes is a follow-up once a channel is chosen.
4. **"eye is the web"** is read as: the console is how a human sees. No capture-side change is
   implied. Flagged in the ledger for confirmation.

## 5. Welfare concern — the structured state

A new optional `welfare` object on the event, and on `SceneDescription`.

```
welfare:
  concerns: [ { kind, confidence, evidence } ]
  basis: "single_frame_vlm"
```

`kind` ∈ `collapse` | `altercation` | `self_harm` | `medication` | `distress` | `other`.
`confidence` ∈ `possible` | `likely` — **deliberately not a float and never `certain`**.
`evidence` is the model's own words about what in the frame prompted it.

Three deliberate choices, all answering the objection in `qwen25vl.py`:

- **No booleans.** `fall_detected: true` is a claim; `{kind: collapse, confidence: possible}` is an
  opinion with its uncertainty attached, which is what a single-frame VLM can honestly produce.
- **`basis` is mandatory and constant.** Every consumer reads, in the payload itself, that this
  came from one still frame — not from a pose model, not from continuous observation.
- **No new `EscalationReason`.** The seven stay. A reason named `fall_detected` would imply a
  detector that does not exist; welfare concern is an *attribute of a description*, not a trigger.

**Nothing routes on `confidence: possible` alone.** A notification requires `likely`, or `possible`
plus a threat score already in the caution band — otherwise a system meant to protect someone
becomes the system that cried wolf and got muted, which is worse than no alarm.

## 6. Medication and ingestion

The prompt gains an explicit question about apparent ingestion — taking a pill, drinking from an
unlabelled container, handling medication — reported as `kind: medication`. Two constraints:

- **It is documentation, not clinical judgement.** The prompt must not ask the model to name a
  substance, assess a dose, or judge whether it was prescribed. It reports that an apparent
  ingestion event was seen and describes what it saw.
- **`evidence` is the record.** For medication specifically the operator's question is "what did
  it actually see", so the free-text evidence matters more than the kind.

## 7. Notification delivery

A new port, `ports/notifier.py`:

```
class Notifier(ABC):
    async def notify(self, note: WelfareNote) -> None: ...
```

`WelfareNote` carries: `camera_id`, `label`, `zone`, `occurred_at`, `severity`, `description`,
`concerns`, `clip_uri`, `event_id`.

Adapters:
- `adapters/notifiers/webhook.py` — POSTs JSON. Timeout-bounded, retried with backoff, and on
  final failure spooled to the existing dead-letter directory rather than dropped.
- `adapters/notifiers/logging.py` — writes the note to the log. The default, so a deployment that
  configures nothing still has an observable trail and tests need no network.

**Delivery is best-effort and must never block or fail the pipeline.** A webhook that hangs must
not stall the camera runner or delay a clip. Notification dispatch happens after the event is
published, off the critical path, with its own timeout.

**Ordering, from the user's own words** ("once we process video and generated text via llm it must
send notification along with 3 second clip"): process → describe → clip finalised → notify. The
clip is only complete one post-roll after the trigger, so notification latency is bounded below by
`clip_postroll_seconds`. That is the real cost of a longer post-roll and belongs in the docs.

## 8. Per-camera configuration

Extends `cameras.json`, `CameraConfig`, `PATCH /cameras/{id}` and the console's Camera record
panel — the surface built in the previous design, which is exactly where the user said this
belongs ("for each camera what kind of notification is needed").

| Field | Meaning | Runtime-editable? |
|---|---|---|
| `notify_on` | Which concern kinds notify for this camera; `[]` means never | **Yes** |
| `notify_min_confidence` | `likely` (default) or `possible` | **Yes** |
| `clip_preroll_seconds` / `clip_postroll_seconds` | Per-camera override of the global default | **Yes** |
| `summary_interval_seconds` | The forced-look floor (§2.1) | **Yes** |

All four are metadata or policy read at escalation time, not structures the pipeline is part-way
through applying — unlike `profile`, which is why that one still requires a restart.

Same rules as the existing editable fields: omitted means "leave alone", the whole document is
re-validated before writing, and the write is atomic.

## 9. Admission control for welfare

A welfare-relevant escalation must not be dropped because a bucket sized for perimeter security
was empty. `escalations_dropped` already counts drops; today a drop is invisible to an operator.

- A **separate small reserve** of admission slots that only `periodic_summary` and welfare-relevant
  escalations may draw on, so a busy scene cannot starve the forced look.
- `escalations_dropped` is surfaced per camera in the console with an alarm state — it already is
  a `Reading` with `alarm={value > 0}` on the camera page; the dashboard should show it too.

## 10. Out of scope

- Pose estimation, action recognition, and the Phase 4 anomaly detectors. This design explicitly
  does **not** claim to detect falls; it claims to ask a VLM and to record its opinion honestly.
- Authentication. Phase 1C.
- Attaching clip bytes to notifications (assumption 3).
- Any change to the recorder appliance, a separate product this repo only reads.

## 11. Risks

- **The false-alarm budget is the whole product.** §5's `likely` gate is the current answer and is
  a guess. It needs tuning against real footage before anyone relies on it.
- **A stretcher carry was missed entirely** in prior measurement. This system will miss things.
  Documentation must say so where an operator will read it, not only in a source docstring.
- **Raising the forced-look rate costs VRAM and GPU time.** Per-camera `summary_interval_seconds`
  makes it easy to set a value that starves other cameras; the admission reserve in §9 bounds the
  damage but does not remove it.
