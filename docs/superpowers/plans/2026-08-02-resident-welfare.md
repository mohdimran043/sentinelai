# Resident Welfare Monitoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Record what the vision model sees about a person's wellbeing in a form software can route on, and get a notification with a clip to a human who is not watching the console.

**Architecture:** A pure `WelfareAssessment` in `domain/`, produced by the VLM adapter, carried on the event, and consumed by a `Notifier` port with swappable adapters. Per-camera policy (which concerns notify, clip lengths, forced-look interval) extends the existing `cameras.json` + `PATCH /cameras/{id}` + Camera record panel surface rather than inventing a second settings path.

**Tech Stack:** Python 3.12, FastAPI, pydantic v2, pytest; React 19 + TanStack Query + Vitest.

## Global Constraints

- **`domain/` and `ports/` stay pure** — no third-party I/O, no clock reads, no dependency on outer layers or `config`. `ai-engine/tests/test_architecture.py` enforces this and catches aliased/dynamic/relative imports. Do not weaken it.
- **The published event carries no model identity** (spec §3.3). Nothing added here may leak one.
- **No structured field may read as a detector output.** No booleans like `fall_detected`; no `confidence: certain`; no new `EscalationReason`. Concerns are `{kind, confidence, evidence}` with a mandatory constant `basis: "single_frame_vlm"`.
- **`confidence`** is exactly `possible` | `likely`. **`kind`** is exactly `collapse` | `altercation` | `self_harm` | `medication` | `distress` | `other`.
- **Notification is best-effort and never blocks or fails the pipeline.** A hanging webhook must not stall a camera runner or delay a clip.
- **Every GPU-requiring test is `@pytest.mark.gpu`**; CI runs `-m "not gpu and not integration"`.
- `filterwarnings = ["error"]` is in `pyproject.toml` — a warning is a test failure.
- Engine commands: `ai-engine/.venv/bin/python -m pytest ai-engine -q -m "not gpu"`, `ai-engine/.venv/bin/ruff check ai-engine`, `ai-engine/.venv/bin/ruff format --check ai-engine`, `cd ai-engine && .venv/bin/mypy`. Web from `web/`: `npx vitest run`, `npx tsc -b`, `npx oxlint`.
- Commit to `master` (standing user directive, no branches). End every commit body with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- After ANY change to `contracts/openapi/ai-engine.yaml`, run `npm run gen` in `web/` and commit the regenerated `web/src/api/engine.types.ts` — CI now fails on drift.

---

### Task 1: `WelfareAssessment` domain types

**Files:** Create `ai-engine/sentinel_ai/domain/welfare.py`; Test `ai-engine/tests/domain/test_welfare.py`

**Produces:**
- `class ConcernKind(StrEnum)` — `COLLAPSE`, `ALTERCATION`, `SELF_HARM`, `MEDICATION`, `DISTRESS`, `OTHER`
- `class Confidence(StrEnum)` — `POSSIBLE`, `LIKELY`
- `@dataclass(frozen=True, slots=True) class WelfareConcern` — `kind: ConcernKind`, `confidence: Confidence`, `evidence: str`
- `@dataclass(frozen=True, slots=True) class WelfareAssessment` — `concerns: tuple[WelfareConcern, ...]`, `basis: Literal["single_frame_vlm"] = "single_frame_vlm"`
- `WelfareAssessment.none() -> WelfareAssessment` — the empty assessment
- `WelfareAssessment.highest_confidence(kind) -> Confidence | None`

Validate in `__post_init__`: `evidence` non-empty after strip, and no duplicate `kind` (keep the highest confidence instead — a model listing `collapse` twice is one concern, and duplicates would double-count in any routing rule).

- [ ] **Step 1: Write the failing test.** Cover: construction; empty `evidence` rejected; whitespace-only `evidence` rejected; duplicate kinds collapse to the highest confidence; `none()` has no concerns; `basis` is always `"single_frame_vlm"` and cannot be set to anything else (assert the type is `Literal` by asserting the value after construction); `highest_confidence` returns `None` for an absent kind.
- [ ] **Step 2: Run to verify it fails.** `ai-engine/.venv/bin/python -m pytest ai-engine/tests/domain/test_welfare.py -q` → ImportError.
- [ ] **Step 3: Implement**, matching the module-docstring style of `domain/entities.py` — say WHY there are no booleans and no `certain`, citing that a single-frame VLM cannot honestly produce either.
- [ ] **Step 4: Run to verify it passes**, plus `ruff check`, `ruff format --check`, `mypy`, and `pytest ai-engine/tests/test_architecture.py -q` (this file is in `domain/`, so purity is enforced).
- [ ] **Step 5: Commit** `feat(domain): model welfare concern as an opinion, not a detection`

---

### Task 2: Carry welfare on `SceneDescription` and the event

**Files:** Modify `ai-engine/sentinel_ai/ports/vision_llm.py`, `ai-engine/sentinel_ai/domain/entities.py` (the `Event`), `contracts/events/anomaly_event.schema.json`, the event codec in `ai-engine/sentinel_ai/adapters/` (find it — `test_event_codec.py` covers it); Tests: the existing codec test file plus the event tests.

**Consumes:** Task 1's types. **Produces:** `SceneDescription.welfare: WelfareAssessment` (defaulting to `WelfareAssessment.none()`), and a `welfare` object on the event JSON schema.

The schema addition must be **optional** with `additionalProperties: false` preserved, so an existing consumer keeps validating. Bump `schema_version` only if the existing tests require it — check what they assert before deciding, and say which you chose and why in the report.

- [ ] **Step 1: Write the failing test** — an event with concerns round-trips through the codec preserving kind/confidence/evidence/basis; an event with no concerns omits `welfare` entirely rather than emitting an empty object (an empty object reads as "assessed, nothing found"; omission reads as "not assessed" — they differ and the wire must not conflate them); a payload from before this change still validates.
- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Verify** full engine suite, ruff, mypy.
- [ ] **Step 5: Commit** `feat(events): carry the vision model's welfare assessment on the event`

---

### Task 3: Ask the model, and parse what it says

**Files:** Modify `ai-engine/sentinel_ai/adapters/vision/qwen25vl.py` AND `ai-engine/sentinel_ai/orchestrator/scheduler.py`; Tests: the vision adapter's existing test file (under `ai-engine/tests/adapters/vision/`) and the scheduler's.

**This task also owns the wiring**, found missing during Task 2: `scheduler.py`'s `_assemble` builds the `Event` from the `SceneDescription` but does not copy `welfare` across, so `Event.welfare` would stay `WelfareAssessment.none()` in production no matter what the model reported — and Task 10 would then notify on nothing, silently, forever. Producing the assessment and carrying it to the event belong together; a test must assert an assessment produced by the adapter reaches the assembled event.

`_build_text_prompt` gains explicit questions for collapse/unresponsiveness, physical altercation, apparent self-harm, apparent medication or unlabelled-container ingestion, and other visible distress. `_parse_response` parses an optional `welfare` array into `WelfareAssessment`.

**Binding on the prompt:** it must NOT ask the model to name a substance, judge a dose, or decide whether medication was prescribed — only to report that an apparent ingestion was seen and describe what it saw. Clinical judgement is not something a single-frame VLM may be asked for in a custodial setting.

**Parsing must be defensive**, since this is untrusted model output: an unknown `kind` maps to `OTHER` rather than raising; an unknown `confidence` maps to `POSSIBLE` (the weaker one — never round *up* into a routing decision); a missing or malformed `welfare` key yields `WelfareAssessment.none()` and never fails the description. `_parse_response` already has a malformed-response fallback path; welfare must degrade to `none()` on that path too, not vanish silently in a way that looks like "assessed, found nothing".

**These tests are CPU-only** — they exercise `_build_text_prompt` and `_parse_response` directly, which is why those are separate functions. Do not add a GPU test.

- [ ] **Step 1: Write the failing test** — prompt mentions each of the five concerns; prompt does NOT contain substance-naming or dosage language (assert on specific words); a well-formed welfare array parses; unknown kind → `OTHER`; unknown confidence → `POSSIBLE`; absent welfare → `none()`; malformed JSON → `none()` and the existing fallback description.
- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement.** Extend the module docstring's existing "what it is and what it is not" section rather than writing a new one — it already makes exactly the right argument and must now cover medication too.
- [ ] **Step 4: Verify.**
- [ ] **Step 5: Commit** `feat(vision): assess welfare concerns including apparent medication intake`

---

### Task 4: The `Notifier` port

**Files:** Create `ai-engine/sentinel_ai/ports/notifier.py`; Test `ai-engine/tests/ports/test_notifier_contract.py` (follow the existing port-contract test pattern — find one, e.g. for `event_publisher`).

**Produces:**
- `@dataclass(frozen=True, slots=True) class WelfareNote` — `event_id: UUID`, `camera_id: str`, `label: str`, `zone: str | None`, `occurred_at: float`, `severity: str`, `description: str`, `concerns: tuple[WelfareConcern, ...]`, `clip_uri: str | None`
- `class Notifier(ABC)` with `async def notify(self, note: WelfareNote) -> None`
- A `FakeNotifier` for tests, recording notes in order — put it wherever the existing CPU fakes live (find them; there is an established location).

`clip_uri` is `str | None` because a notification can legitimately precede a clip that failed to write. A note with no clip is still worth sending; say that in the docstring.

- [ ] **Step 1: Write the contract test** — `FakeNotifier` records in order; `WelfareNote` is frozen; a note with `clip_uri=None` is valid.
- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Verify**, including the architecture purity test (`ports/` is enforced pure).
- [ ] **Step 5: Commit** `feat(ports): add the welfare notifier port`

---

### Task 5: Logging notifier — the default

**Files:** Create `ai-engine/sentinel_ai/adapters/notifiers/__init__.py` and `logging.py`; Test `ai-engine/tests/adapters/notifiers/test_logging_notifier.py`

The default adapter, so a deployment that configures nothing still leaves an observable trail and tests need no network. Logs one structured line per note at INFO.

**Must not log the `clip_uri` credentials problem into existence:** the URI is a MinIO object URL, not a credential, so it is safe — but do not log the full note at DEBUG in a way that would include anything added later without review. Log named fields explicitly, never `repr(note)`.

- [ ] **Step 1: Write the failing test** — caplog captures one INFO line per note containing camera id, severity and each concern kind; a note without concerns still logs; no `repr(note)` in output.
- [ ] **Step 2: Run to verify it fails.** **Step 3: Implement. Step 4: Verify. Step 5: Commit** `feat(notifiers): log welfare notifications by default`

---

### Task 6: Webhook notifier

**Files:** Create `ai-engine/sentinel_ai/adapters/notifiers/webhook.py`; Test `ai-engine/tests/adapters/notifiers/test_webhook_notifier.py`

POSTs JSON to a configured URL. Uses the HTTP client already in the dependency set (check `pyproject.toml` — `httpx` is present; do not add a dependency).

Requirements, each of which needs a test:
- A total timeout, configurable, defaulting to 5 s. A hanging endpoint must not hold anything.
- Retry with backoff on connection errors and 5xx; **no retry on 4xx** — a 400 will be 400 again, and retrying a 401 just repeats a rejected credential.
- On final failure, spool to the existing dead-letter directory rather than dropping, reusing the existing dead-letter writer (`adapters/publishers/dead_letter.py`) rather than writing a second one.
- The body includes `clip_uri` and a literal note that the URL may require credentials the recipient does not have (assumption 3 in the spec).
- **Never raises to its caller.** A notifier that throws would take down the pipeline it is meant to observe.

Test with `httpx.MockTransport` — no real sockets, no `respx` dependency.

- [ ] **Step 1: Write the failing tests** covering every bullet above, including that a 4xx is NOT retried (assert the call count) and that a timeout does not raise.
- [ ] **Step 2: Run to verify they fail. Step 3: Implement. Step 4: Verify. Step 5: Commit** `feat(notifiers): deliver welfare notifications to a webhook`

---

### Task 7: Per-camera welfare policy in `cameras.json`

**Files:** Modify `ai-engine/sentinel_ai/adapters/config/camera_file.py`; Test: its existing test file.

Adds to `CameraConfig`, all optional so every existing camera file keeps loading:
- `notify_on: frozenset[ConcernKind]` — default all kinds; `[]` means never notify
- `notify_min_confidence: Confidence` — default `LIKELY`
- `clip_preroll_seconds: float | None` / `clip_postroll_seconds: float | None` — `None` means use the global default
- `summary_interval_seconds: float | None` — `None` means use the profile default

**Absence and empty are different**, exactly as `zone` already distinguishes them: `notify_on` absent means "all kinds" while `notify_on: []` means "never notify this camera". Get this wrong and a camera the operator silenced starts alerting, or vice versa. Test both.

An unknown concern kind in the file is a **load error**, matching how `_zone_from` treats an unknown zone — a typo'd kind is a concern the operator meant to route and silently would not.

- [ ] **Step 1: Write the failing tests** — each field round-trips; absent vs `[]` differ; unknown kind is a `CameraConfigError` naming the field; a pre-existing camera file with none of these fields still loads.
- [ ] **Step 2: Run to verify. Step 3: Implement. Step 4: Verify. Step 5: Commit** `feat(config): per-camera welfare notification policy`

---

### Task 8: Make the new fields runtime-editable

**Files:** Modify `ai-engine/sentinel_ai/adapters/config/camera_file.py` (the `CameraEdit`/`edited_document` path), `ai-engine/sentinel_ai/api/schemas.py`, `ai-engine/sentinel_ai/api/routes.py`, `contracts/openapi/ai-engine.yaml`; then regenerate `web/src/api/engine.types.ts`.

Extends `CameraEdit` and `CameraEditRequest` with the Task 7 fields. **The `UNSET` sentinel discipline is binding** — for each new field, "not mentioned" and "explicitly null" must stay distinguishable end to end, exactly as `zone` does today. `clip_preroll_seconds: null` means "revert to the global default"; omitting it means "leave alone".

Numeric bounds mirror the existing config: pre-roll `>= 0`, post-roll `> 0`, `summary_interval_seconds > 0`. A body carrying `url` or `profile` must still be a 422.

- [ ] **Step 1: Write the failing tests** — each field edits and persists; omitted vs null differ for the nullable ones; out-of-range values are 422; `url`/`profile` still 422; the whole-document atomicity guarantees still hold.
- [ ] **Step 2: Run to verify. Step 3: Implement.** **Step 4: Verify**, including `test_openapi_contract.py`, then `cd web && npm run gen` and confirm `git diff` shows the regenerated types. **Step 5: Commit** both together: `feat(api): edit per-camera welfare policy at runtime`

---

### Task 9: Honour the per-camera clip lengths and forced-look interval

**Files:** Modify `ai-engine/sentinel_ai/pipeline/runner.py` and wherever `CameraProfile` is built from `CameraConfig` (find it — likely `main.py`); Tests: the runner's existing tests.

The per-camera values override the globals when set. `summary_interval_seconds` flows into the `CameraProfile` the gate uses.

**A live edit of these takes effect on the next escalation, not retroactively** — a clip mid-recording keeps the length it started with. State that in the docstring and test it, because the alternative (mutating an in-flight clip's bounds) is how a clip ends up with impossible timestamps.

- [ ] **Step 1: Write the failing tests** — a camera with overrides gets them; one without gets the globals; an edit applied mid-clip does not change the clip in progress.
- [ ] **Step 2: Run to verify. Step 3: Implement. Step 4: Verify. Step 5: Commit** `feat(pipeline): honour per-camera clip lengths and summary interval`

---

### Task 10: Dispatch notifications, off the critical path

**Files:** Modify `ai-engine/sentinel_ai/orchestrator/service.py` (or wherever the event is published after assembly — find the publish call), `ai-engine/sentinel_ai/main.py` (compose the notifier), `ai-engine/sentinel_ai/config.py` (settings: notifier kind, webhook URL, timeout); Tests: the orchestrator's existing tests.

Order, from the spec: process → describe → clip finalised → publish → notify. Notification is dispatched **after** publishing, with its own timeout, and a failure is logged and never propagated.

The routing rule from spec §5, which needs its own tests: notify when a concern's kind is in the camera's `notify_on` AND its confidence meets `notify_min_confidence`. `possible` alone does not notify unless the threat score is already in the caution band or above.

**Do not spawn an unawaited bare `asyncio.create_task`** — this codebase has already been bitten by unmanaged tasks. Follow whatever pattern the orchestrator already uses for background work, and make sure shutdown waits for or cancels in-flight notifications rather than leaking them.

- [ ] **Step 1: Write the failing tests** — a `likely` collapse on a camera configured for it notifies once with the clip uri; a `possible` collapse below the caution band does not; a camera with `notify_on: []` never notifies; a notifier that raises does not fail the pipeline and the event is still published; shutdown does not leak a pending notification.
- [ ] **Step 2: Run to verify. Step 3: Implement. Step 4: Verify. Step 5: Commit** `feat(orchestrator): notify on welfare concerns after the event is published`

---

### Task 11: Console — welfare policy on the Camera record panel

**Files:** Modify `web/src/routes/camera/CameraRecordPanel.tsx`, `web/src/lib/cameraEdit.ts`, `web/src/api/engineClient.ts`; Tests: `web/src/routes/camera/__tests__/CameraRecordPanel.test.tsx`, `web/src/lib/__tests__/cameraEdit.test.ts`

Adds rows for `notify_on` (checkboxes per concern kind), `notify_min_confidence` (select), and the three numeric fields. Read-only branch shows them as values; editable branch edits them.

**`buildCameraEdit`'s only-changed-fields rule extends to every new field** — including that an emptied `notify_on` sends `[]` (never notify) while an untouched one is omitted. That distinction is the same class of bug the baseline fix caught; test it explicitly.

Keep the panel honest about what it is configuring: a short note that these route a vision-language model's opinion about single frames, not a detector's verdict.

- [ ] **Step 1: Write the failing tests** — each field renders read-only and editable; emptying `notify_on` sends `[]`; an untouched `notify_on` is omitted from the body; numeric validation mirrors the engine's bounds.
- [ ] **Step 2: Run to verify. Step 3: Implement. Step 4: Verify** `npx vitest run && npx tsc -b && npx oxlint`. **Step 5: Commit** `feat(web): configure per-camera welfare notifications`

---

### Task 12: Documentation

**Files:** `docs/operations.md`, `docs/configuration.md`, `README.md`

- The new settings and per-camera fields, with defaults.
- The notification channel choice, and that the clip travels as a URL that may need credentials.
- **The honest limits, where an operator will actually read them** — not only in a source docstring: this asks a vision-language model about single frames; it is not a fall detector; a stretcher carry was missed entirely in measurement; and notification latency is bounded below by the post-roll length, so a shorter post-roll means a faster alert and less evidence.
- Update the README capability table.

- [ ] **Step 1: Write the docs. Step 2: Verify** every claim against the shipped code — do not describe intended behaviour. **Step 3: Commit** `docs: describe welfare monitoring and its limits`

---

## Verification

Engine: `pytest -m "not gpu"`, `ruff check`, `ruff format --check`, `mypy` all clean.
Web: `npx vitest run`, `npx tsc -b` 0 errors, `npx oxlint` no new warnings.
Contract: `cd web && npm run gen:api && git diff --exit-code -- src/api/engine.types.ts` — no output.
