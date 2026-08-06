# `contracts/` — the engine ↔ UI seam

This directory is the **only** coupling between `ai-engine/` and `web/`, and
later between both and the Phase 1C Go backend. Two files:

| File | What it defines | Consumed by |
|---|---|---|
| `openapi/ai-engine.yaml` | The engine's HTTP API | `web/src/api/engine.types.ts` (generated) |
| `events/anomaly_event.schema.json` | The anomaly event wire format | `ai-engine`'s `event_codec`, which validates every payload before it is published; and the Phase 1C consumer, when it exists |

## The rule

> **`web/` may read `contracts/`, never `ai-engine/`. `ai-engine/` never reads
> `web/`. Types are generated from the contract, never hand-written.**

Two trees, two toolchains, two languages, and a third consumer arriving. A
generated shared schema is the only coupling that survives that.

**Enforcement status, honestly:** the OpenAPI side is enforced by a test (below).
The directory boundary is **not** — it is true structurally today, but no
fitness test asserts it, and there is no web CI at all. Closing that gap is a
known outstanding item. The model to copy is
`ai-engine/tests/test_architecture.py`, which does exactly this job for the
Python layering and has caught real drift.

A second boundary *inside* `web/` is documented and must also hold: nothing
under `src/recorder/` imports from `src/api/`, or the reverse. Those are two
different products — the AI engine and the `sentinel-ingest` recorder appliance
— with two clients, two type sets and two error hierarchies. They are
deliberately **not** factored into a shared `createClient`: a shared abstraction
is exactly how the two would end up sharing a base URL or an error type by
accident.

## Generating types

```bash
cd web && npm run gen      # openapi-typescript -> src/api/engine.types.ts
```

**The event schema has no TypeScript generation, and the console does not
consume it.** `json-schema-to-typescript` is in `web/`'s devDependencies from an
earlier intent, but there is no `gen:events` script and no `src/events/`. What
the console renders is `RecentEventEntry` from the *OpenAPI* types — the engine's
bounded in-memory ring entry, which is a deliberately smaller shape than the
published event: no `schema_version`, and its `welfare_concerns` is a flattened
list rather than the event's `welfare` object (the assessment's `basis` is not
projected). Do not read console code as evidence of what goes on the wire; the
wire is the JSON schema and the Phase 1C consumer is its first real reader.

Generated output is **committed**, so a fresh checkout builds without running
codegen. Never hand-edit a generated file; regenerate it.

**Exception:** the recorder appliance publishes no OpenAPI document (verified —
`/openapi.json`, `/openapi` and `/api/schema` all fail to return one), so
`web/src/recorder/recorder.types.ts` is hand-derived from live responses. The
"generate, never hand-write" rule applies to the AI engine only.

## The drift test

`ai-engine/tests/test_openapi_contract.py` re-renders the OpenAPI document from
the live FastAPI app and asserts **byte equality** with the committed copy.

```
the committed OpenAPI document has drifted from the app —
run `python -m sentinel_ai.openapi_export` and commit ai-engine.yaml
```

That is the fix: regenerate and commit. It also asserts the document covers
**exactly** the set of endpoints Phase 1C consumes — an equality assertion, so
adding a route is a deliberate, visible change to that test, not a silent one —
that `describe` is POST-only and `telemetry` GET-only, and that the 404 a client
will actually meet
(`UnknownCameraError`) is in the contract — a generated Go client that does not
know about it would treat it as a transport failure.

It is a pytest assertion rather than a CI workflow step deliberately: CI already
runs the suite, so there is nothing new to remember to add, and it fails on the
developer's machine at the moment the route changes rather than twenty minutes
later on a push.

**There is no equivalent drift test for the event schema.** `event_codec`
validates outgoing payloads against it at runtime, which catches structural
violations, but nothing checks that the schema's *prose descriptions* still
describe the code. See the known drift below.

## Changing a contract

1. Change the engine (a route, or `Event`/`event_codec`).
2. For OpenAPI: `python -m sentinel_ai.openapi_export`, commit the YAML.
   For events: hand-edit `anomaly_event.schema.json` — including the
   `description` prose, which is the only documentation a consumer gets.
3. `cd web && npm run gen`, commit the generated types.
4. Run both suites. The drift test fails loudly if you skipped step 2.

**Adding an `EscalationReason` requires a schema change.** The `reason` enum is
closed; a new value fails validation at publish time, in production, after the
VLM has already been paid for. Add it to the schema in the same commit as the
trigger.

**`schema_version` is currently `1`.** Adding an *optional*, nullable field does
not need a bump — a payload written by an older build still validates, which is
what lets the disk spool replay across a deploy. Removing a field, changing a
type, or narrowing an enum does need one.

## Known drift: `occurred_at`

The schema currently describes `occurred_at` as:

> "Derived from a single wall+monotonic anchor taken once per engine process…"

**That is out of date.** The code uses a **per-camera** anchor
(`VlmScheduler._to_epoch`), established lazily from each camera's own first-seen
`source_timestamp`. The per-process design it describes was tried, and it was
wrong: because `RtspSource` stamps `time.monotonic()` while `FileSource` stamps
container pts from 0.0, a replay camera's events landed roughly one machine
uptime in the past — about 14 hours on the box it was found on — and two cameras
in the same process landed hours apart for events observed seconds apart.

The consequence for a consumer is a *weaker* guarantee than the schema promises:
the difference between two `occurred_at` values **from the same camera** is
exact, but comparing across two cameras is precise only to when each camera's
anchor was established (typically well under a second). Full reasoning in
[ADR 5](../docs/decisions.md#5-occurred_at-is-unix-epoch-with-a-per-camera-anchor).

**Fix the description before the Phase 1C Go consumer is written against it.**
It was not caught because the drift test covers OpenAPI only, and prose is not
machine-checkable.

## Field notes for consumers

- **Sort and display on `occurred_at`** — Unix epoch seconds, UTC, fractional.
- **Never sort on `source_timestamp`** — it is the camera's own timeline,
  meaningful only within one process run for one camera. It is what correlates
  an event with its clip and telemetry, and nothing else. Optional and
  nullable.
- **Events may arrive out of order.** Sort on receipt.
- **`description_unavailable: true`** means the VLM timed out or ran out of
  memory. The event is still real; the prose is not. Render it as missing, not
  as a description.
- **Events carry no model identity**, deliberately. Neither the description nor
  any field names the model. `version()` and `capabilities()` do, but those are
  orchestrator-internal, behind `/health`, and a different boundary.
