# Camera Record Editing in the Console

**Date:** 2026-08-02
**Status:** Approved for planning
**Predecessor:** [Phase 1B AI engine slice](2026-08-01-sentinelai-phase1b-ai-engine-slice-design.md)
**Engine side:** already built — `PATCH /cameras/{camera_id}`, `CameraFileStore`, `SENTINEL_ENABLE_CAMERA_WRITES`

## 1. Goal

Give the console a way to read and edit a camera's stored record — its `label` and its
`zone` — against the engine's existing write endpoint, and make the label the name the
console actually displays.

**Done means:** an operator on a deployment with camera writes enabled can rename a camera
and move it between zones from that camera's page, see the stored result rather than what
they typed, and see the new name on the dashboard, the camera page and the site map. On a
deployment with writes disabled — the default — the same panel shows the same record,
read-only, and says why.

### 1.1 Why this exists now

The engine side of this landed complete: the endpoint, an atomic `cameras.json` writer, the
opt-in flag, the contract and the documentation. Nothing consumes it. Two consequences, both
of which this design closes:

- `README.md` lists "Editing cameras from the console" as **Working**. There is no console
  UI, so that row currently overstates what ships.
- `CameraStatus` gained a `label` field that no console code reads. `DashboardPage` prints
  `camera.camera_id`, `CameraPage` prints the route param, and `CameraMarker` prints
  `cameraId` in both its visible text and its `aria-label`. A rename would therefore take
  visible effect in exactly no place — making the write endpoint, as shipped, unobservable
  from the product it exists to serve.

## 2. Inherited constraints

Binding, from the engine side and from the console's existing conventions:

- **The engine is the authority on whether a write is permitted.** `config_writable` is a
  display hint that lets the console avoid offering a control which would 403. It is not a
  security control, and the 403 path is handled as a real outcome rather than an impossible
  one. The engine has no authentication at all in this phase.
- **`config_writable` is published on `GET /cameras`, not on `GET /cameras/{id}/telemetry`.**
  Any screen that offers editing must read the list endpoint.
- **Omitted and null are different instructions.** Omitting `zone` leaves the grouping alone;
  sending `zone: null` ungroups the camera. Omitting `label` leaves it alone; sending
  `label: null` is a 422. `{}` is also a 422.
- **`url` and `profile` are not editable and are rejected, not ignored.** A body carrying
  either is a 422 naming the field. Neither is returned by any endpoint — an RTSP URL
  routinely carries credentials, and an unauthenticated API moves it in neither direction.
- **The console never conflates the two backends.** The AI engine (`:8000`) and the recorder
  appliance (`:8080`) are separate products with separate clients. This work touches only
  `web/src/api/*` and the non-recorder routes.
- `web/src/api/engine.types.ts` is generated from `contracts/openapi/ai-engine.yaml` by
  `npm run gen:api` and committed. Hand-written duplicates of contract types are not
  acceptable; the generated types are the source of truth.

## 3. What is being built

### 3.1 Client layer — `web/src/api/engineClient.ts`

Add `updateCamera(cameraId, edit)` issuing `PATCH /cameras/{id}`, reusing the existing
`request<T>` helper. That helper already unwraps FastAPI's `{"detail": ...}` into
`EngineHttpError.status` and `.message`, so every failure arrives typed and carrying the
engine's own prose.

`edit` is typed as `components['schemas']['CameraEditRequest']` from the generated types, so a
contract change breaks `tsc -b` rather than production.

**One change to `request` is required.** It currently sets only `Accept`, never
`Content-Type` — correct so far, because every call has been a GET or a bodyless POST. A
PATCH carrying a JSON body without `Content-Type: application/json` is a 422 from FastAPI.
The header is therefore set on this call rather than globally, so bodyless requests continue
to send no content header.

### 3.2 The request body carries only what changed

The builder serialises `label` only when it differs from the stored record, and `zone` only
when it differs. This is not an optimisation — it is the semantics:

- A form that always sent both could not express "ungroup this camera" distinctly from
  "don't touch the zone", because both would be a `zone` key on the wire.
- Re-asserting an unchanged field is how one operator's open window silently reverts another
  operator's change.

Since `{}` is a 422 by design, Save is disabled while nothing is dirty; the empty request is
therefore unreachable rather than merely handled.

### 3.3 Form state against a polling query — uncontrolled until dirty

`useCameras()` repolls every 5 s. The panel keeps local state seeded from the server record,
but takes over only once the operator touches a field:

- **Pristine:** polls flow through, the panel tracks the engine, including a change another
  operator made.
- **Dirty:** polls no longer overwrite the field. The value the record held at the moment
  editing began is retained as a baseline; when a poll returns a record differing from that
  baseline, a "changed elsewhere since you started editing" notice appears, naming the field,
  rather than the input silently changing under the cursor. The operator's text is never
  discarded — the notice informs, and Save remains available.
- **After a successful save:** local state is cleared and the panel returns to tracking the
  server, rendering the `CameraEditResponse` body — the record as stored, not as hoped.

Rejected alternatives: seed-once-and-never-re-seed (the panel shows a stale record forever,
and after a 409 the operator is looking at a form that no longer matches the file); an
explicit edit-mode toggle (unambiguous, but adds a mode and a click to a two-field form).

The chosen approach costs one effect and a dirty flag, and buys the property that matters:
two operators on one console cannot silently overwrite each other — the same failure the
engine's 409 exists to prevent.

### 3.4 The panel — `web/src/routes/camera/CameraRecordPanel.tsx`

A new file rather than more weight on `CameraPage.tsx`, which is already around 300 lines and
holds four panels. The panel takes the stored record and a `writable` boolean as props and
owns no queries, so it is testable by direct render with no server at all. `CameraPage` does
the fetching and passes down.

| Row | Writable | Read-only |
|---|---|---|
| Label | `Input`, trimmed, non-empty, ≤120 chars | static text |
| Zone | `Select` over `room` / `corridor` / `dayroom` / *Ungrouped* | static text |
| Zone kind | derived, always read-only — shown beside the zone as its consequence | same |
| Source | `restart required`, **no value** | same |
| Profile | `restart required`, **no value** | same |

The Source and Profile rows are driven off `restart_required_fields` from the response rather
than a hardcoded pair, so if the engine ever makes one of them editable the panel stops
claiming otherwise. They show no value because no endpoint returns one (§2).

Label validation mirrors the engine's `CameraLabel` constraint exactly — trim first, then
require non-empty and ≤120 — so the two agree on what an empty label is. `""` and `"   "`
are the same answer.

### 3.5 Gating

`CameraPage` adds `useCameras()` and takes **both** the flag and the panel's record from it.
`label` and `zone` are also on the telemetry response the page already polls at 2 s, but the
flag is not, and sourcing the record and its writability from one snapshot is what stops the
panel from rendering a record it is simultaneously wrong about the editability of. The panel
therefore refreshes at the list's 5 s cadence, which is ample for a field an operator edits
by hand.

When the flag is false the panel renders read-only with a `Notice` stating that camera writes
are disabled on this engine (`SENTINEL_ENABLE_CAMERA_WRITES`) and that the way to change the
record is editing `cameras.json` and restarting. The record is still shown: hiding the panel
would leave the camera's stored label and zone with nowhere in the console they can be read.

Because the flag is re-read on every poll, an engine restarted with writes disabled flips the
panel to read-only without a page reload.

### 3.6 Error handling

All branches key off `EngineHttpError.status` and render the engine's own `detail` rather
than a re-worded client-side copy.

| Status | Meaning | Console behaviour |
|---|---|---|
| 403 | Writes were turned off between load and save | Show the detail; invalidate the cameras query so the panel flips to read-only rather than inviting a retry that cannot work |
| 404 | The camera is not configured in the running engine | Show the detail |
| 409 | `cameras.json` changed by hand and no longer holds this camera; **nothing was written** | Show the detail and say the file needs reconciling. **No auto-retry** — retrying is precisely how a console overwrites a file someone is editing |
| 422 | Client and contract have drifted | Should be unreachable, since the client validates the label and offers only real zones. Surfaced anyway: silence would hide the drift |
| — | `EngineUnreachableError` | Existing path, unchanged |

### 3.7 Making `label` the displayed name

With `camera_id` as the fallback while a query is pending:

- **Dashboard tiles** — heading renders the label.
- **Camera page** — `h1` renders the label; the camera id moves to a `<code>` subtitle, so
  the identifier an operator needs for a bug report does not vanish.
- **Site map markers** — visible text and `aria-label` both render the label.

`CameraMarker` needs care: it already has a local `const label` holding its composed aria
string. The new prop is therefore `displayName`, and the local is renamed `ariaLabel`.
Shadowing two different meanings of "label" in one component is how the wrong string reaches
the accessible name.

## 4. Testing

Following existing patterns — MSW against the shared `web/src/test/mswServer.ts`, as
`DashboardPage.test.tsx` already does, rather than fetch stubs.

**`engineClient`**
- PATCH is issued with `Content-Type: application/json`.
- Only dirty fields are serialised.
- An omitted `zone` and an explicit `zone: null` produce different request bodies.
- Each status maps to the right typed error carrying the engine's detail.

**`CameraRecordPanel`**
- Read-only render when `writable` is false, including the notice naming the env var.
- Save is disabled while pristine.
- Label validation: whitespace-only rejected, 121 characters rejected.
- Ungrouping via the zone select sends `zone: null`.
- 403, 409 and 422 each render the engine's detail.
- **A poll landing mid-edit does not clobber a dirty field, but does update a pristine one.**
  This is the test the §3.3 design exists for.

**Display**
- A camera with a label renders it on tile, page heading and marker.
- Before the query resolves, all three fall back to the camera id rather than rendering
  empty. Note that `label` is a **required** field on `CameraStatus` and the engine defaults
  it to the camera id at load, so a successful response never omits it — the fallback covers
  the pending state, not a missing field, and should not be described as the latter.

## 5. Out of scope

- Authentication. JWT is Phase 1C's Go backend; this design does not make the port safe and
  does not pretend to.
- Editing `url` or `profile` from the console, which are restart-required by the engine's own
  design.
- Creating or deleting cameras. The engine exposes no such endpoint.
- Bulk or multi-camera editing.

## 6. Consequences for existing documentation

- `README.md`'s "Editing cameras from the console" row becomes true as written.
- `docs/operations.md` §"Editing cameras from the console" should gain a short note that the
  console surfaces this, and how the read-only state presents.
