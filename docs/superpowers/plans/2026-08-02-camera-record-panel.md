# Camera Record Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an operator edit a camera's label and zone from its camera page, gated on the engine's `config_writable` flag, and make the label the name the console displays.

**Architecture:** A presentational `CameraRecordPanel` owns the form state and the PATCH mutation but does no fetching; `CameraPage` supplies the stored record and the writable flag from `useCameras()`. Request bodies carry only fields that actually changed, because on this endpoint an omitted key and a null key are different instructions. The form is uncontrolled until the operator touches it, so a 5 s poll cannot overwrite what they are typing.

**Tech Stack:** React 19, TanStack Query v5, react-router-dom v7, Tailwind v4, Vitest + Testing Library + MSW, TypeScript strict.

## Global Constraints

- **The generated types are the source of truth.** All contract types come from `components['schemas'][...]` in `web/src/api/engine.types.ts`. Never hand-write a duplicate of a contract type. If the contract changes, run `npm run gen` in `web/` — CI now fails on drift.
- **Omitted ≠ null on `PATCH /cameras/{id}`.** Omitting `zone` leaves the grouping alone; `zone: null` ungroups. Omitting `label` leaves it alone; `label: null` is a 422. `{}` is a 422.
- **`config_writable` is a display hint, never a security control.** The engine is the authority. The 403 path must be implemented as a real outcome.
- **`url` and `profile` are never displayed with a value.** No endpoint returns them; an RTSP URL carries credentials.
- **Label validation mirrors the engine's `CameraLabel`:** trim first, then require non-empty and ≤ 120 characters.
- **Commands** run from `web/`: `npx vitest run <path>`, `npx tsc -b`, `npx oxlint`.
- Every task ends green on `npx tsc -b` and its own test file.

---

### Task 1: `updateCamera` client call

**Files:**
- Modify: `web/src/api/engineClient.ts`
- Test: `web/src/api/__tests__/engineClient.test.ts` (create)

**Interfaces:**
- Consumes: the existing `request<T>` helper and `EngineHttpError` in `engineClient.ts`.
- Produces:
  - `updateCamera(cameraId: string, edit: CameraEditRequest): Promise<CameraEditResponse>`
  - `EngineHttpError.detail: string` — the engine's raw `detail` string, without the `AI engine returned 409: ` prefix that `.message` carries.
  - Re-exported types `CameraEditRequest`, `CameraEditResponse`, `Zone`, `ZoneKind`.

- [ ] **Step 1: Write the failing test**

Create `web/src/api/__tests__/engineClient.test.ts`:

```ts
import { afterEach, describe, expect, it } from 'vitest'
import { http, HttpResponse } from 'msw'
import { server } from '@/test/mswServer'
import { ENGINE_BASE_URL } from '@/api/config'
import { EngineHttpError, updateCamera } from '@/api/engineClient'

afterEach(() => server.resetHandlers())

/** Captures the request the client actually put on the wire. */
function capturePatch(status = 200, body: unknown = null) {
  const seen: { contentType: string | null; json: unknown } = { contentType: null, json: null }
  server.use(
    http.patch(`${ENGINE_BASE_URL}/cameras/:cameraId`, async ({ request }) => {
      seen.contentType = request.headers.get('content-type')
      seen.json = await request.json()
      if (status !== 200) {
        return HttpResponse.json({ detail: body }, { status })
      }
      return HttpResponse.json(body)
    }),
  )
  return seen
}

const storedRecord = {
  camera_id: 'avenue_01',
  label: 'Avenue entrance',
  zone: 'corridor' as const,
  zone_kind: 'common_area' as const,
  persisted: true as const,
  restart_required_fields: ['url', 'profile'],
}

describe('updateCamera', () => {
  it('sends a JSON content type, which FastAPI rejects the body without', async () => {
    const seen = capturePatch(200, storedRecord)

    await updateCamera('avenue_01', { label: 'Avenue entrance' })

    expect(seen.contentType).toContain('application/json')
  })

  it('sends exactly the fields it was given, so an omitted zone stays omitted', async () => {
    const seen = capturePatch(200, storedRecord)

    await updateCamera('avenue_01', { label: 'Avenue entrance' })

    expect(seen.json).toEqual({ label: 'Avenue entrance' })
    expect(Object.keys(seen.json as object)).not.toContain('zone')
  })

  it('sends an explicit null zone, which is the instruction to ungroup', async () => {
    const seen = capturePatch(200, { ...storedRecord, zone: null, zone_kind: null })

    await updateCamera('avenue_01', { zone: null })

    expect(seen.json).toEqual({ zone: null })
  })

  it('returns the stored record the engine sends back', async () => {
    capturePatch(200, storedRecord)

    const result = await updateCamera('avenue_01', { label: 'Avenue entrance' })

    expect(result.label).toBe('Avenue entrance')
    expect(result.persisted).toBe(true)
  })

  it('exposes the engine detail separately from the prefixed message', async () => {
    capturePatch(409, 'cameras.json no longer holds avenue_01')

    const error = await updateCamera('avenue_01', { label: 'x' }).catch((e: unknown) => e)

    expect(error).toBeInstanceOf(EngineHttpError)
    expect((error as EngineHttpError).status).toBe(409)
    expect((error as EngineHttpError).detail).toBe('cameras.json no longer holds avenue_01')
    expect((error as EngineHttpError).message).toContain('409')
  })

  it('percent-encodes a camera id so a slash cannot escape the path segment', async () => {
    let seenUrl = ''
    server.use(
      http.patch(`${ENGINE_BASE_URL}/cameras/*`, ({ request }) => {
        seenUrl = request.url
        return HttpResponse.json(storedRecord)
      }),
    )

    await updateCamera('wing a/cam 1', { label: 'x' })

    // Asserted on the raw URL, not on a route param: MSW decodes params, which
    // would hide exactly the bug this guards against.
    expect(seenUrl).toContain('wing%20a%2Fcam%201')
    expect(seenUrl).not.toContain('wing a/cam 1')
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/api/__tests__/engineClient.test.ts`
Expected: FAIL — `updateCamera` is not exported from `@/api/engineClient`.

- [ ] **Step 3: Write minimal implementation**

In `web/src/api/engineClient.ts`, add these type exports beside the existing ones:

```ts
export type CameraEditRequest = components['schemas']['CameraEditRequest']
export type CameraEditResponse = components['schemas']['CameraEditResponse']
export type Zone = components['schemas']['Zone']
export type ZoneKind = components['schemas']['ZoneKind']
```

Give `EngineHttpError` a `detail` field. Replace the existing class with:

```ts
/** The engine answered with a non-2xx status. */
export class EngineHttpError extends Error {
  readonly status: number
  /**
   * The engine's own `detail` string, unprefixed. `message` is for a log line;
   * this is for showing an operator, because the engine's write endpoints
   * explain themselves at length and re-wording that in the console would only
   * make the two disagree.
   */
  readonly detail: string
  constructor(status: number, detail: string) {
    super(`AI engine returned ${status}: ${detail}`)
    this.name = 'EngineHttpError'
    this.status = status
    this.detail = detail
  }
}
```

Add the call at the end of the file:

```ts
/**
 * Edit a camera's `label` and/or `zone`, persisted to the engine's
 * `cameras.json` before the response is sent.
 *
 * Sends `edit` verbatim, because *which keys are present* is the instruction:
 * an omitted `zone` leaves the grouping alone and an explicit `zone: null`
 * ungroups the camera. Build the body with `buildCameraEdit` rather than
 * assembling it at the call site.
 */
export function updateCamera(
  cameraId: string,
  edit: CameraEditRequest,
): Promise<CameraEditResponse> {
  return request<CameraEditResponse>(`/cameras/${encodeURIComponent(cameraId)}`, {
    method: 'PATCH',
    // `request` sets only `Accept`. Every other call is a GET or a bodyless
    // POST, so nothing has needed this before; a PATCH carrying JSON without
    // it is a 422 from FastAPI.
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(edit),
  })
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/api/__tests__/engineClient.test.ts && npx tsc -b`
Expected: 6 tests PASS, 0 type errors.

- [ ] **Step 5: Commit**

```bash
git add web/src/api/engineClient.ts web/src/api/__tests__/engineClient.test.ts
git commit -m "feat(web): add the camera edit client call"
```

---

### Task 2: The edit body builder and label validation

Pure functions in their own module, so the rule that matters most here — only changed fields go on the wire — is testable without rendering anything.

**Files:**
- Create: `web/src/lib/cameraEdit.ts`
- Test: `web/src/lib/__tests__/cameraEdit.test.ts` (create)

**Interfaces:**
- Consumes: `Zone`, `CameraEditRequest` from Task 1.
- Produces:
  - `interface CameraRecordDraft { label: string; zone: Zone | null }`
  - `buildCameraEdit(stored: CameraRecordDraft, draft: CameraRecordDraft): CameraEditRequest`
  - `isEmptyEdit(edit: CameraEditRequest): boolean`
  - `labelError(label: string): string | null`
  - `MAX_LABEL_LENGTH: 120`
  - `ZONES: readonly Zone[]`

- [ ] **Step 1: Write the failing test**

Create `web/src/lib/__tests__/cameraEdit.test.ts`:

```ts
import { describe, expect, it } from 'vitest'
import {
  MAX_LABEL_LENGTH,
  ZONES,
  buildCameraEdit,
  isEmptyEdit,
  labelError,
} from '@/lib/cameraEdit'

const stored = { label: 'Avenue entrance', zone: 'corridor' as const }

describe('buildCameraEdit', () => {
  it('is empty when nothing changed, which the engine rejects as a 422', () => {
    expect(buildCameraEdit(stored, { ...stored })).toEqual({})
    expect(isEmptyEdit(buildCameraEdit(stored, { ...stored }))).toBe(true)
  })

  it('carries only the label when only the label changed', () => {
    const edit = buildCameraEdit(stored, { label: 'East door', zone: 'corridor' })
    expect(edit).toEqual({ label: 'East door' })
    expect('zone' in edit).toBe(false)
  })

  it('carries only the zone when only the zone changed', () => {
    const edit = buildCameraEdit(stored, { label: 'Avenue entrance', zone: 'room' })
    expect(edit).toEqual({ zone: 'room' })
    expect('label' in edit).toBe(false)
  })

  it('carries an explicit null zone when ungrouping', () => {
    const edit = buildCameraEdit(stored, { label: 'Avenue entrance', zone: null })
    expect(edit).toEqual({ zone: null })
    expect('zone' in edit).toBe(true)
  })

  it('omits the zone entirely when it did not change, rather than re-asserting it', () => {
    // Re-asserting an unchanged zone is how one operator's open window reverts
    // another operator's regrouping.
    const edit = buildCameraEdit({ label: 'a', zone: null }, { label: 'b', zone: null })
    expect(edit).toEqual({ label: 'b' })
    expect('zone' in edit).toBe(false)
  })

  it('trims the label before comparing, so whitespace alone is not a change', () => {
    expect(buildCameraEdit(stored, { label: '  Avenue entrance  ', zone: 'corridor' })).toEqual({})
  })

  it('sends the trimmed label, not the raw input', () => {
    expect(buildCameraEdit(stored, { label: '  East door  ', zone: 'corridor' })).toEqual({
      label: 'East door',
    })
  })
})

describe('labelError', () => {
  it('accepts an ordinary label', () => {
    expect(labelError('East corridor, door end')).toBeNull()
  })

  it('rejects an empty label', () => {
    expect(labelError('')).toMatch(/empty/i)
  })

  it('rejects a whitespace-only label, which is empty to an operator', () => {
    expect(labelError('   ')).toMatch(/empty/i)
  })

  it(`rejects a label longer than ${MAX_LABEL_LENGTH} characters`, () => {
    expect(labelError('x'.repeat(MAX_LABEL_LENGTH))).toBeNull()
    expect(labelError('x'.repeat(MAX_LABEL_LENGTH + 1))).toMatch(/120/)
  })

  it('measures length after trimming, matching the engine', () => {
    expect(labelError(`  ${'x'.repeat(MAX_LABEL_LENGTH)}  `)).toBeNull()
  })
})

describe('ZONES', () => {
  it('lists every zone the contract defines', () => {
    expect([...ZONES]).toEqual(['room', 'corridor', 'dayroom'])
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/lib/__tests__/cameraEdit.test.ts`
Expected: FAIL — cannot resolve `@/lib/cameraEdit`.

- [ ] **Step 3: Write minimal implementation**

Create `web/src/lib/cameraEdit.ts`:

```ts
import type { CameraEditRequest, Zone } from '@/api/engineClient'

/** The two fields of a camera record this console can change. */
export interface CameraRecordDraft {
  label: string
  zone: Zone | null
}

export const MAX_LABEL_LENGTH = 120

/**
 * Every zone the contract defines, in the order the console offers them.
 *
 * A runtime list is unavoidable — a TypeScript union cannot be enumerated — so
 * `zonesAreExhaustive` below fails the build if the contract gains a zone that
 * nobody added here. Silently offering an operator a short list is how a camera
 * ends up ungrouped because the zone it belonged in was not on the menu.
 */
export const ZONES = ['room', 'corridor', 'dayroom'] as const satisfies readonly Zone[]

type MissingZones = Exclude<Zone, (typeof ZONES)[number]>
const zonesAreExhaustive: MissingZones extends never ? true : never = true
void zonesAreExhaustive

/**
 * The request body for a draft, carrying **only** what actually changed.
 *
 * Which keys are present is the instruction, not an optimisation. On this
 * endpoint an omitted `zone` means "leave the grouping alone" and an explicit
 * `zone: null` means "ungroup", so a body that always carried both could not
 * express the difference. Omitting unchanged fields is also what stops one
 * operator's open window from reverting another's edit.
 *
 * An empty result is a request the engine answers 422 — see `isEmptyEdit`.
 */
export function buildCameraEdit(
  stored: CameraRecordDraft,
  draft: CameraRecordDraft,
): CameraEditRequest {
  const edit: CameraEditRequest = {}
  const label = draft.label.trim()
  if (label !== stored.label) {
    edit.label = label
  }
  if (draft.zone !== stored.zone) {
    edit.zone = draft.zone
  }
  return edit
}

export function isEmptyEdit(edit: CameraEditRequest): boolean {
  return Object.keys(edit).length === 0
}

/**
 * Mirrors the engine's `CameraLabel` constraint: trim first, then require
 * non-empty and bounded. Trimming first is what makes `""` and `"   "` the same
 * answer — both are empty to an operator, only one is falsy to JavaScript.
 *
 * Returns the message to show, or null when the label is acceptable.
 */
export function labelError(label: string): string | null {
  const trimmed = label.trim()
  if (trimmed.length === 0) {
    return 'A camera label cannot be empty.'
  }
  if (trimmed.length > MAX_LABEL_LENGTH) {
    return `A camera label can be at most ${MAX_LABEL_LENGTH} characters (this one is ${trimmed.length}).`
  }
  return null
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/lib/__tests__/cameraEdit.test.ts && npx tsc -b`
Expected: 13 tests PASS, 0 type errors.

- [ ] **Step 5: Commit**

```bash
git add web/src/lib/cameraEdit.ts web/src/lib/__tests__/cameraEdit.test.ts
git commit -m "feat(web): build camera edit bodies from only what changed"
```

---

### Task 3: The `useUpdateCamera` mutation

**Files:**
- Modify: `web/src/api/queries.ts`
- Test: covered by Tasks 5 and 6 through the panel; no separate test file.

**Interfaces:**
- Consumes: `updateCamera`, `CameraEditRequest`, `CameraEditResponse`, `EngineHttpError` from Task 1.
- Produces: `useUpdateCamera(cameraId: string)` — a TanStack mutation whose `mutate` takes a `CameraEditRequest` and whose `data` is a `CameraEditResponse`.

- [ ] **Step 1: Write the implementation**

There is no separate test here on purpose: a mutation hook with no component around it can only be tested by building a harness that duplicates the panel. Tasks 5 and 6 exercise every branch of it through the real panel. Do not write a `renderHook` test.

In `web/src/api/queries.ts`, extend the import from `@/api/engineClient`:

```ts
import {
  describeCameraNow,
  getCameraEvents,
  getCameraTelemetry,
  getHealth,
  listCameras,
  updateCamera,
  EngineHttpError,
  type CameraEditRequest,
  type CameraEventsResponse,
  type CameraStatus,
  type CamerasResponse,
  type HealthResponse,
} from '@/api/engineClient'
```

Add at the end of the file:

```ts
/**
 * Edit a camera's stored record.
 *
 * Invalidating `['engine', 'cameras']` also refreshes that camera's telemetry:
 * TanStack matches query keys by prefix, and the telemetry key is
 * `['engine', 'cameras', id, 'telemetry']`. One call covers both.
 *
 * A 403 invalidates too. It means the engine's `config_writable` disagrees with
 * what this console last read — writes were turned off underneath it — and
 * refetching is what flips the panel to read-only instead of leaving an
 * operator retrying a control that cannot work.
 */
export function useUpdateCamera(cameraId: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (edit: CameraEditRequest) => updateCamera(cameraId, edit),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['engine', 'cameras'] })
    },
    onError: (error: Error) => {
      if (error instanceof EngineHttpError && error.status === 403) {
        void queryClient.invalidateQueries({ queryKey: ['engine', 'cameras'] })
      }
    },
  })
}
```

- [ ] **Step 2: Verify it compiles**

Run: `npx tsc -b && npx oxlint`
Expected: 0 type errors; no new lint warnings.

- [ ] **Step 3: Commit**

```bash
git add web/src/api/queries.ts
git commit -m "feat(web): add the camera edit mutation"
```

---

### Task 4: `CameraRecordPanel` — the read-only state

The default state on every deployment, so it is built first and on its own.

**Files:**
- Create: `web/src/routes/camera/CameraRecordPanel.tsx`
- Test: `web/src/routes/camera/__tests__/CameraRecordPanel.test.tsx` (create)

**Interfaces:**
- Consumes: `Zone`, `ZoneKind` from Task 1; `ZONES` from Task 2; `Panel`, `Notice`, `KvList` from `@/components/ui/*`; `humanizeEnum` from `@/lib/format`.
- Produces:
  - `interface CameraRecordPanelProps { cameraId: string; record: StoredCameraRecord; writable: boolean }`
  - `interface StoredCameraRecord { label: string; zone: Zone | null; zone_kind: ZoneKind | null }`
  - `DEFAULT_RESTART_REQUIRED_FIELDS: readonly string[]`

**Note — a clarification of the spec.** §3.4 says the Source and Profile rows are driven off `restart_required_fields` from the response. That field only exists on a `CameraEditResponse`, so before any successful edit there is nothing to read. The panel therefore falls back to `DEFAULT_RESTART_REQUIRED_FIELDS` (`['url', 'profile']`) and switches to the response's list once one arrives. The intent of the spec is preserved: the engine's answer wins whenever the engine has given one.

- [ ] **Step 1: Write the failing test**

Create `web/src/routes/camera/__tests__/CameraRecordPanel.test.tsx`:

```tsx
import { describe, expect, it } from 'vitest'
import { screen } from '@testing-library/react'
import { renderWithProviders } from '@/test/renderWithProviders'
import { CameraRecordPanel } from '@/routes/camera/CameraRecordPanel'

const record = { label: 'Avenue entrance', zone: 'corridor' as const, zone_kind: 'common_area' as const }

describe('CameraRecordPanel, read-only', () => {
  it('shows the stored record even when writes are disabled', () => {
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={false} />)

    expect(screen.getByText('Avenue entrance')).toBeInTheDocument()
    expect(screen.getByText(/corridor/i)).toBeInTheDocument()
  })

  it('offers no editing controls at all, rather than disabled ones', () => {
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={false} />)

    expect(screen.queryByLabelText(/label/i)).not.toBeInTheDocument()
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /save/i })).not.toBeInTheDocument()
  })

  it('names the flag and the way to change the record without it', () => {
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={false} />)

    const notice = screen.getByRole('status')
    expect(notice).toHaveTextContent('SENTINEL_ENABLE_CAMERA_WRITES')
    expect(notice).toHaveTextContent(/cameras\.json/)
    expect(notice).toHaveTextContent(/restart/i)
  })

  it('says an ungrouped camera is ungrouped rather than showing an empty row', () => {
    renderWithProviders(
      <CameraRecordPanel
        cameraId="avenue_01"
        record={{ label: 'Loose camera', zone: null, zone_kind: null }}
        writable={false}
      />,
    )

    expect(screen.getByText(/ungrouped/i)).toBeInTheDocument()
  })

  it('names url and profile as restart-required and shows no value for either', () => {
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={false} />)

    // The engine returns neither, because an RTSP URL carries credentials.
    expect(screen.getByText(/^url$/i)).toBeInTheDocument()
    expect(screen.getByText(/^profile$/i)).toBeInTheDocument()
    expect(screen.getAllByText(/restart required/i).length).toBeGreaterThanOrEqual(2)
    expect(screen.queryByText(/rtsp:/i)).not.toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/routes/camera/__tests__/CameraRecordPanel.test.tsx`
Expected: FAIL — cannot resolve `@/routes/camera/CameraRecordPanel`.

- [ ] **Step 3: Write minimal implementation**

Create `web/src/routes/camera/CameraRecordPanel.tsx`:

```tsx
import type { Zone, ZoneKind } from '@/api/engineClient'
import { Panel } from '@/components/ui/Panel'
import { Notice } from '@/components/ui/Notice'
import { KvList, type KvRow } from '@/components/ui/Kv'
import { humanizeEnum } from '@/lib/format'

/**
 * Fields of a camera record the engine will not change at runtime. Used until a
 * successful edit returns the engine's own `restart_required_fields`, which then
 * wins — the engine is the authority on what it refuses to edit.
 */
export const DEFAULT_RESTART_REQUIRED_FIELDS = ['url', 'profile'] as const

export interface StoredCameraRecord {
  label: string
  zone: Zone | null
  zone_kind: ZoneKind | null
}

export interface CameraRecordPanelProps {
  cameraId: string
  record: StoredCameraRecord
  /**
   * The engine's `config_writable`. A **display hint only** — it hides controls
   * that would 403. The engine remains the authority, which is why the save
   * path handles a 403 as a real outcome rather than an impossible one.
   */
  writable: boolean
}

export function describeZone(zone: Zone | null, zoneKind: ZoneKind | null): string {
  if (zone === null) return 'Ungrouped'
  return zoneKind === null ? humanizeEnum(zone) : `${humanizeEnum(zone)} · ${humanizeEnum(zoneKind)}`
}

function restartRequiredRows(fields: readonly string[]): KvRow[] {
  return fields.map((field) => ({
    key: field,
    label: field,
    // No value, deliberately: no endpoint returns one. An RTSP URL routinely
    // carries credentials, so the engine moves it in neither direction.
    value: 'restart required',
  }))
}

// `writable` is declared on the props interface but deliberately not destructured
// here: this task builds only the read-only rendering, and Task 5 adds the branch
// that reads the flag. An unused destructured name would fail lint; an unused
// interface field is fine.
export function CameraRecordPanel({ cameraId, record }: CameraRecordPanelProps) {
  return (
    <Panel data-testid="camera-record-panel">
      <h2 className="mb-2">Camera record</h2>
      <p className="lede">
        What <code>cameras.json</code> holds for this camera, and what the engine is using now.
      </p>

      <Notice tone="inert" className="mb-3">
        This engine is read-only for camera configuration. Camera writes are disabled
        (<code>SENTINEL_ENABLE_CAMERA_WRITES</code> is not set), so this record can only be
        changed by editing <code>cameras.json</code> and restarting the engine.
      </Notice>

      <KvList
        rows={[
          { key: 'label', label: 'Label', value: record.label },
          { key: 'zone', label: 'Zone', value: describeZone(record.zone, record.zone_kind) },
          ...restartRequiredRows(DEFAULT_RESTART_REQUIRED_FIELDS),
        ]}
      />
      <p className="muted mt-2">
        Camera id <code>{cameraId}</code>.
      </p>
    </Panel>
  )
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/routes/camera/__tests__/CameraRecordPanel.test.tsx && npx tsc -b`
Expected: 5 tests PASS, 0 type errors.

- [ ] **Step 5: Commit**

```bash
git add web/src/routes/camera/
git commit -m "feat(web): show a camera's stored record, read-only"
```

---

### Task 5: `CameraRecordPanel` — editing and saving

**Files:**
- Modify: `web/src/routes/camera/CameraRecordPanel.tsx`
- Modify: `web/src/routes/camera/__tests__/CameraRecordPanel.test.tsx`

**Interfaces:**
- Consumes: `useUpdateCamera` (Task 3); `buildCameraEdit`, `labelError`, `isEmptyEdit`, `ZONES`, `CameraRecordDraft` (Task 2); `Input`, `Select`, `Button`, `Label` from `@/components/ui/*`.
- Produces: no new exports; the panel now renders a form when `writable` is true.

- [ ] **Step 1: Write the failing test**

Append to `web/src/routes/camera/__tests__/CameraRecordPanel.test.tsx`. Add these imports at the top of the file:

```tsx
import { beforeEach, vi } from 'vitest'
import userEvent from '@testing-library/user-event'
import * as engineClient from '@/api/engineClient'

vi.mock('@/api/engineClient', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/engineClient')>()
  return { ...actual, updateCamera: vi.fn() }
})

const updateCamera = vi.mocked(engineClient.updateCamera)

const storedResponse = {
  camera_id: 'avenue_01',
  label: 'East door',
  zone: 'corridor' as const,
  zone_kind: 'common_area' as const,
  persisted: true as const,
  restart_required_fields: ['url', 'profile'],
}
```

Then append this block:

```tsx
describe('CameraRecordPanel, editing', () => {
  beforeEach(() => {
    updateCamera.mockReset()
  })

  it('offers a label input and a zone select when writes are enabled', () => {
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    expect(screen.getByLabelText(/label/i)).toHaveValue('Avenue entrance')
    expect(screen.getByLabelText(/zone/i)).toHaveValue('corridor')
    expect(screen.getByRole('button', { name: /save/i })).toBeInTheDocument()
  })

  it('disables save until something actually changes, since an empty edit is a 422', async () => {
    const user = userEvent.setup()
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    expect(screen.getByRole('button', { name: /save/i })).toBeDisabled()

    await user.type(screen.getByLabelText(/label/i), '!')

    expect(screen.getByRole('button', { name: /save/i })).toBeEnabled()
  })

  it('sends only the changed field', async () => {
    updateCamera.mockResolvedValue(storedResponse)
    const user = userEvent.setup()
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    const input = screen.getByLabelText(/label/i)
    await user.clear(input)
    await user.type(input, 'East door')
    await user.click(screen.getByRole('button', { name: /save/i }))

    expect(updateCamera).toHaveBeenCalledWith('avenue_01', { label: 'East door' })
  })

  it('sends an explicit null zone when ungrouping', async () => {
    updateCamera.mockResolvedValue({ ...storedResponse, label: 'Avenue entrance', zone: null, zone_kind: null })
    const user = userEvent.setup()
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    await user.selectOptions(screen.getByLabelText(/zone/i), '')
    await user.click(screen.getByRole('button', { name: /save/i }))

    expect(updateCamera).toHaveBeenCalledWith('avenue_01', { zone: null })
  })

  it('refuses an empty label without calling the engine', async () => {
    const user = userEvent.setup()
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    await user.clear(screen.getByLabelText(/label/i))

    expect(screen.getByText(/cannot be empty/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /save/i })).toBeDisabled()
    expect(updateCamera).not.toHaveBeenCalled()
  })

  it('confirms the save by reporting what was stored and that it survives a restart', async () => {
    updateCamera.mockResolvedValue(storedResponse)
    const user = userEvent.setup()
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    const input = screen.getByLabelText(/label/i)
    await user.clear(input)
    await user.type(input, 'East door')
    await user.click(screen.getByRole('button', { name: /save/i }))

    expect(await screen.findByTestId('camera-record-feedback')).toHaveTextContent(/saved/i)
    expect(screen.getByTestId('camera-record-feedback')).toHaveTextContent(/cameras\.json/)
  })

  it('does not clobber a field the operator is editing when the record refreshes', async () => {
    const user = userEvent.setup()
    const { rerender } = renderWithProviders(
      <CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />,
    )

    const input = screen.getByLabelText(/label/i)
    await user.clear(input)
    await user.type(input, 'Half-typed name')

    // A 5s poll lands mid-edit carrying a change someone else made.
    rerender(
      <CameraRecordPanel
        cameraId="avenue_01"
        record={{ ...record, label: 'Renamed by someone else' }}
        writable={true}
      />,
    )

    expect(screen.getByLabelText(/label/i)).toHaveValue('Half-typed name')
  })

  it('does track the record while the form is untouched', () => {
    const { rerender } = renderWithProviders(
      <CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />,
    )

    rerender(
      <CameraRecordPanel
        cameraId="avenue_01"
        record={{ ...record, label: 'Renamed by someone else' }}
        writable={true}
      />,
    )

    expect(screen.getByLabelText(/label/i)).toHaveValue('Renamed by someone else')
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/routes/camera/__tests__/CameraRecordPanel.test.tsx`
Expected: FAIL — no label input is rendered; `getByLabelText(/label/i)` finds nothing.

- [ ] **Step 3: Write the implementation**

Rewrite `web/src/routes/camera/CameraRecordPanel.tsx`. Keep `DEFAULT_RESTART_REQUIRED_FIELDS`, `StoredCameraRecord`, `CameraRecordPanelProps`, `describeZone` and `restartRequiredRows` exactly as they are, and replace the component:

```tsx
import { useState, type FormEvent } from 'react'
import { useUpdateCamera } from '@/api/queries'
import { Panel } from '@/components/ui/Panel'
import { Notice } from '@/components/ui/Notice'
import { KvList, type KvRow } from '@/components/ui/Kv'
import { Input } from '@/components/ui/Input'
import { Select } from '@/components/ui/Select'
import { Button } from '@/components/ui/Button'
import { Label } from '@/components/ui/Label'
import { humanizeEnum } from '@/lib/format'
import {
  ZONES,
  buildCameraEdit,
  isEmptyEdit,
  labelError,
  type CameraRecordDraft,
} from '@/lib/cameraEdit'
import type { Zone, ZoneKind } from '@/api/engineClient'

/** The `<option>` value standing for "no zone". `null` is not a DOM value. */
const UNGROUPED_OPTION = ''

export function CameraRecordPanel({ cameraId, record, writable }: CameraRecordPanelProps) {
  const mutation = useUpdateCamera(cameraId)

  /**
   * `null` means pristine — the form is showing the server's record and every
   * refresh flows straight through. It becomes a draft the moment the operator
   * touches a field, and from then on refreshes no longer overwrite it.
   */
  const [draft, setDraft] = useState<CameraRecordDraft | null>(null)

  const stored: CameraRecordDraft = { label: record.label, zone: record.zone }
  const shown = draft ?? stored

  function editDraft(patch: Partial<CameraRecordDraft>) {
    setDraft((current) => ({ ...(current ?? stored), ...patch }))
  }

  const invalidLabel = labelError(shown.label)
  const edit = buildCameraEdit(stored, shown)
  const nothingToSave = isEmptyEdit(edit)
  const canSave = draft !== null && !nothingToSave && invalidLabel === null && !mutation.isPending

  function handleSubmit(event: FormEvent) {
    event.preventDefault()
    if (!canSave) return
    mutation.mutate(edit, {
      // Clear the draft only once the write is on disk, so the panel goes back
      // to tracking the server and renders what was stored rather than what was
      // typed. On failure the draft stays, so nothing the operator wrote is lost.
      onSuccess: () => setDraft(null),
    })
  }

  const restartFields = mutation.data?.restart_required_fields ?? DEFAULT_RESTART_REQUIRED_FIELDS

  if (!writable) {
    return (
      <Panel data-testid="camera-record-panel">
        <h2 className="mb-2">Camera record</h2>
        <p className="lede">
          What <code>cameras.json</code> holds for this camera, and what the engine is using now.
        </p>
        <Notice tone="inert" className="mb-3">
          This engine is read-only for camera configuration. Camera writes are disabled
          (<code>SENTINEL_ENABLE_CAMERA_WRITES</code> is not set), so this record can only be
          changed by editing <code>cameras.json</code> and restarting the engine.
        </Notice>
        <KvList
          rows={[
            { key: 'label', label: 'Label', value: record.label },
            { key: 'zone', label: 'Zone', value: describeZone(record.zone, record.zone_kind) },
            ...restartRequiredRows(restartFields),
          ]}
        />
        <p className="muted mt-2">
          Camera id <code>{cameraId}</code>.
        </p>
      </Panel>
    )
  }

  return (
    <Panel data-testid="camera-record-panel">
      <h2 className="mb-2">Camera record</h2>
      <p className="lede">
        Applied to the running camera and written to <code>cameras.json</code>, so an edit
        survives a restart.
      </p>

      <form onSubmit={handleSubmit}>
        <Label htmlFor="camera-record-label">Label</Label>
        <Input
          id="camera-record-label"
          value={shown.label}
          maxLength={200}
          onChange={(event) => editDraft({ label: event.target.value })}
        />
        {invalidLabel !== null ? (
          <p className="err mt-1" role="alert">
            {invalidLabel}
          </p>
        ) : null}

        <Label htmlFor="camera-record-zone">Zone</Label>
        <Select
          id="camera-record-zone"
          value={shown.zone ?? UNGROUPED_OPTION}
          onChange={(event) =>
            editDraft({
              zone: event.target.value === UNGROUPED_OPTION ? null : (event.target.value as Zone),
            })
          }
        >
          <option value={UNGROUPED_OPTION}>Ungrouped</option>
          {ZONES.map((zone) => (
            <option key={zone} value={zone}>
              {humanizeEnum(zone)}
            </option>
          ))}
        </Select>

        <div className="mt-3 flex items-center gap-3">
          <Button type="submit" variant="act" size="small" disabled={!canSave}>
            {mutation.isPending ? 'Saving…' : 'Save changes'}
          </Button>
          {mutation.isSuccess && draft === null ? (
            <span className="ok-text" data-testid="camera-record-feedback">
              Saved to <code>cameras.json</code> — this survives a restart.
            </span>
          ) : null}
        </div>
      </form>

      <div className="mt-4">
        <p className="muted mb-1">Not editable here — edit <code>cameras.json</code> and restart:</p>
        <KvList rows={restartRequiredRows(restartFields)} />
      </div>
      <p className="muted mt-2">
        Camera id <code>{cameraId}</code>.
      </p>
    </Panel>
  )
}
```

Note the `maxLength={200}` on the input rather than 120: the operator must be able to type past the limit and *see* the error, rather than have characters silently swallowed at the boundary. 200 stops a runaway paste.

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/routes/camera/__tests__/CameraRecordPanel.test.tsx && npx tsc -b`
Expected: 13 tests PASS, 0 type errors.

- [ ] **Step 5: Commit**

```bash
git add web/src/routes/camera/
git commit -m "feat(web): edit a camera's label and zone"
```

---

### Task 6: `CameraRecordPanel` — failures and concurrent edits

**Files:**
- Modify: `web/src/routes/camera/CameraRecordPanel.tsx`
- Modify: `web/src/routes/camera/__tests__/CameraRecordPanel.test.tsx`

**Interfaces:**
- Consumes: `EngineHttpError` from Task 1.
- Produces: no new exports.

- [ ] **Step 1: Write the failing test**

Append to the test file (add `EngineHttpError` to the `@/api/engineClient` import used for types — import it as a value: `import { EngineHttpError } from '@/api/engineClient'`):

```tsx
describe('CameraRecordPanel, failures', () => {
  beforeEach(() => {
    updateCamera.mockReset()
  })

  async function editAndSave() {
    const user = userEvent.setup()
    const input = screen.getByLabelText(/label/i)
    await user.clear(input)
    await user.type(input, 'East door')
    await user.click(screen.getByRole('button', { name: /save/i }))
  }

  it('shows the engine\'s own words on a 409 and does not retry', async () => {
    updateCamera.mockRejectedValue(
      new EngineHttpError(409, 'cameras.json no longer contains avenue_01'),
    )
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    await editAndSave()

    const alert = await screen.findByTestId('camera-record-error')
    expect(alert).toHaveTextContent('cameras.json no longer contains avenue_01')
    expect(alert).toHaveTextContent(/nothing was written/i)
    expect(updateCamera).toHaveBeenCalledTimes(1)
  })

  it('keeps the operator\'s text after a failure', async () => {
    updateCamera.mockRejectedValue(new EngineHttpError(409, 'conflict'))
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    await editAndSave()

    await screen.findByTestId('camera-record-error')
    expect(screen.getByLabelText(/label/i)).toHaveValue('East door')
  })

  it('explains a 403 as writes having been turned off', async () => {
    updateCamera.mockRejectedValue(new EngineHttpError(403, 'camera writes are disabled'))
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    await editAndSave()

    expect(await screen.findByTestId('camera-record-error')).toHaveTextContent(
      /disabled on this engine/i,
    )
  })

  it('surfaces a 422 rather than swallowing it, since it means a contract drift', async () => {
    updateCamera.mockRejectedValue(new EngineHttpError(422, 'url is not an editable field'))
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    await editAndSave()

    expect(await screen.findByTestId('camera-record-error')).toHaveTextContent(
      'url is not an editable field',
    )
  })

  it('warns when the record changed elsewhere while an edit was in progress', async () => {
    const user = userEvent.setup()
    const { rerender } = renderWithProviders(
      <CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />,
    )

    await user.type(screen.getByLabelText(/label/i), '!')

    rerender(
      <CameraRecordPanel
        cameraId="avenue_01"
        record={{ ...record, label: 'Renamed by someone else' }}
        writable={true}
      />,
    )

    const warning = await screen.findByTestId('camera-record-stale')
    expect(warning).toHaveTextContent(/changed elsewhere/i)
    expect(warning).toHaveTextContent(/label/i)
    // The operator's text is informed against, never replaced.
    expect(screen.getByLabelText(/label/i)).toHaveValue('Avenue entrance!')
  })

  it('does not warn when nothing changed underneath', async () => {
    const user = userEvent.setup()
    const { rerender } = renderWithProviders(
      <CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />,
    )

    await user.type(screen.getByLabelText(/label/i), '!')
    rerender(<CameraRecordPanel cameraId="avenue_01" record={{ ...record }} writable={true} />)

    expect(screen.queryByTestId('camera-record-stale')).not.toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/routes/camera/__tests__/CameraRecordPanel.test.tsx`
Expected: FAIL — no `camera-record-error` or `camera-record-stale` element exists.

- [ ] **Step 3: Write the implementation**

In `CameraRecordPanel.tsx`, add `useRef` to the React import (Task 5 left it out because nothing used it yet):

```tsx
import { useRef, useState, type FormEvent } from 'react'
```

and import `EngineHttpError` as a **value**, alongside the existing type-only import of `Zone`/`ZoneKind`:

```tsx
import { EngineHttpError } from '@/api/engineClient'
```

Add this helper above the component:

```tsx
/**
 * What to tell an operator about a failed save.
 *
 * Every branch ends in the engine's own `detail`. The engine's write endpoint
 * explains itself at length and in the operator's terms; re-wording it here
 * would only let the console and the engine disagree about what happened.
 */
function saveFailureText(error: Error): string {
  if (!(error instanceof EngineHttpError)) {
    return `${error.message} Nothing was written.`
  }
  switch (error.status) {
    case 403:
      return `Camera writes are disabled on this engine, so nothing was written. ${error.detail}`
    case 404:
      return `This camera is not configured in the running engine. ${error.detail}`
    case 409:
      return `cameras.json cannot take this edit as it currently stands, and nothing was written — reconcile the file and restart rather than overwriting it. ${error.detail}`
    case 422:
      return `The engine rejected this edit as malformed, which means the console and the engine disagree about the contract. ${error.detail}`
    default:
      return `${error.detail} Nothing was written.`
  }
}
```

Track the baseline. Inside the component, after the `draft` state:

```tsx
  /**
   * The record as it stood when this edit began. Compared against the live
   * record to notice another operator's change landing underneath — which is
   * worth saying out loud, and never worth silently applying over what someone
   * is in the middle of typing.
   */
  const baseline = useRef<CameraRecordDraft | null>(null)
```

Change `editDraft` to capture it on the first touch:

```tsx
  function editDraft(patch: Partial<CameraRecordDraft>) {
    setDraft((current) => {
      if (current === null) {
        baseline.current = stored
      }
      return { ...(current ?? stored), ...patch }
    })
  }
```

Clear it on success — change the `onSuccess` in `handleSubmit` to:

```tsx
      onSuccess: () => {
        setDraft(null)
        baseline.current = null
      },
```

Compute the warning, after `canSave`:

```tsx
  const base = baseline.current
  const changedElsewhere =
    draft !== null && base !== null
      ? [
          base.label !== record.label ? 'label' : null,
          base.zone !== record.zone ? 'zone' : null,
        ].filter((field): field is string => field !== null)
      : []
```

Render both, inside the form, immediately before the submit `<div className="mt-3 …">`:

```tsx
        {changedElsewhere.length > 0 ? (
          <Notice tone="caution" className="mt-3" data-testid="camera-record-stale">
            This camera's {changedElsewhere.join(' and ')} changed elsewhere since you started
            editing — another console, or an edit to <code>cameras.json</code>. Your text has been
            left alone. Saving will overwrite the newer value.
          </Notice>
        ) : null}

        {mutation.isError ? (
          <Notice tone="breach" className="mt-3" data-testid="camera-record-error">
            {saveFailureText(mutation.error)}
          </Notice>
        ) : null}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/routes/camera/__tests__/CameraRecordPanel.test.tsx && npx tsc -b && npx oxlint`
Expected: 19 tests PASS, 0 type errors, no new lint warnings.

- [ ] **Step 5: Commit**

```bash
git add web/src/routes/camera/
git commit -m "feat(web): report camera edit failures in the engine's own words"
```

---

### Task 7: Wire the panel into `CameraPage`

**Files:**
- Modify: `web/src/routes/CameraPage.tsx`
- Modify: `web/src/routes/__tests__/CameraPage.test.tsx`

**Interfaces:**
- Consumes: `CameraRecordPanel` (Tasks 4–6), `useCameras` from `@/api/queries`.
- Produces: nothing new.

- [ ] **Step 1: Write the failing test**

In `web/src/routes/__tests__/CameraPage.test.tsx`, add `listCameras: vi.fn()` to the `vi.mock` factory and `const listCameras = vi.mocked(engineClient.listCameras)` beside the other mocked handles. Add `listCameras.mockReset()` to the existing `beforeEach`, then a default in each existing test is not needed — add this helper and give `listCameras` a default in `beforeEach`:

```tsx
function camerasResponse(config_writable: boolean, label = 'avenue_01') {
  return {
    cameras: [{ ...telemetry, label }],
    config_writable,
  }
}
```

In `beforeEach`, after the resets, add:

```tsx
    listCameras.mockResolvedValue(camerasResponse(false))
```

Then append:

```tsx
describe('CameraPage camera record', () => {
  it('shows the record read-only when the engine has writes disabled', async () => {
    getCameraTelemetry.mockResolvedValue(telemetry)
    getCameraEvents.mockResolvedValue(makeEventsResponse())
    listCameras.mockResolvedValue(camerasResponse(false))

    renderCameraPage()

    const panel = await screen.findByTestId('camera-record-panel')
    expect(within(panel).queryByRole('button', { name: /save/i })).not.toBeInTheDocument()
    expect(within(panel).getByText(/SENTINEL_ENABLE_CAMERA_WRITES/)).toBeInTheDocument()
  })

  it('offers the form when the engine reports writes are enabled', async () => {
    getCameraTelemetry.mockResolvedValue(telemetry)
    getCameraEvents.mockResolvedValue(makeEventsResponse())
    listCameras.mockResolvedValue(camerasResponse(true))

    renderCameraPage()

    const panel = await screen.findByTestId('camera-record-panel')
    expect(within(panel).getByLabelText(/label/i)).toBeInTheDocument()
    expect(within(panel).getByRole('button', { name: /save/i })).toBeInTheDocument()
  })

  it('shows no record panel for a camera the list does not have', async () => {
    getCameraTelemetry.mockResolvedValue(telemetry)
    getCameraEvents.mockResolvedValue(makeEventsResponse())
    listCameras.mockResolvedValue({ cameras: [], config_writable: true })

    renderCameraPage()

    await screen.findByText('Frames seen')
    expect(screen.queryByTestId('camera-record-panel')).not.toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/routes/__tests__/CameraPage.test.tsx`
Expected: FAIL — no `camera-record-panel` is rendered.

- [ ] **Step 3: Write the implementation**

In `web/src/routes/CameraPage.tsx`, extend the queries import and add the panel import:

```tsx
import { useCameraEvents, useCameraTelemetry, useCameras, useDescribeCameraNow } from '@/api/queries'
import { CameraRecordPanel } from '@/routes/camera/CameraRecordPanel'
```

Inside `CameraPage`, after `describeMutation`:

```tsx
  /**
   * The record and its writability come from the *same* snapshot on purpose.
   * `label` and `zone` are on the telemetry poll too, but `config_writable` is
   * not, and taking them from two sources would let the panel render a record
   * it is simultaneously wrong about the editability of.
   */
  const camerasQuery = useCameras()
  const cameraRecord = camerasQuery.data?.cameras.find((c) => c.camera_id === cameraId)
```

Render it just before the closing `</div>` of the component, after the notifications `<Panel>`:

```tsx
      {cameraRecord ? (
        <CameraRecordPanel
          cameraId={cameraId}
          record={{
            label: cameraRecord.label,
            zone: cameraRecord.zone ?? null,
            zone_kind: cameraRecord.zone_kind ?? null,
          }}
          writable={camerasQuery.data?.config_writable ?? false}
        />
      ) : null}
```

Add `className="mb-4"` to the existing notifications `<Panel data-testid="notifications-panel">` so it does not sit flush against the new panel.

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/routes/__tests__/CameraPage.test.tsx && npx tsc -b`
Expected: all existing CameraPage tests plus 3 new ones PASS, 0 type errors.

- [ ] **Step 5: Commit**

```bash
git add web/src/routes/CameraPage.tsx web/src/routes/__tests__/CameraPage.test.tsx
git commit -m "feat(web): put the camera record panel on the camera page"
```

---

### Task 8: Make `label` the name the console displays

Without this the write endpoint is unobservable: a rename would take visible effect nowhere.

**Files:**
- Modify: `web/src/routes/CameraPage.tsx`
- Modify: `web/src/routes/DashboardPage.tsx:59`
- Modify: `web/src/sitemap/CameraMarker.tsx`
- Modify: `web/src/sitemap/FloorplanStage.tsx:82`
- Modify: `web/src/routes/__tests__/CameraPage.test.tsx`, `web/src/routes/__tests__/DashboardPage.test.tsx`, `web/src/routes/__tests__/SiteMapPage.test.tsx`

**Interfaces:**
- Consumes: `CameraStatus.label` from the contract.
- Produces: `CameraMarkerProps.displayName: string`.

- [ ] **Step 1: Write the failing test**

In `web/src/routes/__tests__/CameraPage.test.tsx`:

```tsx
  it('shows the camera label as the heading, keeping the id visible for a bug report', async () => {
    getCameraTelemetry.mockResolvedValue({ ...telemetry, label: 'East corridor, door end' })
    getCameraEvents.mockResolvedValue(makeEventsResponse())

    renderCameraPage()

    expect(
      await screen.findByRole('heading', { name: 'East corridor, door end', level: 1 }),
    ).toBeInTheDocument()
    expect(screen.getByTestId('camera-id-subtitle')).toHaveTextContent('avenue_01')
  })
```

In `web/src/routes/__tests__/DashboardPage.test.tsx`:

```tsx
  it('names a tile by its label rather than its id', async () => {
    registerEmptyEventStream()
    getHealth.mockResolvedValue(sampleHealth)
    listCameras.mockResolvedValue({
      cameras: [makeCameraStatus({ camera_id: 'room_4b', label: 'Room 4B, window side' })],
      config_writable: false,
    })

    renderWithProviders(<DashboardPage />)

    expect(await screen.findByText('Room 4B, window side')).toBeInTheDocument()
  })
```

In `web/src/routes/__tests__/SiteMapPage.test.tsx`:

```tsx
  it('names a marker by its label, including in its accessible name', async () => {
    registerEmptyEventStream()
    listCameras.mockResolvedValue({
      cameras: [makeCameraStatus({ camera_id: 'room_4b', label: 'Room 4B', zone: 'room' })],
      config_writable: false,
    })

    renderWithProviders(<SiteMapPage />)

    expect(await screen.findByRole('link', { name: /Room 4B/ })).toBeInTheDocument()
  })
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/routes/__tests__/CameraPage.test.tsx src/routes/__tests__/DashboardPage.test.tsx src/routes/__tests__/SiteMapPage.test.tsx`
Expected: FAIL — all three render the camera id where a label is expected.

- [ ] **Step 3: Write the implementation**

**`CameraPage.tsx`** — replace the heading block:

```tsx
      <div className="mb-1 flex flex-wrap items-center gap-2.5">
        <h1 className="mb-0">{telemetryQuery.data?.label ?? cameraId}</h1>
        <Pill tone={tone}>
          {liveness === 'live' ? 'Live' : liveness === 'stale' ? 'Stale' : 'No data yet'}
        </Pill>
      </div>
      {/* The id stays on the page even once the label is the heading: it is what
          an operator quotes in a bug report and what every log line uses. */}
      <p className="muted mb-2" data-testid="camera-id-subtitle">
        <code>{cameraId}</code>
      </p>
```

**`DashboardPage.tsx:59`** — replace `<h3>{camera.camera_id}</h3>` with:

```tsx
          <h3>{camera.label}</h3>
```

**`CameraMarker.tsx`** — add to `CameraMarkerProps`, directly under `cameraId`:

```tsx
  /**
   * What to call this camera on screen — its `label`, not its id. Named
   * `displayName` rather than `label` because this component already computes a
   * `label` for its accessible name, and two different meanings of the word in
   * one file is how the wrong string reaches a screen reader.
   */
  displayName: string
```

Add `displayName` to the destructured props, then rename the local and use the new prop:

```tsx
  const zoneWords = zone ? humanizeEnum(zone) : 'ungrouped'
  const ariaLabel = `${displayName}, ${zoneWords}, ${toneLabel[tone]}${
    isDefaultPosition ? ', not yet placed' : ''
  }`
```

Replace every remaining use in the file: `aria-label={`${label}. Use arrow keys…`}` becomes `aria-label={`${ariaLabel}. Use arrow keys to move, hold shift to move further.`}`; `aria-label={label}` becomes `aria-label={ariaLabel}`; and both `<span className={labelClass}>{cameraId}</span>` become `<span className={labelClass}>{displayName}</span>`. Leave `labelClass` alone — it is a CSS class name, not a caption.

**`FloorplanStage.tsx:82`** — add the prop beside `cameraId`:

```tsx
            cameraId={camera.camera_id}
            displayName={camera.label}
```

**`FloorplanStage.test.tsx`** — its `makeCamera` already supplies `label: 'cam'`, so no change is needed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run && npx tsc -b && npx oxlint`
Expected: every test file passes, 0 type errors, no new lint warnings.

- [ ] **Step 5: Update the docs and commit**

In `docs/operations.md`, in the "Editing cameras from the console" section, after the paragraph ending "…rather than letting the console overwrite it.", add:

```markdown
In the console, this is the **Camera record** panel on a camera's own page. When
`config_writable` is false the panel still shows the stored record, read-only,
and names the environment variable — hiding it would leave a camera's label and
zone with nowhere in the console they can be read. `url` and `profile` are listed
as restart-required with no value, because no endpoint returns one.
```

In `README.md`, the "Editing cameras from the console" row already claims this works; it is now true. No change needed.

```bash
git add web/src docs/operations.md
git commit -m "feat(web): show a camera's label as its name across the console"
```

---

## Verification

After Task 8, from `web/`:

```bash
npx tsc -b && npx oxlint && npx vitest run
```

Expected: 0 type errors; only the two pre-existing `FootagePage.tsx` `exhaustive-deps` warnings; all test files pass.

From the repo root, confirm the contract and the generated types still agree — the check CI now enforces:

```bash
cd web && npm run gen:api && git diff --exit-code -- src/api/engine.types.ts
```

Expected: no output, exit 0. This task touches no contract, so any diff here means something went wrong.
