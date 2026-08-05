import { useRef, useState, type FormEvent } from 'react'
import {
  EngineHttpError,
  type ConcernKind,
  type Confidence,
  type Zone,
  type ZoneKind,
} from '@/api/engineClient'
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
  CONCERN_KINDS,
  CONFIDENCE_TIERS,
  DURATION_FIELDS,
  ZONES,
  buildCameraEdit,
  durationError,
  durationText,
  isEmptyEdit,
  labelError,
  sameKinds,
  type CameraRecordDraft,
  type DurationField,
} from '@/lib/cameraEdit'

/** The `<option>` value standing for "no zone". `null` is not a DOM value. */
const UNGROUPED_OPTION = ''

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
  notify_on: readonly ConcernKind[]
  notify_min_confidence: Confidence
  clip_preroll_seconds: number | null
  clip_postroll_seconds: number | null
  summary_interval_seconds: number | null
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

/**
 * An empty list is a stored decision, not an unset field: somebody silenced this
 * camera, and it goes on detecting and recording while telling nobody. Rendering
 * it as a blank row would read as "not configured yet", which is the opposite.
 */
function describeNotifyOn(kinds: readonly ConcernKind[]): string {
  if (kinds.length === 0) return 'Muted — notifies nobody'
  return CONCERN_KINDS.filter((kind) => kinds.includes(kind)).map(humanizeEnum).join(', ')
}

/**
 * Null is the answer to "what is stored", never a report of the effective value.
 * Resolving the default here and showing it as this camera's number is how a
 * console pins a camera to a value nobody chose the next time it writes.
 */
function describeDuration(value: number | null, field: DurationField): string {
  return value === null ? `Follows the ${field.fallback}` : `${value}s`
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

/** The stored record as a draft — the shape the form and the diff both use. */
function draftFrom(record: StoredCameraRecord): CameraRecordDraft {
  return {
    label: record.label,
    zone: record.zone,
    notifyOn: record.notify_on,
    notifyMinConfidence: record.notify_min_confidence,
    clipPrerollSeconds: durationText(record.clip_preroll_seconds),
    clipPostrollSeconds: durationText(record.clip_postroll_seconds),
    summaryIntervalSeconds: durationText(record.summary_interval_seconds),
  }
}

/**
 * Which fields moved underneath an in-progress edit, named as the wire names the
 * engine and `cameras.json` use, so an operator can go and look at the same word.
 */
function fieldsChangedElsewhere(base: CameraRecordDraft, current: CameraRecordDraft): string[] {
  const changed: string[] = []
  if (base.label !== current.label) changed.push('label')
  if (base.zone !== current.zone) changed.push('zone')
  if (!sameKinds(base.notifyOn, current.notifyOn)) changed.push('notify_on')
  if (base.notifyMinConfidence !== current.notifyMinConfidence) {
    changed.push('notify_min_confidence')
  }
  for (const field of DURATION_FIELDS) {
    if (base[field.draftKey] !== current[field.draftKey]) changed.push(field.wireKey)
  }
  return changed
}

/** `a`, `a and b`, `a, b and c` — there are seven editable fields now. */
function listFields(fields: readonly string[]): string {
  if (fields.length <= 1) return fields.join('')
  return `${fields.slice(0, -1).join(', ')} and ${fields[fields.length - 1]}`
}

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

/**
 * The standing caveat on every welfare control here. It is not decoration: these
 * boxes route a vision-language model's opinion about one still frame, and an
 * operator who reads them as a detector's verdict will trust a `likely collapse`
 * more than it has earned and read silence as nothing having happened.
 */
function WelfareNote() {
  return (
    <p className="muted mt-1" data-testid="camera-record-welfare-note">
      These route what a vision-language model says it sees in a single frame — an
      opinion, not a detector's verdict. It has no memory of the frame before, and
      it misses things.
    </p>
  )
}

export function CameraRecordPanel({ cameraId, record, writable }: CameraRecordPanelProps) {
  const mutation = useUpdateCamera(cameraId)

  /**
   * `null` means pristine — the form is showing the server's record and every
   * refresh flows straight through. It becomes a draft the moment the operator
   * touches a field, and from then on refreshes no longer overwrite it.
   */
  const [draft, setDraft] = useState<CameraRecordDraft | null>(null)

  /**
   * The record as it stood the moment editing began — captured once, when
   * `draft` first goes from `null` to non-null, and held fixed until save
   * succeeds (or the draft is otherwise cleared). The submitted edit diffs
   * against THIS, not the live `stored`, so a poll landing mid-edit can
   * change `stored` without that change leaking into the diff: a field the
   * operator never touched still equals its baseline, so `buildCameraEdit`
   * leaves it out of the request rather than "reverting" someone else's
   * concurrent change.
   */
  const baseline = useRef<CameraRecordDraft | null>(null)

  const stored = draftFrom(record)
  const shown = draft ?? stored

  function editDraft(patch: Partial<CameraRecordDraft>) {
    if (draft === null) {
      baseline.current = stored
    }
    setDraft((current) => ({ ...(current ?? stored), ...patch }))
  }

  function editDuration(field: DurationField, value: string) {
    const patch: Partial<CameraRecordDraft> = {}
    patch[field.draftKey] = value
    editDraft(patch)
  }

  function toggleKind(kind: ConcernKind, routed: boolean) {
    const next = new Set(shown.notifyOn)
    if (routed) next.add(kind)
    else next.delete(kind)
    editDraft({ notifyOn: CONCERN_KINDS.filter((candidate) => next.has(candidate)) })
  }

  const invalidLabel = labelError(shown.label)
  const durationErrors = DURATION_FIELDS.map((field) => ({
    field,
    message: durationError(shown[field.draftKey], field),
  }))
  const anyDurationInvalid = durationErrors.some((entry) => entry.message !== null)

  const edit = buildCameraEdit(baseline.current ?? stored, shown)
  const nothingToSave = isEmptyEdit(edit)
  const canSave =
    draft !== null &&
    !nothingToSave &&
    invalidLabel === null &&
    !anyDurationInvalid &&
    !mutation.isPending

  const base = baseline.current
  const changedElsewhere =
    draft !== null && base !== null ? fieldsChangedElsewhere(base, stored) : []

  function handleSubmit(event: FormEvent) {
    event.preventDefault()
    if (!canSave) return
    mutation.mutate(edit, {
      // Clear the draft only once the write is on disk, so the panel goes back
      // to tracking the server and renders what was stored rather than what was
      // typed. On failure the draft stays, so nothing the operator wrote is lost.
      onSuccess: () => {
        setDraft(null)
        baseline.current = null
      },
    })
  }

  const restartFields = mutation.data?.restart_required_fields ?? DEFAULT_RESTART_REQUIRED_FIELDS

  const policyRows: KvRow[] = [
    {
      key: 'notify_on',
      label: 'Notifies on',
      value: (
        <span data-testid="camera-record-notify-on">{describeNotifyOn(record.notify_on)}</span>
      ),
    },
    {
      key: 'notify_min_confidence',
      label: 'Minimum confidence',
      value: (
        <span data-testid="camera-record-min-confidence">
          {humanizeEnum(record.notify_min_confidence)}
        </span>
      ),
    },
    {
      key: 'clip_preroll_seconds',
      label: 'Clip pre-roll',
      value: (
        <span data-testid="camera-record-clip-preroll">
          {describeDuration(record.clip_preroll_seconds, DURATION_FIELDS[0])}
        </span>
      ),
    },
    {
      key: 'clip_postroll_seconds',
      label: 'Clip post-roll',
      value: (
        <span data-testid="camera-record-clip-postroll">
          {describeDuration(record.clip_postroll_seconds, DURATION_FIELDS[1])}
        </span>
      ),
    },
    {
      key: 'summary_interval_seconds',
      label: 'Summary interval',
      value: (
        <span data-testid="camera-record-summary-interval">
          {describeDuration(record.summary_interval_seconds, DURATION_FIELDS[2])}
        </span>
      ),
    },
  ]

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

        <div className="mt-4">
          <p className="muted mb-1">Welfare notifications:</p>
          <KvList rows={policyRows} />
          <WelfareNote />
        </div>

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

        <fieldset className="mt-4 min-w-0 border-0 p-0">
          <legend className="mb-[5px] text-[13px] text-fg">Notifies on</legend>
          <WelfareNote />
          <div className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1">
            {CONCERN_KINDS.map((kind) => (
              <div key={kind} className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id={`camera-record-notify-${kind}`}
                  checked={shown.notifyOn.includes(kind)}
                  onChange={(event) => toggleKind(kind, event.target.checked)}
                />
                <label htmlFor={`camera-record-notify-${kind}`} className="text-[13px] text-fg">
                  {humanizeEnum(kind)}
                </label>
              </div>
            ))}
          </div>
          {shown.notifyOn.length === 0 ? (
            <p className="muted mt-1" data-testid="camera-record-muted-warning">
              Nothing checked means this camera notifies nobody. It keeps detecting, keeps
              recording clips and keeps publishing events — it just stops telling anyone.
            </p>
          ) : null}
        </fieldset>

        <Label htmlFor="camera-record-min-confidence">Minimum confidence</Label>
        <Select
          id="camera-record-min-confidence"
          value={shown.notifyMinConfidence}
          onChange={(event) =>
            editDraft({ notifyMinConfidence: event.target.value as Confidence })
          }
        >
          {CONFIDENCE_TIERS.map((tier) => (
            <option key={tier} value={tier}>
              {humanizeEnum(tier)}
            </option>
          ))}
        </Select>

        {durationErrors.map(({ field, message }) => (
          <div key={field.draftKey}>
            <Label htmlFor={`camera-record-${field.draftKey}`}>{field.label}</Label>
            <Input
              id={`camera-record-${field.draftKey}`}
              type="text"
              inputMode="decimal"
              // Deliberately not `type="number"`: that reports unparseable input
              // as an empty string, so a typo would read as "clear the override"
              // and silently revert the camera to the default.
              value={shown[field.draftKey]}
              placeholder={`Empty — follows the ${field.fallback}`}
              onChange={(event) => editDuration(field, event.target.value)}
            />
            <p className="muted mt-1">{field.hint}</p>
            {message !== null ? (
              <p className="err mt-1" role="alert">
                {message}
              </p>
            ) : null}
          </div>
        ))}

        {changedElsewhere.length > 0 ? (
          <Notice tone="caution" className="mt-3" data-testid="camera-record-stale">
            This camera's {listFields(changedElsewhere)} changed elsewhere since you started
            editing — another console, or an edit to <code>cameras.json</code>. Your text has been
            left alone. Saving sends only the fields you actually changed, so anything you did not
            touch keeps the newer value; a field you did edit will overwrite it.
          </Notice>
        ) : null}

        {mutation.isError ? (
          <Notice tone="breach" className="mt-3" data-testid="camera-record-error">
            {saveFailureText(mutation.error)}
          </Notice>
        ) : null}

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
