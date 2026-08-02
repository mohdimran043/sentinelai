import { useRef, useState, type FormEvent } from 'react'
import type { Zone, ZoneKind } from '@/api/engineClient'
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

  const stored: CameraRecordDraft = { label: record.label, zone: record.zone }
  const shown = draft ?? stored

  function editDraft(patch: Partial<CameraRecordDraft>) {
    if (draft === null) {
      baseline.current = stored
    }
    setDraft((current) => ({ ...(current ?? stored), ...patch }))
  }

  const invalidLabel = labelError(shown.label)
  const edit = buildCameraEdit(baseline.current ?? stored, shown)
  const nothingToSave = isEmptyEdit(edit)
  const canSave = draft !== null && !nothingToSave && invalidLabel === null && !mutation.isPending

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
