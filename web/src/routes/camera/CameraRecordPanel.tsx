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
