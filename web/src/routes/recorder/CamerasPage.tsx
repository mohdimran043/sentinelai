import { useEffect, useRef, useState } from 'react'
import { PageHeader } from '@/components/ui/PageHeader'
import { Panel } from '@/components/ui/Panel'
import { Reading } from '@/components/ui/Reading'
import { Notice } from '@/components/ui/Notice'
import { Pill } from '@/components/ui/Pill'
import { Button } from '@/components/ui/Button'
import { KvList } from '@/components/ui/Kv'
import { formatCount, humanizeEnum } from '@/lib/format'
import { recorderSnapshotUrl } from '@/recorder/recorderClient'
import { useRecorderCameraAction, useRecorderCameras, useRecorderStatus } from '@/recorder/queries'
import {
  INERT_CAMERA_FIELDS,
  formatFrameAge,
  formatResolution,
  joinCamerasWithStatus,
  type JoinedCamera,
} from '@/recorder/cameraJoin'
import { toneForIntegrityState } from '@/recorder/tone'
import type { RecorderCameraActionVariables } from '@/recorder/queries'
import type { UseMutationResult } from '@tanstack/react-query'

type CameraAction = UseMutationResult<void, Error, RecorderCameraActionVariables>

export function CamerasPage() {
  const camerasQuery = useRecorderCameras()
  const statusQuery = useRecorderStatus()
  const cameraAction = useRecorderCameraAction()

  const mountedAtRef = useRef(Date.now())
  const cacheBust = statusQuery.data?.server_time_ns ?? mountedAtRef.current

  const joined: JoinedCamera[] = camerasQuery.data
    ? joinCamerasWithStatus(camerasQuery.data.cameras, statusQuery.data?.cameras ?? [])
    : []

  return (
    <>
      <PageHeader
        eyebrow="camera registry"
        title="Cameras"
        lede="The recorder's full camera record, joined with live worker state. This describes hardware and a recording pipeline — nothing here detects a person."
      />

      {camerasQuery.isError ? (
        <Notice tone="breach">
          {camerasQuery.error.message} The registry below cannot be shown until the recorder
          answers again.
        </Notice>
      ) : null}

      {statusQuery.isError && !camerasQuery.isError ? (
        <Notice tone="breach" className="mt-3.5">
          {statusQuery.error.message} Worker state (integrity, running, restarts) is unavailable
          — the registry below is shown without it, not replaced with a guess.
        </Notice>
      ) : null}

      {camerasQuery.isPending ? <p className="muted">Loading the camera registry…</p> : null}

      {camerasQuery.data ? (
        <>
          <Notice tone="inert" className="mt-3.5">
            {camerasQuery.data.context_note}
          </Notice>

          <div className="mt-3.5 grid grid-cols-1 gap-3 lg:grid-cols-2">
            {joined.map(({ camera, status }) => (
              <CameraCard
                key={camera.id}
                camera={camera}
                status={status}
                cacheBust={cacheBust}
                cameraAction={cameraAction}
              />
            ))}
          </div>
        </>
      ) : null}
    </>
  )
}

function CameraCard({
  camera,
  status,
  cacheBust,
  cameraAction,
}: {
  camera: JoinedCamera['camera']
  status: JoinedCamera['status']
  cacheBust: number
  cameraAction: CameraAction
}) {
  const tone = status ? toneForIntegrityState(status.integrity_state) : 'inert'
  const isThisRow = cameraAction.variables?.cameraId === camera.id
  const pending = isThisRow && cameraAction.isPending

  return (
    <Panel>
      <div className="mb-2.5 flex flex-wrap items-center justify-between gap-2">
        <h2>{camera.name}</h2>
        {status ? (
          <Pill tone={tone}>{status.integrity_state}</Pill>
        ) : (
          <Pill tone="inert">no status reported</Pill>
        )}
      </div>

      <CameraSnapshot cameraId={camera.id} cameraName={camera.name} cacheBust={cacheBust} />

      <KvList
        rows={[
          { key: 'id', label: 'Camera id', value: camera.id },
          { key: 'mode', label: 'Mode', value: humanizeEnum(camera.mode) },
          { key: 'space', label: 'Space type', value: humanizeEnum(camera.space_type) },
          { key: 'source', label: 'Source', value: camera.source },
          { key: 'resolution', label: 'Resolution', value: formatResolution(camera) },
          {
            key: 'mask',
            label: 'Mask file',
            value: camera.mask_path || 'none configured',
          },
          {
            key: 'preroll',
            label: 'Pre-roll',
            value: `${formatCount(camera.preroll_seconds)} s`,
          },
        ]}
      />

      {status ? (
        <div className="mt-3 grid grid-cols-2 gap-2.5 md:grid-cols-4">
          <Reading label="running" value={status.running ? 'Yes' : 'No'} alarm={!status.running} />
          <Reading label="failed" value={status.failed ? 'Yes' : 'No'} alarm={status.failed} />
          <Reading
            label="restarts"
            value={formatCount(status.restarts)}
            alarm={status.restarts > 0}
          />
          <Reading label="last frame" value={formatFrameAge(status.last_frame_age_ms)} />
          <Reading
            label="frames dropped"
            value={formatCount(status.frames_dropped)}
            alarm={status.frames_dropped > 0}
            className="col-span-2 md:col-span-4"
          />
        </div>
      ) : (
        <Notice tone="caution" className="mt-3">
          The recorder is not reporting worker status for this camera. Start/stop is disabled
          until it does.
        </Notice>
      )}

      <div className="mt-3 rounded-sm border border-line-2 bg-panel-2 p-2.5">
        <div className="eyebrow">stored, read by nothing</div>
        <p className="muted mt-1 mb-2">
          Recorded and validated by the recorder, but acted on by nothing in this slice — see the
          note above.
        </p>
        <KvList
          rows={INERT_CAMERA_FIELDS.map((field) => ({
            key: field,
            label: humanizeEnum(field),
            value: String(camera[field]),
          }))}
        />
      </div>

      {status ? (
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <Button
            variant={status.running ? 'live' : 'act'}
            size="small"
            disabled={cameraAction.isPending}
            onClick={() =>
              cameraAction.mutate({
                cameraId: camera.id,
                action: status.running ? 'stop' : 'start',
              })
            }
          >
            {pending
              ? status.running
                ? 'Stopping…'
                : 'Starting…'
              : status.running
                ? 'Stop camera'
                : 'Start camera'}
          </Button>
          {isThisRow && cameraAction.isSuccess ? (
            <p className="ok-text m-0" data-testid={`camera-action-outcome-${camera.id}`}>
              Sent. Status will refresh.
            </p>
          ) : null}
          {isThisRow && cameraAction.isError ? (
            <p className="err m-0" role="alert" data-testid={`camera-action-outcome-${camera.id}`}>
              {cameraAction.error.message}
            </p>
          ) : null}
        </div>
      ) : null}
    </Panel>
  )
}

function CameraSnapshot({
  cameraId,
  cameraName,
  cacheBust,
}: {
  cameraId: string
  cameraName: string
  cacheBust: number
}) {
  const src = recorderSnapshotUrl(cameraId, cacheBust)
  const [failed, setFailed] = useState(false)

  // A fresh cache-bust means a fresh attempt: a snapshot that failed a moment
  // ago is not evidence the next one will too.
  useEffect(() => {
    setFailed(false)
  }, [src])

  if (failed) {
    return (
      <div className="mb-3 flex aspect-video items-center justify-center rounded-sm border border-dashed border-line-2 bg-panel-2 text-[12px] text-dim">
        Snapshot unavailable
      </div>
    )
  }

  return (
    <img
      src={src}
      alt={`Live snapshot from ${cameraName}`}
      className="mb-3 aspect-video w-full rounded-sm border border-line bg-panel-2 object-cover"
      onError={() => setFailed(true)}
    />
  )
}
