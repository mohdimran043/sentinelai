import { Link, useParams } from 'react-router-dom'
import { useCameraEvents, useCameraTelemetry, useCameras, useDescribeCameraNow } from '@/api/queries'
import type { RecentEventEntry } from '@/api/engineClient'
import { HlsPlayer } from '@/live/HlsPlayer'
import { CameraRecordPanel } from '@/routes/camera/CameraRecordPanel'
import { Panel } from '@/components/ui/Panel'
import { Reading } from '@/components/ui/Reading'
import { Pill } from '@/components/ui/Pill'
import { Notice } from '@/components/ui/Notice'
import { Absent } from '@/components/ui/Absent'
import { Button } from '@/components/ui/Button'
import { Ribbon } from '@/components/ui/Ribbon'
import { buildThreatRibbon } from '@/lib/ribbon'
import { panelStateFor, type PanelState } from '@/lib/descriptionPanel'
import { cameraLiveness, toneForLiveness, toneForSeverity } from '@/lib/severity'
import {
  formatAgo,
  formatClockTime,
  formatCount,
  formatThreatScore,
  humanizeEnum,
} from '@/lib/format'

/**
 * What the vision-language model's read of a scene actually is, spelled out
 * where an operator looking at the live panel will see it — not buried in a
 * tooltip. It is a judgement about one sampled keyframe taken when an
 * escalation fires, not continuous detection, and it has no pose or action
 * model behind it (YOLO reports that a `person` box exists, never what that
 * person is doing). Measured on real hardware ahead of this slice: grappling
 * moved 0.3 -> 0.9 and a person lying motionless moved 0.2 -> 0.6, but a
 * stretcher carry was missed entirely. Read it as a hint worth a look, never
 * as a verdict.
 */
const VLM_CAVEAT =
  'A vision-language model’s read of one still frame, taken when an escalation fires — ' +
  'sampled judgement, not continuous detection, and not backed by any pose or action model. ' +
  'A stretcher carry was missed entirely in testing. Treat this as a hint worth a look, never as a verdict.'

function DescriptionEvent({
  event,
  degraded,
}: {
  event: RecentEventEntry
  degraded: boolean
}) {
  const severityTone = toneForSeverity(event.severity)
  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <Pill tone={severityTone}>{event.severity}</Pill>
        <span className="readout text-[12px] text-dim">
          {formatClockTime(event.occurred_at)} · {formatAgo(event.occurred_at)}
        </span>
      </div>
      {degraded ? (
        <Notice tone="caution" className="mb-2">
          The vision model could not answer for this escalation. What follows is derived from
          event metadata, not a description of the scene.
        </Notice>
      ) : null}
      <p className="m-0 text-[13px] text-fg">{event.description}</p>
      <p className="muted mt-2">
        Threat {formatThreatScore(event.threat_score)} · Suggested: {event.suggested_action}
        {event.labels.length > 0 ? ` · ${event.labels.join(', ')}` : ''}
      </p>
    </div>
  )
}

/**
 * The exhaustive switch `PanelState`'s own doc comment refers to. Adding the
 * fourth state described there is: one more member on `PanelState`, one more
 * `case` here.
 */
function renderPanelState(state: PanelState, cameraId: string) {
  switch (state.kind) {
    case 'none':
      return (
        <Absent title="No description yet">
          Nothing has escalated on <code>{cameraId}</code> in this process yet — this is an
          invitation, not an error. The panel fills in the moment the camera's first escalation
          is described.
        </Absent>
      )
    case 'available':
      return <DescriptionEvent event={state.event} degraded={false} />
    case 'unavailable':
      return <DescriptionEvent event={state.event} degraded={true} />
  }
}

function NotificationRow({ event }: { event: RecentEventEntry }) {
  const tone = toneForSeverity(event.severity)
  return (
    <li className="border-b border-line py-3 last:border-b-0">
      <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Pill tone={tone}>{event.severity}</Pill>
          <span className="text-[13px] text-fg">{humanizeEnum(event.reason)}</span>
        </div>
        <span className="readout text-[12px] text-dim">{formatClockTime(event.occurred_at)}</span>
      </div>
      {event.description_unavailable ? (
        <p className="muted italic">
          Description unavailable — the vision model could not answer; this line is a
          metadata-derived stand-in, not a description of the scene.
        </p>
      ) : (
        <p className="m-0 text-[13px] text-fg">{event.description}</p>
      )}
      <p className="muted mt-1">
        Suggested: {event.suggested_action} · threat {formatThreatScore(event.threat_score)}
        {event.labels.length > 0 ? ` · ${event.labels.join(', ')}` : ''}
      </p>
      {/* RecentEventEntry never carries clip_uri: `event_history.py`'s module
          docstring explains why — the durable event's clip is attached after
          this console projection is written, and a permanently-null field
          would be worse than omitting it entirely. This volatile ring cannot
          link to clips; the durable event store (Phase 1C) can. If a future
          endpoint attaches one here, this is the one place to add
          `<a href={clipHref}>Clip</a>`. */}
    </li>
  )
}

export function CameraPage() {
  const { cameraId = '' } = useParams<{ cameraId: string }>()
  const telemetryQuery = useCameraTelemetry(cameraId)
  const eventsQuery = useCameraEvents(cameraId)
  const describeMutation = useDescribeCameraNow(cameraId)

  /**
   * The record and its writability come from the *same* snapshot on purpose.
   * `label` and `zone` are on the telemetry poll too, but `config_writable` is
   * not, and taking them from two sources would let the panel render a record
   * it is simultaneously wrong about the editability of.
   */
  const camerasQuery = useCameras()
  const cameraRecord = camerasQuery.data?.cameras.find((c) => c.camera_id === cameraId)

  const liveness =
    telemetryQuery.status === 'success' ? cameraLiveness(telemetryQuery.data.last_frame_at) : 'no-data'
  const tone = toneForLiveness(liveness)

  return (
    <div>
      <p className="mb-2">
        <Link to="/dashboard" className="muted hover:text-fg hover:underline">
          ← Dashboard
        </Link>
      </p>
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
      <p className="lede">
        Telemetry, the scene description, the chart and notifications below are all live from
        the AI engine.
      </p>

      {telemetryQuery.isError ? (
        <Notice tone="breach" className="mb-4">
          Telemetry unreachable: {telemetryQuery.error.message}
        </Notice>
      ) : null}

      {/*
        The centre-with-rails composition: the video canvas is the thing an
        operator actually clicked through for, so it gets the wide column;
        the live description sits beside it, close enough to read while
        watching. Everything below (chart, readings, notifications) is
        secondary telemetry and spans the full width underneath.
      */}
      <div className="mb-4 grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1.6fr)_minmax(280px,1fr)]">
        <Panel data-testid="video-panel">
          <h2 className="mb-2">Live video</h2>
          <HlsPlayer cameraId={cameraId} />
        </Panel>

        <Panel data-testid="scene-description-panel">
          <div className="mb-1 flex flex-wrap items-center justify-between gap-3">
            <h2>Live scene description</h2>
            <Button
              variant="act"
              size="small"
              onClick={() => describeMutation.mutate()}
              disabled={describeMutation.isPending}
            >
              {describeMutation.isPending ? 'Describing…' : 'Describe now'}
            </Button>
          </div>
          <p className="muted mb-3">{VLM_CAVEAT}</p>
          {describeMutation.isError ? (
            <p className="err mt-1 mb-2" data-testid="describe-feedback" role="alert">
              {describeMutation.error.message}
            </p>
          ) : null}
          {describeMutation.isSuccess ? (
            <p className="ok-text mt-1 mb-2" data-testid="describe-feedback">
              Requested — event <code>{describeMutation.data.event_id}</code> queued.
            </p>
          ) : null}

          {eventsQuery.isPending ? (
            <p className="muted">Loading…</p>
          ) : eventsQuery.isError ? (
            <Notice tone="breach" className="mt-1">
              Live scene description unreachable: {eventsQuery.error.message}
            </Notice>
          ) : (
            renderPanelState(panelStateFor(eventsQuery.data), cameraId)
          )}
        </Panel>
      </div>

      <Panel className="mb-4" data-testid="chart-panel">
        <h2>Threat over time</h2>
        {eventsQuery.isPending ? (
          <p className="muted mt-3">Loading…</p>
        ) : eventsQuery.isError ? (
          <Absent title="Chart unavailable" className="mt-3">
            The AI engine did not answer: {eventsQuery.error.message}
          </Absent>
        ) : eventsQuery.data.events.length === 0 ? (
          <Absent title="No events yet" className="mt-3">
            Nothing has escalated on this camera in this process yet — the chart will begin
            drawing threat over time the moment it does.
          </Absent>
        ) : (
          <>
            <div className="mt-3 overflow-x-auto">
              <Ribbon
                cells={buildThreatRibbon(eventsQuery.data.events)}
                ariaLabel={`Threat over time for ${cameraId}, oldest to newest, ${eventsQuery.data.events.length} of up to ${eventsQuery.data.capacity} retained events`}
              />
            </div>
            <p className="muted mt-2" data-testid="chart-footnote">
              Showing {formatCount(eventsQuery.data.returned)} of up to{' '}
              {formatCount(eventsQuery.data.capacity)} retained events, oldest to newest.
              Volatile — held in engine memory only and cleared on restart; this is not the
              audit trail.
            </p>
          </>
        )}
        <div className="mt-3 flex flex-wrap gap-4 text-[12px] text-dim">
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-[11px] w-[11px] rounded-[1px] bg-signal" /> nominal
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-[11px] w-[11px] rounded-[1px] bg-caution" /> caution
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-[11px] w-[11px] rounded-[1px] bg-breach" /> breach
          </span>
        </div>
      </Panel>

      <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
        {telemetryQuery.isSuccess ? (
          <>
            <Reading
              label="Last frame"
              value={
                telemetryQuery.data.last_frame_at !== null
                  ? formatAgo(telemetryQuery.data.last_frame_at)
                  : '—'
              }
            />
            <Reading label="Frames seen" value={formatCount(telemetryQuery.data.frames_seen)} />
            <Reading label="Frames dropped" value={formatCount(telemetryQuery.data.frames_dropped)} />
            <Reading label="Detections run" value={formatCount(telemetryQuery.data.detections_run)} />
            <Reading label="Escalations" value={formatCount(telemetryQuery.data.escalations)} />
            <Reading
              label="Escalations dropped"
              value={formatCount(telemetryQuery.data.escalations_dropped)}
              alarm={telemetryQuery.data.escalations_dropped > 0}
            />
            <Reading label="Discontinuities" value={formatCount(telemetryQuery.data.discontinuities)} />
            <Reading
              label="Last escalation"
              value={
                telemetryQuery.data.last_escalation_at !== null
                  ? formatAgo(telemetryQuery.data.last_escalation_at)
                  : 'none yet'
              }
            />
          </>
        ) : (
          <p className="muted col-span-full">
            {telemetryQuery.isPending ? 'Loading telemetry…' : 'Telemetry unavailable.'}
          </p>
        )}
      </div>

      <Panel className="mb-4" data-testid="notifications-panel">
        <h2>Notifications</h2>
        <p className="lede">Recent events for this camera, newest first.</p>
        {eventsQuery.isPending ? (
          <p className="muted">Loading…</p>
        ) : eventsQuery.isError ? (
          <Absent title="Notifications unavailable">{eventsQuery.error.message}</Absent>
        ) : eventsQuery.data.events.length === 0 ? (
          <Absent title="No notifications yet">Nothing has happened on this camera yet.</Absent>
        ) : (
          <ul className="max-h-[46vh] list-none overflow-y-auto p-0">
            {[...eventsQuery.data.events].reverse().map((event) => (
              <NotificationRow key={event.event_id} event={event} />
            ))}
          </ul>
        )}
      </Panel>

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
    </div>
  )
}
