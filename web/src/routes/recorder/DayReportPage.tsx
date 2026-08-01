import { useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import { PageHeader } from '@/components/ui/PageHeader'
import { Panel } from '@/components/ui/Panel'
import { Reading } from '@/components/ui/Reading'
import { Notice } from '@/components/ui/Notice'
import { Absent } from '@/components/ui/Absent'
import { Pill } from '@/components/ui/Pill'
import { Select } from '@/components/ui/Select'
import { Input } from '@/components/ui/Input'
import { formatCount, formatDurationSeconds, formatPercentValue, humanizeEnum } from '@/lib/format'
import { useRecorderCameras, useRecorderReport } from '@/recorder/queries'
import { toneForIntegrityState } from '@/recorder/tone'
import {
  hasNoCoverageRecord,
  integrityTimeBreakdown,
  summarizeGapCauses,
  summarizeRetrievalHoles,
} from '@/recorder/reportSummary'
import type { RecorderBlindSpots, RecorderCoverageReport } from '@/recorder/recorder.types'

/** `YYYY-MM-DD` in the operator's local zone — never UTC, which can read as tomorrow after evening. */
function todayDateString(): string {
  const now = new Date()
  const pad = (value: number) => String(value).padStart(2, '0')
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`
}

/** Above this many retrieval holes, the cause breakdown alone carries the point — the reference does not render one row per hole either. */
const MAX_HOLES_LISTED = 8

/**
 * The reference's "Day report" screen (rail id `report`, subtitled
 * `coverage.jsonl + segment index` in its own bundle) — `GET
 * /api/report?camera=&date=`. Both `camera` and `date` are required query
 * parameters, not optional filters: the recorder answers 400 without them and
 * 200-with-zeroed-coverage for a well-formed but unknown pair, VERIFIED
 * against `room_4b`/`corridor_1`/an unregistered camera id.
 */
export function DayReportPage() {
  const cameras = useRecorderCameras()
  const cameraList = cameras.data?.cameras ?? []
  const [searchParams, setSearchParams] = useSearchParams()

  const cameraId = searchParams.get('camera') ?? undefined
  const date = searchParams.get('date') ?? undefined

  // Fill in defaults once the camera list arrives, without clobbering a
  // camera/date a link (e.g. from Footage) already put in the URL.
  useEffect(() => {
    if (cameraId !== undefined && date !== undefined) return
    if (cameraId === undefined && cameraList.length === 0) return
    const next = new URLSearchParams(searchParams)
    if (cameraId === undefined) next.set('camera', cameraList[0]?.id ?? '')
    if (date === undefined) next.set('date', todayDateString())
    setSearchParams(next, { replace: true })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameraId, date, cameraList])

  const report = useRecorderReport(cameraId, date)

  function changeCamera(nextCameraId: string) {
    const next = new URLSearchParams(searchParams)
    next.set('camera', nextCameraId)
    setSearchParams(next)
  }

  function changeDate(nextDate: string) {
    if (!nextDate) return
    const next = new URLSearchParams(searchParams)
    next.set('date', nextDate)
    setSearchParams(next)
  }

  return (
    <>
      <PageHeader
        eyebrow="coverage.jsonl + segment index"
        title="Day report"
        lede="What the recorder can account for on one camera on one day: how much of it was actually recorded, what state it was in when it wasn't, and what a privacy mask hides from all of it. This reads the live report, not a cached summary — asking again after retention has run can give a smaller answer about the same day."
      />

      {cameras.isError ? (
        <Notice tone="breach">
          {cameras.error.message} No camera can be picked until the recorder answers again.
        </Notice>
      ) : null}

      <Panel className="mt-3.5">
        <div className="grid gap-3 md:grid-cols-2">
          <div>
            <label htmlFor="report-camera" className="eyebrow mb-1.5 block">
              Camera
            </label>
            <Select
              id="report-camera"
              value={cameraId ?? ''}
              onChange={(event) => changeCamera(event.target.value)}
              disabled={cameraList.length === 0}
            >
              {cameraList.length === 0 ? <option value="">No cameras</option> : null}
              {cameraList.map((camera) => (
                <option key={camera.id} value={camera.id}>
                  {camera.name}
                </option>
              ))}
            </Select>
          </div>
          <div>
            <label htmlFor="report-date" className="eyebrow mb-1.5 block">
              Date
            </label>
            <Input
              id="report-date"
              type="date"
              value={date ?? ''}
              max={todayDateString()}
              onChange={(event) => changeDate(event.target.value)}
            />
          </div>
        </div>
      </Panel>

      {report.isError ? (
        <Notice tone="breach" className="mt-3.5">
          {report.error.message} Nothing below reflects this camera-day until the recorder answers
          again.
        </Notice>
      ) : null}

      {report.isPending && cameraId && date ? (
        <p className="muted mt-3.5">
          Loading the day report for {cameraId} on {date}…
        </p>
      ) : null}

      {report.data ? <ReportBody report={report.data} /> : null}
    </>
  )
}

function ReportBody({ report }: { report: RecorderCoverageReport }) {
  if (hasNoCoverageRecord(report)) {
    return (
      <Absent
        className="mt-3.5"
        title={
          <>
            <Pill tone="inert">no record</Pill>
            {report.camera_id} · {report.date}
          </>
        }
      >
        The recorder has no coverage record at all for this camera on this date — not "nothing
        happened", but nothing to account for. Check the camera id and date; a day the journal has
        never chaptered can still be asked for here, but an unregistered camera or a date with
        nothing behind it looks exactly like this.
      </Absent>
    )
  }

  const fullyLost = report.coverage.expected_seconds > 0 && report.coverage.recorded_pct === 0
  const integrityRows = integrityTimeBreakdown(report.integrity_time)
  const gapCauses = summarizeGapCauses(report.intervals)
  const holes = summarizeRetrievalHoles(report.retrieval_holes)

  return (
    <>
      {fullyLost ? (
        <Notice tone="breach" className="mt-3.5">
          The recorder expected {formatDurationSeconds(report.coverage.expected_seconds)} of
          coverage on this day and recorded none of it. This is a real outage, not an empty report
          — see the integrity breakdown below for why.
        </Notice>
      ) : null}

      <div className="mt-3.5 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Reading
          label="recorded"
          value={formatPercentValue(report.coverage.recorded_pct)}
          alarm={report.coverage.recorded_pct < 90}
        />
        <Reading label="expected" value={formatDurationSeconds(report.coverage.expected_seconds)} />
        <Reading
          label="gap time"
          value={formatDurationSeconds(report.coverage.gap_seconds)}
          alarm={report.coverage.gap_seconds > 0}
        />
        <Reading
          label="segments"
          value={formatCount(report.segments.count)}
          unit={formatDurationSeconds(report.segments.seconds)}
        />
      </div>

      {report.coverage.unexplained_shortfall_secs !== 0 ? (
        <Notice tone="caution" className="mt-3.5">
          The recorder counted{' '}
          {formatDurationSeconds(Math.abs(report.coverage.unexplained_shortfall_secs))}{' '}
          {report.coverage.unexplained_shortfall_secs > 0 ? 'less' : 'more'} coverage than its own
          expected/gap arithmetic predicts. That is the recorder's own accounting gap, not this
          console's — it is surfaced rather than smoothed away.
        </Notice>
      ) : null}

      <Panel className="mt-3.5 p-0">
        <div className="p-[15px] pb-0">
          <div className="eyebrow">integrity, by time spent</div>
        </div>
        {integrityRows.length === 0 ? (
          <p className="muted p-[15px]">
            No integrity state other than healthy recording occurred on this day.
          </p>
        ) : (
          <div className="table-scroll">
            <table className="data-table">
              <caption className="sr-only">Time spent in each integrity state</caption>
              <thead>
                <tr>
                  <th scope="col">State</th>
                  <th scope="col">Time</th>
                </tr>
              </thead>
              <tbody>
                {integrityRows.map((row) => (
                  <tr key={row.state}>
                    <td>
                      <Pill tone={toneForIntegrityState(row.state)}>{row.state}</Pill>
                    </td>
                    <td className="readout">{formatDurationSeconds(row.seconds)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      {gapCauses.length > 0 ? (
        <Panel className="mt-3.5 p-0">
          <div className="p-[15px] pb-0">
            <div className="eyebrow">what caused the gaps</div>
            <p className="muted mt-2">
              {formatCount(report.intervals.length)} recorded intervals carried{' '}
              {formatCount(gapCauses.reduce((sum, cause) => sum + cause.count, 0))} gaps between
              them — grouped by cause below rather than listed one row per interval.
            </p>
          </div>
          <div className="table-scroll mt-2">
            <table className="data-table">
              <caption className="sr-only">Gap causes, worst first</caption>
              <thead>
                <tr>
                  <th scope="col">Cause</th>
                  <th scope="col">Occurrences</th>
                  <th scope="col">Total time</th>
                </tr>
              </thead>
              <tbody>
                {gapCauses.map((cause) => (
                  <tr key={cause.cause}>
                    <td>
                      <Pill tone={toneForIntegrityState(cause.cause)}>{cause.cause}</Pill>
                    </td>
                    <td className="readout">{formatCount(cause.count)}</td>
                    <td className="readout">{formatDurationSeconds(cause.totalMs / 1000)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      ) : null}

      <BlindSpotsPanel blindSpots={report.blind_spots} />

      <Panel className="mt-3.5 p-0">
        <div className="p-[15px] pb-0">
          <div className="eyebrow">retrieval holes</div>
          <p className="muted mt-2">
            {holes.count === 0
              ? 'No retrieval holes on this day — every expected interval has some record behind it.'
              : `${formatCount(holes.count)} gaps in what can be retrieved, totalling ${formatDurationSeconds(holes.totalMs / 1000)}.`}
          </p>
        </div>
        {holes.count > 0 ? (
          <>
            <div className="table-scroll mt-2">
              <table className="data-table">
                <caption className="sr-only">Retrieval holes grouped by cause</caption>
                <thead>
                  <tr>
                    <th scope="col">Cause</th>
                    <th scope="col">Count</th>
                    <th scope="col">Total time</th>
                  </tr>
                </thead>
                <tbody>
                  {holes.byCause.map((cause) => (
                    <tr key={cause.cause}>
                      <td className="readout">{humanizeEnum(cause.cause)}</td>
                      <td className="readout">{formatCount(cause.count)}</td>
                      <td className="readout">{formatDurationSeconds(cause.totalMs / 1000)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {holes.count > MAX_HOLES_LISTED ? (
              <p className="muted p-[15px] pt-2 mb-0">
                Showing causes grouped rather than all {formatCount(holes.count)} individual holes
                — the breakdown above accounts for every one of them.
              </p>
            ) : null}
          </>
        ) : null}
      </Panel>

      <p className="muted mt-3.5">{report.note}</p>
    </>
  )
}

function BlindSpotsPanel({ blindSpots }: { blindSpots: RecorderBlindSpots | null }) {
  return (
    <Panel className="mt-3.5">
      <div className="eyebrow">blind spots</div>
      {blindSpots === null ? (
        <p className="muted mt-2 mb-0">
          No privacy mask is configured for this camera, so nothing is deliberately excluded from
          coverage.
        </p>
      ) : blindSpots.path === '' || blindSpots.regions.length === 0 ? (
        <p className="muted mt-2 mb-0">
          A mask file path is recorded but no regions are defined, so nothing is deliberately
          excluded from coverage.
        </p>
      ) : (
        <>
          <div className="mt-2 grid grid-cols-2 gap-3 md:grid-cols-4">
            <Reading label="masked" value={formatPercentValue(blindSpots.total_pct)} />
            <Reading label="regions" value={formatCount(blindSpots.regions.length)} />
          </div>
          <div className="table-scroll mt-3">
            <table className="data-table">
              <caption className="sr-only">Masked regions</caption>
              <thead>
                <tr>
                  <th scope="col">Region</th>
                  <th scope="col">Share of frame</th>
                  <th scope="col">Box (x1, y1, x2, y2)</th>
                </tr>
              </thead>
              <tbody>
                {blindSpots.regions.map((region) => (
                  <tr key={region.region_id}>
                    <td className="readout">{region.region_id}</td>
                    <td className="readout">{formatPercentValue(region.area_pct)}</td>
                    <td className="readout">{region.bbox.join(', ')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted mt-3 mb-0">{blindSpots.note}</p>
        </>
      )}
    </Panel>
  )
}
