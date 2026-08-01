import { useEffect, useState } from 'react'
import type { UseQueryResult } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { PageHeader } from '@/components/ui/PageHeader'
import { Panel } from '@/components/ui/Panel'
import { Reading } from '@/components/ui/Reading'
import { Notice } from '@/components/ui/Notice'
import { Absent } from '@/components/ui/Absent'
import { Pill } from '@/components/ui/Pill'
import { Button } from '@/components/ui/Button'
import { Select } from '@/components/ui/Select'
import { KvList } from '@/components/ui/Kv'
import { formatCount, formatDurationSeconds, formatNanosDateTime, formatPercentValue } from '@/lib/format'
import { useRecorderCameras, useRecorderJournal, useRecorderJournalDay } from '@/recorder/queries'
import { recorderSnapshotUrl } from '@/recorder/recorderClient'
import { toneForJournalStatus } from '@/recorder/tone'
import type {
  RecorderJournalDay,
  RecorderJournalDayResponse,
  RecorderJournalResponse,
} from '@/recorder/recorder.types'

/** How often the snapshot re-requests a frame. Data refresh, not decoration — unaffected by reduced-motion. */
const SNAPSHOT_REFRESH_MS = 8_000

export function FootagePage() {
  const cameras = useRecorderCameras()
  const cameraList = cameras.data?.cameras ?? []
  const [cameraId, setCameraId] = useState<string | undefined>(undefined)
  const [selectedDate, setSelectedDate] = useState<string | undefined>(undefined)
  const [showHistory, setShowHistory] = useState(false)

  // Pick the first camera once the registry arrives, if nothing is picked yet.
  useEffect(() => {
    if (cameraId === undefined && cameraList.length > 0) {
      setCameraId(cameraList[0]?.id)
    }
  }, [cameraId, cameraList])

  const journal = useRecorderJournal(cameraId)
  const days = journal.data?.days ?? []

  // Default to the most recent journal day once it arrives, or when the
  // camera changed and the previous selection no longer applies.
  useEffect(() => {
    if (days.length === 0) return
    if (selectedDate === undefined || !days.some((day) => day.date === selectedDate)) {
      setSelectedDate(days[0]?.date)
    }
  }, [days, selectedDate])

  const dayDetail = useRecorderJournalDay(cameraId, selectedDate, { history: showHistory })

  function changeCamera(nextCameraId: string) {
    setCameraId(nextCameraId)
    setSelectedDate(undefined)
    setShowHistory(false)
  }

  function changeDate(nextDate: string) {
    setSelectedDate(nextDate)
    setShowHistory(false)
  }

  return (
    <>
      <PageHeader
        eyebrow="evidence"
        title="Footage"
        lede="The recorder has no segment or clip index to browse — /api/segments and /api/cameras/{id}/clips both answer 404. This assembles what the recorder does have: the most recent frame it can show, and the sealed daily record with its revision history. There is no clip player here because there is no clip endpoint to play one from."
      />

      {cameras.isError ? (
        <Notice tone="breach">
          {cameras.error.message} No camera can be picked until the recorder answers again.
        </Notice>
      ) : null}

      {cameras.isPending ? <p className="muted">Loading the camera list…</p> : null}

      {cameras.isSuccess && cameraList.length === 0 ? (
        <Absent title="No cameras registered">
          The recorder has no cameras configured, so there is no footage to look at.
        </Absent>
      ) : null}

      {cameraList.length > 0 ? (
        <Panel className="mt-3.5">
          <label htmlFor="footage-camera" className="eyebrow mb-1.5 block">
            Camera
          </label>
          <Select
            id="footage-camera"
            value={cameraId ?? ''}
            onChange={(event) => changeCamera(event.target.value)}
          >
            {cameraList.map((camera) => (
              <option key={camera.id} value={camera.id}>
                {camera.name}
              </option>
            ))}
          </Select>
        </Panel>
      ) : null}

      {cameraId ? (
        <>
          <SnapshotPanel cameraId={cameraId} />
          <JournalPanel
            journalQuery={journal}
            days={days}
            selectedDate={selectedDate}
            onSelectDate={changeDate}
          />
          {selectedDate ? (
            <DayDetailPanel
              cameraId={cameraId}
              date={selectedDate}
              dayQuery={dayDetail}
              showHistory={showHistory}
              onToggleHistory={() => setShowHistory((current) => !current)}
            />
          ) : null}
        </>
      ) : null}
    </>
  )
}

function SnapshotPanel({ cameraId }: { cameraId: string }) {
  const [cacheBust, setCacheBust] = useState<number>(() => Date.now())
  const [failed, setFailed] = useState(false)

  // A new camera means a new image — reset the failure flag and get a fresh frame immediately.
  useEffect(() => {
    setFailed(false)
    setCacheBust(Date.now())
  }, [cameraId])

  useEffect(() => {
    const intervalId = window.setInterval(() => setCacheBust(Date.now()), SNAPSHOT_REFRESH_MS)
    return () => window.clearInterval(intervalId)
  }, [cameraId])

  return (
    <Panel className="mt-3.5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="eyebrow">most recent frame</div>
        <Button variant="quiet" size="small" onClick={() => setCacheBust(Date.now())}>
          Refresh now
        </Button>
      </div>

      {failed ? (
        <Notice tone="breach" className="mt-2">
          The recorder could not produce a frame for {cameraId} just now. That does not by itself
          mean recording has stopped — check the journal below before assuming an outage.
        </Notice>
      ) : (
        <img
          key={cameraId}
          src={recorderSnapshotUrl(cameraId, cacheBust)}
          alt={`Most recent available frame from ${cameraId}`}
          className="mt-2.5 w-full max-w-[640px] rounded-sm border border-line-2"
          onError={() => setFailed(true)}
          onLoad={() => setFailed(false)}
        />
      )}

      <p className="muted mt-2.5 mb-0">
        Requested {formatNanosDateTime(cacheBust * 1_000_000)} — the recorder assembles this from
        recorded segments, not a raw camera tap, and it is not confirmed live. Every request is
        cache-busted, so this is never a stale browser copy; it refreshes automatically every{' '}
        {SNAPSHOT_REFRESH_MS / 1000}s.
      </p>
    </Panel>
  )
}

function JournalPanel({
  journalQuery,
  days,
  selectedDate,
  onSelectDate,
}: {
  journalQuery: UseQueryResult<RecorderJournalResponse, Error>
  days: RecorderJournalDay[]
  selectedDate: string | undefined
  onSelectDate: (date: string) => void
}) {
  return (
    <Panel className="mt-3.5 p-0">
      <div className="p-[15px] pb-0">
        <div className="eyebrow">daily journal</div>
      </div>

      {journalQuery.isError ? (
        <Notice tone="breach" className="m-[15px]">
          {journalQuery.error.message} The list of sealed days below cannot be trusted until the
          recorder answers again.
        </Notice>
      ) : null}

      {journalQuery.isPending ? <p className="muted p-[15px]">Loading the journal…</p> : null}

      {journalQuery.data ? (
        days.length === 0 ? (
          <Absent className="m-[15px]" title="No day has ever been sealed for this camera">
            The journal has no entries at all. That is a fact about this camera's recorder
            history, not about whether it is currently recording.
          </Absent>
        ) : (
          <>
            <p className="muted px-[15px] pb-2">{journalQuery.data.note}</p>
            <div className="table-scroll">
              <table className="data-table">
                <caption className="sr-only">Daily journal, newest first</caption>
                <thead>
                  <tr>
                    <th scope="col">Date</th>
                    <th scope="col">Status</th>
                    <th scope="col">Revisions</th>
                    <th scope="col">Sealed at</th>
                    <th scope="col">
                      <span className="sr-only">Select</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {days.map((day) => (
                    <tr key={day.date} className={day.date === selectedDate ? 'bg-panel-2' : undefined}>
                      <td className="readout whitespace-nowrap">{day.date}</td>
                      <td>
                        <Pill tone={toneForJournalStatus(day.status)}>{day.status}</Pill>
                      </td>
                      <td className="readout">{formatCount(day.revisions)}</td>
                      <td className="readout whitespace-nowrap">
                        {formatNanosDateTime(day.sealed_at_ns)}
                      </td>
                      <td>
                        <Button
                          variant="quiet"
                          size="small"
                          aria-pressed={day.date === selectedDate}
                          onClick={() => onSelectDate(day.date)}
                        >
                          {day.date === selectedDate ? 'Selected' : 'View'}
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )
      ) : null}
    </Panel>
  )
}

function DayDetailPanel({
  cameraId,
  date,
  dayQuery,
  showHistory,
  onToggleHistory,
}: {
  cameraId: string
  date: string
  dayQuery: UseQueryResult<RecorderJournalDayResponse, Error>
  showHistory: boolean
  onToggleHistory: () => void
}) {
  if (dayQuery.isError) {
    return (
      <Notice tone="breach" className="mt-3.5">
        {dayQuery.error.message} This day's record cannot be shown until the recorder answers
        again.
      </Notice>
    )
  }

  if (dayQuery.isPending) {
    return <p className="muted mt-3.5">Loading {date}…</p>
  }

  const data = dayQuery.data
  if (!data) return null

  if (!data.revision) {
    return (
      <Absent
        className="mt-3.5"
        title={
          <>
            <Pill tone="inert">not yet written</Pill>
            {date}
          </>
        }
      >
        {data.detail ?? 'No chapter has been written for this day.'}
      </Absent>
    )
  }

  const revision = data.revision
  const report = revision.report

  return (
    <Panel className="mt-3.5">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <h2 className="flex flex-wrap items-center gap-2.5">
          <Pill tone={toneForJournalStatus(revision.status)}>{revision.status}</Pill>
          {date} · revision {formatCount(revision.revision)}
          {revision.supersedes !== undefined ? (
            <span className="muted text-[12px] font-normal normal-case tracking-normal">
              supersedes revision {formatCount(revision.supersedes)}
            </span>
          ) : null}
        </h2>
        <Button variant="quiet" size="small" onClick={onToggleHistory}>
          {showHistory
            ? 'Hide revision history'
            : `Show revision history${data.history ? ` (${formatCount(data.history.length)})` : ''}`}
        </Button>
      </div>

      <p className="lede mt-2 mb-3">{revision.reason}</p>

      <KvList
        rows={[
          { key: 'written', label: 'Written at', value: formatNanosDateTime(revision.created_at_ns) },
          { key: 'sha256', label: 'Report SHA-256', value: revision.report_sha256 },
        ]}
      />

      <div className="mt-3.5 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Reading
          label="recorded"
          value={formatPercentValue(report.coverage.recorded_pct)}
          alarm={report.coverage.recorded_pct < 90}
        />
        <Reading
          label="segments"
          value={formatCount(report.segments.count)}
          unit={formatDurationSeconds(report.segments.seconds)}
        />
        <Reading
          label="expected"
          value={formatDurationSeconds(report.coverage.expected_seconds)}
        />
        {report.blind_spots ? (
          <Reading label="masked out" value={formatPercentValue(report.blind_spots.total_pct)} />
        ) : (
          <Reading label="masked out" value="No mask" />
        )}
      </div>

      <p className="muted mt-3 mb-0">
        This is provenance for the day's record, not the full coverage picture — the full
        breakdown of gaps, integrity states and blind spots is on the{' '}
        <Link
          to={`/recorder/report?camera=${encodeURIComponent(cameraId)}&date=${encodeURIComponent(date)}`}
          className="text-signal underline-offset-2 hover:underline"
        >
          Day report
        </Link>
        .
      </p>

      {showHistory ? (
        <div className="mt-4">
          <h3 className="mb-2 text-[13px]">Revision history</h3>
          {data.history ? (
            <div className="table-scroll">
              <table className="data-table">
                <caption className="sr-only">Every revision written for {date}</caption>
                <thead>
                  <tr>
                    <th scope="col">Revision</th>
                    <th scope="col">Status</th>
                    <th scope="col">Written at</th>
                    <th scope="col">Reason</th>
                    <th scope="col">SHA-256</th>
                  </tr>
                </thead>
                <tbody>
                  {data.history.map((entry) => (
                    <tr
                      key={entry.revision}
                      className={entry.revision === revision.revision ? 'bg-panel-2' : undefined}
                    >
                      <td className="readout">
                        {formatCount(entry.revision)}
                        {entry.revision === revision.revision ? (
                          <Pill tone="inert" className="ml-1.5">
                            current
                          </Pill>
                        ) : null}
                      </td>
                      <td>
                        <Pill tone={toneForJournalStatus(entry.status)}>{entry.status}</Pill>
                      </td>
                      <td className="readout whitespace-nowrap">
                        {formatNanosDateTime(entry.created_at_ns)}
                      </td>
                      <td className="muted">{entry.reason}</td>
                      <td className="readout text-[11px]">{entry.report_sha256}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="muted">Loading the revision history…</p>
          )}
        </div>
      ) : null}
    </Panel>
  )
}
