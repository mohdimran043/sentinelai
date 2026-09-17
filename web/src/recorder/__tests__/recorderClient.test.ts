// oxlint-disable no-loss-of-precision -- real recorder nanosecond timestamps; see mocks/fixtures.ts.
import { describe, expect, it } from 'vitest'
import { http, HttpResponse } from 'msw'
import { server } from '@/test/mswServer'
import { RECORDER_BASE_URL } from '@/recorder/config'
import {
  RecorderHttpError,
  RecorderUnreachableError,
  getRecorderAlerts,
  getRecorderJournal,
  getRecorderJournalDay,
  getRecorderModels,
  getRecorderReport,
} from '@/recorder/recorderClient'
import {
  recorderErrorHandler,
  recorderUnreachableHandler,
} from '@/recorder/mocks/handlers'
import {
  REAL_ALERTS,
  REAL_JOURNAL_DAY_HISTORY,
  REAL_JOURNAL_DAY_NO_CHAPTER,
  REAL_JOURNAL_DAY_SEALED,
  REAL_JOURNAL_ROOM_4B,
  REAL_REPORT,
} from '@/recorder/mocks/fixtures'

describe('recorderClient', () => {
  it('parses a real /api/alerts payload into the declared shape', async () => {
    const response = await getRecorderAlerts()

    expect(response.alerts).toHaveLength(REAL_ALERTS.alerts.length)
    const [newest] = response.alerts
    expect(newest?.id).toBe('1785578902552363255-11105')
    expect(newest?.camera_id).toBe('room_4b')
    expect(newest?.event_type).toBe('UNDERLIT')
    expect(newest?.detail).toBe('luma_mean 0.063 below floor 0.08')
    expect(response.dropped.welfare_locked).toBe(0)
  })

  it('parses a real /api/models payload, including the not-ready status block', async () => {
    const response = await getRecorderModels()

    expect(response.status.ready).toBe(false)
    expect(response.status.reason).toBe('no_model_configured')
    expect(response.frontier_credentialed).toBe(false)
    expect(response.local.map((model) => model.name)).toContain('qwen2.5vl:3b')
    expect(response.frontier.map((model) => model.id)).toContain('claude-opus-5')
  })

  it('requests the recorder base URL, never the engine one', async () => {
    let seen = ''
    server.use(
      http.get(`${RECORDER_BASE_URL}/alerts`, ({ request }) => {
        seen = new URL(request.url).pathname
        return HttpResponse.json(REAL_ALERTS)
      }),
    )

    await getRecorderAlerts()

    expect(seen).toBe('/recorder/api/alerts')
    expect(seen).not.toContain('/engine')
  })

  it('raises RecorderHttpError carrying the recorder’s own `error` message', async () => {
    server.use(recorderErrorHandler('/alerts', 404, 'no such endpoint: GET /api/alerts'))

    const failure = await getRecorderAlerts().catch((error: unknown) => error)

    expect(failure).toBeInstanceOf(RecorderHttpError)
    expect((failure as RecorderHttpError).status).toBe(404)
    expect((failure as RecorderHttpError).message).toContain('no such endpoint: GET /api/alerts')
  })

  it('distinguishes "nothing answered" from "answered with an error"', async () => {
    server.use(recorderUnreachableHandler('/alerts'))

    const failure = await getRecorderAlerts().catch((error: unknown) => error)

    expect(failure).toBeInstanceOf(RecorderUnreachableError)
    expect(failure).not.toBeInstanceOf(RecorderHttpError)
    // The address, not just the fact: an operator seeing this needs to know *where*
    // the console looked, because the commonest cause is that the recorder — a
    // separate product from the AI engine — simply is not running there.
    expect((failure as Error).message).toContain('not reachable at')
    expect((failure as Error).message).toContain('separate service from the AI engine')
  })

  it('requests /journal/{camera} and parses the real days list', async () => {
    let seen = ''
    server.use(
      http.get(`${RECORDER_BASE_URL}/journal/:cameraId`, ({ request, params }) => {
        seen = new URL(request.url).pathname
        expect(params.cameraId).toBe('room_4b')
        return HttpResponse.json(REAL_JOURNAL_ROOM_4B)
      }),
    )

    const response = await getRecorderJournal('room_4b')

    expect(seen).toBe('/recorder/api/journal/room_4b')
    expect(response.days).toHaveLength(REAL_JOURNAL_ROOM_4B.days.length)
    expect(response.days[0]).toEqual({
      date: '2026-07-31',
      status: 'sealed',
      revisions: 2,
      sealed_at_ns: 1785540246958127621,
    })
  })

  it('requests /journal/{camera}/{date} with no query string when history is not asked for', async () => {
    let seen = ''
    server.use(
      http.get(`${RECORDER_BASE_URL}/journal/:cameraId/:date`, ({ request }) => {
        seen = request.url
        return HttpResponse.json(REAL_JOURNAL_DAY_SEALED)
      }),
    )

    const response = await getRecorderJournalDay('room_4b', '2026-07-31')

    expect(new URL(seen).search).toBe('')
    expect(response.revision?.status).toBe('sealed')
    expect(response.revision?.supersedes).toBe(1)
  })

  it('adds ?history=1 only when explicitly asked for, and the response carries the history array', async () => {
    let seen = ''
    server.use(
      http.get(`${RECORDER_BASE_URL}/journal/:cameraId/:date`, ({ request }) => {
        seen = request.url
        return HttpResponse.json(REAL_JOURNAL_DAY_HISTORY)
      }),
    )

    const response = await getRecorderJournalDay('room_4b', '2026-07-31', { history: true })

    expect(new URL(seen).search).toBe('?history=1')
    expect(response.history).toHaveLength(2)
    expect(response.history?.[0]?.revision).toBe(1)
    expect(response.history?.[0]?.status).toBe('provisional')
  })

  it('parses a day with no written chapter as detail-only, with no revision key', async () => {
    server.use(
      http.get(`${RECORDER_BASE_URL}/journal/:cameraId/:date`, () =>
        HttpResponse.json(REAL_JOURNAL_DAY_NO_CHAPTER),
      ),
    )

    const response = await getRecorderJournalDay('room_4b', '2020-01-01')

    expect(response.revision).toBeUndefined()
    expect(response.detail).toContain('No chapter has been written')
  })

  it('builds /report?camera=&date= from its two arguments', async () => {
    let seen = ''
    server.use(
      http.get(`${RECORDER_BASE_URL}/report`, ({ request }) => {
        seen = request.url
        return HttpResponse.json(REAL_REPORT)
      }),
    )

    const response = await getRecorderReport('room_4b', '2026-08-01')

    const url = new URL(seen)
    expect(url.searchParams.get('camera')).toBe('room_4b')
    expect(url.searchParams.get('date')).toBe('2026-08-01')
    expect(response.coverage.recorded_pct).toBeCloseTo(79.933, 2)
    expect(response.segments.count).toBe(328)
  })
})
