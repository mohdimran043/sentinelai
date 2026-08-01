import { describe, expect, it } from 'vitest'
import { http, HttpResponse } from 'msw'
import { server } from '@/test/mswServer'
import { RECORDER_BASE_URL } from '@/recorder/config'
import {
  RecorderHttpError,
  RecorderUnreachableError,
  getRecorderAlerts,
  getRecorderModels,
} from '@/recorder/recorderClient'
import {
  recorderErrorHandler,
  recorderUnreachableHandler,
} from '@/recorder/mocks/handlers'
import { REAL_ALERTS } from '@/recorder/mocks/fixtures'

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
    expect((failure as Error).message).toBe('The recorder is unreachable.')
  })
})
