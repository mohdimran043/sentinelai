import { afterEach, describe, expect, it } from 'vitest'
import { http, HttpResponse, type JsonBodyType } from 'msw'
import { server } from '@/test/mswServer'
import { ENGINE_BASE_URL } from '@/api/config'
import { EngineHttpError, updateCamera } from '@/api/engineClient'

afterEach(() => server.resetHandlers())

/** Captures the request the client actually put on the wire. */
function capturePatch(status = 200, body: JsonBodyType = null) {
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
