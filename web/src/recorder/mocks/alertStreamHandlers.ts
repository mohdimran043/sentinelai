import { http, HttpResponse } from 'msw'
import { RECORDER_BASE_URL } from '@/recorder/config'

/**
 * MSW handler for `GET /api/alerts/stream` — **tests only**. Kept in its own
 * file for the same reason `maskHandlers.ts`/`maskFixtures.ts` are: several
 * other sections were landing in the shared `recorder/mocks/handlers.ts` at
 * the same time this was written, so this only needs one import and one line
 * there rather than a contested block.
 *
 * `AlertsPage` now opens this stream on every render (see
 * `useRecorderAlertStream`), so every test that renders it — including all
 * the pre-existing filtering/paging tests that know nothing about the stream
 * — needs SOME handler registered, or the connection would fail and start a
 * real background reconnect loop against a route nothing answers. The
 * default below sends an empty backlog and otherwise stays open (mirroring
 * the real recorder, which keeps the connection open indefinitely); tests
 * that care about the stream itself build a controllable one instead — see
 * `alertStream.test.ts` and the "live alert stream" describe block in
 * `AlertsPage.test.tsx`.
 */
export const recorderAlertStreamHandler = (initialBacklogText = 'event: backlog\ndata: []\n\n') =>
  http.get(`${RECORDER_BASE_URL}/alerts/stream`, () => {
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(initialBacklogText))
      },
    })
    return new HttpResponse(stream, { headers: { 'Content-Type': 'text/event-stream' } })
  })

export const alertStreamHandlers = [recorderAlertStreamHandler()]
