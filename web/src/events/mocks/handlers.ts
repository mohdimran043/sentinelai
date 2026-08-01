import { http, HttpResponse } from 'msw'
import { EVENTS_MOCK_PATH } from '@/events/eventClient'
import { generateMockEvents } from '@/events/mocks/data'

/**
 * Stands in for the Go events API (spec §7: RabbitMQ -> Go -> Postgres, Phase
 * 1C). Shaped by `contracts/events/anomaly_event.schema.json` via the generated
 * `SentinelAIAnomalyEvent` type — the same contract the real Go service will
 * have to satisfy, so swapping this out later is a base-URL change in
 * `eventClient.ts`, not a shape change here.
 */
export const eventHandlers = [
  http.get(`${EVENTS_MOCK_PATH}/cameras/:cameraId/events`, ({ params }) => {
    const cameraId = String(params.cameraId)
    const events = generateMockEvents(cameraId)
    return HttpResponse.json({ events })
  }),
]
