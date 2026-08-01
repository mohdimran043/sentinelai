import { useQuery } from '@tanstack/react-query'
import { fetchEventFeed, fetchEventHistory } from '@/events/eventClient'

const FEED_POLL_MS = 10_000

/** Mocked (MSW) — see `eventClient.ts`. Polls to simulate the live feed a real
 * WebSocket hub will eventually push (Phase 1C). */
export function useEventFeed(cameraId: string) {
  return useQuery({
    queryKey: ['events', 'feed', cameraId],
    queryFn: () => fetchEventFeed(cameraId),
    refetchInterval: FEED_POLL_MS,
    enabled: cameraId.length > 0,
  })
}

export function useEventHistory(cameraId: string) {
  return useQuery({
    queryKey: ['events', 'history', cameraId],
    queryFn: () => fetchEventHistory(cameraId),
    enabled: cameraId.length > 0,
  })
}
