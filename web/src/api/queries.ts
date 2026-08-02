import { useMutation, useQuery, useQueryClient, type UseQueryResult } from '@tanstack/react-query'
import {
  describeCameraNow,
  getCameraEvents,
  getCameraTelemetry,
  getHealth,
  listCameras,
  updateCamera,
  EngineHttpError,
  type CameraEditRequest,
  type CameraEventsResponse,
  type CameraStatus,
  type CamerasResponse,
  type HealthResponse,
} from '@/api/engineClient'

const HEALTH_POLL_MS = 5_000
const CAMERAS_POLL_MS = 5_000
const TELEMETRY_POLL_MS = 2_000
/**
 * `CameraProfile`'s token bucket (capacity 2, refill every 10s) plus its
 * cooldown caps a camera at roughly one escalation per 10s sustained — polling
 * this ring faster than that buys nothing but load. 10s keeps the live
 * description and chart current within one escalation cycle.
 */
const EVENTS_POLL_MS = 10_000

/**
 * `retry: 1` (not the default 3) and a short `staleTime: 0`: this dashboard needs
 * to say "the engine is unreachable" quickly, not spend several retries and tens
 * of seconds looking like it is merely loading. `refetchInterval` keeps polling
 * through an error state, which is what lets the UI recover automatically once
 * the engine comes back.
 */
export function useEngineHealth(): UseQueryResult<HealthResponse, Error> {
  return useQuery({
    queryKey: ['engine', 'health'],
    queryFn: getHealth,
    retry: 1,
    refetchInterval: HEALTH_POLL_MS,
    refetchIntervalInBackground: false,
  })
}

export function useCameras(): UseQueryResult<CamerasResponse, Error> {
  return useQuery({
    queryKey: ['engine', 'cameras'],
    queryFn: listCameras,
    retry: 1,
    refetchInterval: CAMERAS_POLL_MS,
    refetchIntervalInBackground: false,
  })
}

export function useCameraTelemetry(cameraId: string): UseQueryResult<CameraStatus, Error> {
  return useQuery({
    queryKey: ['engine', 'cameras', cameraId, 'telemetry'],
    queryFn: () => getCameraTelemetry(cameraId),
    retry: 1,
    refetchInterval: TELEMETRY_POLL_MS,
    refetchIntervalInBackground: false,
    enabled: cameraId.length > 0,
  })
}

/**
 * The camera's volatile event ring (`GET /cameras/{id}/events`): the recent
 * events, the `latest` one lifted out for the live scene panel, and
 * `latest_description_state` (`none` | `available` | `unavailable`) — see
 * `engineClient.ts`'s doc comment for what this is and, just as importantly,
 * is not.
 */
export function useCameraEvents(cameraId: string): UseQueryResult<CameraEventsResponse, Error> {
  return useQuery({
    queryKey: ['engine', 'cameras', cameraId, 'events'],
    queryFn: () => getCameraEvents(cameraId),
    retry: 1,
    refetchInterval: EVENTS_POLL_MS,
    refetchIntervalInBackground: false,
    enabled: cameraId.length > 0,
  })
}

export function useDescribeCameraNow(cameraId: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: () => describeCameraNow(cameraId),
    onSuccess: () => {
      void queryClient.invalidateQueries({
        queryKey: ['engine', 'cameras', cameraId, 'telemetry'],
      })
      void queryClient.invalidateQueries({
        queryKey: ['engine', 'cameras', cameraId, 'events'],
      })
    },
  })
}

/**
 * Edit a camera's stored record.
 *
 * Invalidating `['engine', 'cameras']` also refreshes that camera's telemetry:
 * TanStack matches query keys by prefix, and the telemetry key is
 * `['engine', 'cameras', id, 'telemetry']`. One call covers both.
 *
 * A 403 invalidates too. It means the engine's `config_writable` disagrees with
 * what this console last read — writes were turned off underneath it — and
 * refetching is what flips the panel to read-only instead of leaving an
 * operator retrying a control that cannot work.
 */
export function useUpdateCamera(cameraId: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (edit: CameraEditRequest) => updateCamera(cameraId, edit),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['engine', 'cameras'] })
    },
    onError: (error: Error) => {
      if (error instanceof EngineHttpError && error.status === 403) {
        void queryClient.invalidateQueries({ queryKey: ['engine', 'cameras'] })
      }
    },
  })
}
