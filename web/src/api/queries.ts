import { useMutation, useQuery, useQueryClient, type UseQueryResult } from '@tanstack/react-query'
import {
  describeCameraNow,
  getCameraTelemetry,
  getHealth,
  listCameras,
  type CameraStatus,
  type CamerasResponse,
  type HealthResponse,
} from '@/api/engineClient'

const HEALTH_POLL_MS = 5_000
const CAMERAS_POLL_MS = 5_000
const TELEMETRY_POLL_MS = 2_000

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

export function useDescribeCameraNow(cameraId: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: () => describeCameraNow(cameraId),
    onSuccess: () => {
      void queryClient.invalidateQueries({
        queryKey: ['engine', 'cameras', cameraId, 'telemetry'],
      })
    },
  })
}
