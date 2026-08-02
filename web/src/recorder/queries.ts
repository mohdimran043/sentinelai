import {
  useMutation,
  useQuery,
  useQueryClient,
  type UseMutationResult,
  type UseQueryResult,
} from '@tanstack/react-query'
import {
  getRecorderAlerts,
  getRecorderCapabilities,
  getRecorderJournal,
  getRecorderJournalDay,
  getRecorderMask,
  getRecorderModels,
  getRecorderNotifications,
  getRecorderReport,
  getRecorderSettings,
  getRecorderStatus,
  getRecorderUntrackedStorage,
  listRecorderCameras,
  startRecorderCamera,
  stopRecorderCamera,
} from '@/recorder/recorderClient'
import type {
  RecorderAlertsResponse,
  RecorderCamerasResponse,
  RecorderCapabilitiesResponse,
  RecorderCoverageReport,
  RecorderJournalDayResponse,
  RecorderJournalResponse,
  RecorderMaskResponse,
  RecorderModelsResponse,
  RecorderNotificationsResponse,
  RecorderSettingsResponse,
  RecorderStatusResponse,
  RecorderUntrackedResponse,
} from '@/recorder/recorder.types'

/**
 * React Query hooks for the recorder. Every key is namespaced under
 * `['recorder', ...]` so nothing can collide with the AI engine's
 * `['engine', ...]` keys, and so the whole recorder cache can be invalidated
 * independently of the engine's.
 */

/**
 * The recorder's own console falls back to a 15s refresh when its SSE stream is
 * down; the alert feed genuinely moves (the live instance adds one roughly
 * every five seconds), so match that cadence. `refetchInterval` keeps polling
 * through an error state, so the page recovers by itself once the recorder
 * answers again.
 *
 * There is a live `GET /api/alerts/stream` (SSE, verified: emits a `backlog`
 * event then `alert` events) that would remove the poll entirely. Not wired
 * yet — polling is enough for F1 and has one failure mode instead of two.
 */
const ALERTS_POLL_MS = 15_000
const STATUS_POLL_MS = 5_000
/** Model configuration only changes when an operator changes it. Don't hammer it. */
const MODELS_POLL_MS = 60_000

export function useRecorderAlerts(): UseQueryResult<RecorderAlertsResponse, Error> {
  return useQuery({
    queryKey: ['recorder', 'alerts'],
    queryFn: getRecorderAlerts,
    retry: 1,
    refetchInterval: ALERTS_POLL_MS,
    refetchIntervalInBackground: false,
  })
}

export function useRecorderModels(): UseQueryResult<RecorderModelsResponse, Error> {
  return useQuery({
    queryKey: ['recorder', 'models'],
    queryFn: getRecorderModels,
    retry: 1,
    refetchInterval: MODELS_POLL_MS,
    refetchIntervalInBackground: false,
  })
}

export function useRecorderStatus(): UseQueryResult<RecorderStatusResponse, Error> {
  return useQuery({
    queryKey: ['recorder', 'status'],
    queryFn: getRecorderStatus,
    retry: 1,
    refetchInterval: STATUS_POLL_MS,
    refetchIntervalInBackground: false,
  })
}

export function useRecorderCameras(): UseQueryResult<RecorderCamerasResponse, Error> {
  return useQuery({
    queryKey: ['recorder', 'cameras'],
    queryFn: listRecorderCameras,
    retry: 1,
  })
}

export function useRecorderCapabilities(): UseQueryResult<RecorderCapabilitiesResponse, Error> {
  return useQuery({
    queryKey: ['recorder', 'capabilities'],
    queryFn: getRecorderCapabilities,
    retry: 1,
  })
}

export function useRecorderNotifications(): UseQueryResult<RecorderNotificationsResponse, Error> {
  return useQuery({
    queryKey: ['recorder', 'notifications'],
    queryFn: getRecorderNotifications,
    retry: 1,
  })
}

export function useRecorderSettings(): UseQueryResult<RecorderSettingsResponse, Error> {
  return useQuery({
    queryKey: ['recorder', 'settings'],
    queryFn: getRecorderSettings,
    retry: 1,
  })
}

export function useRecorderUntrackedStorage(): UseQueryResult<RecorderUntrackedResponse, Error> {
  return useQuery({
    queryKey: ['recorder', 'storage', 'untracked'],
    queryFn: getRecorderUntrackedStorage,
    retry: 1,
  })
}

/** The seal/revision index for one camera. Undefined `cameraId` means "nothing picked yet". */
export function useRecorderJournal(
  cameraId: string | undefined,
): UseQueryResult<RecorderJournalResponse, Error> {
  return useQuery({
    queryKey: ['recorder', 'journal', cameraId],
    queryFn: () => getRecorderJournal(cameraId as string),
    enabled: cameraId !== undefined,
    retry: 1,
  })
}

/**
 * One journal day, optionally with its full revision history. `history` is
 * part of the query key on purpose: toggling it is a different request (a
 * ~2x larger one on the live instance), and this lets the page keep both
 * responses cached independently instead of re-fetching every toggle.
 */
export function useRecorderJournalDay(
  cameraId: string | undefined,
  date: string | undefined,
  options: { history?: boolean } = {},
): UseQueryResult<RecorderJournalDayResponse, Error> {
  const history = options.history ?? false
  return useQuery({
    queryKey: ['recorder', 'journal', cameraId, date, history],
    queryFn: () => getRecorderJournalDay(cameraId as string, date as string, { history }),
    enabled: cameraId !== undefined && date !== undefined,
    retry: 1,
  })
}

/** The live coverage/integrity/segments report for one camera-day. */
export function useRecorderReport(
  cameraId: string | undefined,
  date: string | undefined,
): UseQueryResult<RecorderCoverageReport, Error> {
  return useQuery({
    queryKey: ['recorder', 'report', cameraId, date],
    queryFn: () => getRecorderReport(cameraId as string, date as string),
    enabled: cameraId !== undefined && date !== undefined,
    retry: 1,
  })
}

/** The mask editor's state for one camera. Undefined `cameraId` means "no camera picked yet". */
export function useRecorderMask(
  cameraId: string | undefined,
): UseQueryResult<RecorderMaskResponse, Error> {
  return useQuery({
    queryKey: ['recorder', 'masks', cameraId],
    queryFn: () => getRecorderMask(cameraId as string),
    enabled: cameraId !== undefined,
    retry: 1,
  })
}

export interface RecorderCameraActionVariables {
  cameraId: string
  action: 'start' | 'stop'
}

/**
 * Worker lifecycle control. Deliberately one shared mutation rather than one
 * per camera row: `CamerasPage` reads `mutation.variables` to know which row
 * (if any) is mid-action, so only one camera can be acted on at a time and the
 * pending/outcome UI never has to be duplicated per row.
 *
 * No `retry` (the mutation default already has none via `renderWithProviders`'
 * QueryClient in tests, and the default QueryClient elsewhere): retrying a
 * start/stop automatically is the one thing that must never happen silently —
 * an operator who clicked once must not risk the recorder seeing two calls.
 *
 * On success, only `['recorder', 'status']` is invalidated. `['recorder',
 * 'cameras']` is the registry (name, mode, source, ...); starting or stopping
 * a worker cannot change any of that, only its running state.
 */
export function useRecorderCameraAction(): UseMutationResult<
  void,
  Error,
  RecorderCameraActionVariables
> {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ cameraId, action }: RecorderCameraActionVariables) =>
      action === 'start' ? startRecorderCamera(cameraId) : stopRecorderCamera(cameraId),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['recorder', 'status'] })
    },
  })
}
