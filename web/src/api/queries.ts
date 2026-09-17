import { useMutation, useQuery, useQueryClient, type UseQueryResult } from '@tanstack/react-query'
import {
  acknowledgeAlert,
  clearAlerts,
  createCamera,
  deleteCamera,
  deleteFace,
  deletePerson,
  enrollFace,
  listAlerts,
  listPeople,
  resolveAlert,
  listFaces,
  probeSource,
  upsertPerson,
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
  type AlertsResponse,
  type PeopleResponse,
  type CameraCreateRequest,
  type FacesResponse,
  type PersonRequest,
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

/**
 * The alert list. Polled fast — this is the one view where being a few seconds
 * stale changes what an operator does, and the payload is a few dozen rows.
 */
const ALERTS_POLL_MS = 3_000

export function useAlerts(): UseQueryResult<AlertsResponse, Error> {
  return useQuery({
    queryKey: ['engine', 'alerts'],
    queryFn: listAlerts,
    retry: 1,
    refetchInterval: ALERTS_POLL_MS,
    refetchIntervalInBackground: false,
  })
}

export function useAcknowledgeAlert() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ alertId, by }: { alertId: string; by: string }) =>
      acknowledgeAlert(alertId, by),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['engine', 'alerts'] })
    },
  })
}

export function useResolveAlert() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (alertId: string) => resolveAlert(alertId),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['engine', 'alerts'] })
    },
  })
}

/**
 * The enrolled roster. Polled slowly: it only changes when an operator changes
 * it, unlike everything else on this console.
 */
const PEOPLE_POLL_MS = 30_000

export function usePeople(): UseQueryResult<PeopleResponse, Error> {
  return useQuery({
    queryKey: ['engine', 'people'],
    queryFn: listPeople,
    retry: 1,
    refetchInterval: PEOPLE_POLL_MS,
    refetchIntervalInBackground: false,
  })
}

/**
 * The references on one person's record.
 *
 * Not folded into `usePeople`: a roster of fifty people would then fetch fifty face
 * lists to draw a table that shows a count, and the faces are only looked at when
 * somebody expands a person. Enabled explicitly by the caller for that reason.
 */
/**
 * Probe a URL. A mutation rather than a query: it is an action the operator takes, it
 * costs a network fetch and a decode on the engine, and it must never re-run on its own
 * because the operator is still typing.
 */
export function useProbeSource() {
  return useMutation({ mutationFn: (url: string) => probeSource(url) })
}

export function useCreateCamera() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (camera: CameraCreateRequest) => createCamera(camera),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['engine', 'cameras'] })
    },
  })
}

export function useDeleteCamera() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (cameraId: string) => deleteCamera(cameraId),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['engine', 'cameras'] })
    },
  })
}

export function useClearAlerts() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: clearAlerts,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['engine', 'alerts'] })
    },
  })
}

export function useFaces(personId: string, enabled: boolean): UseQueryResult<FacesResponse, Error> {
  return useQuery({
    queryKey: ['engine', 'people', personId, 'faces'],
    queryFn: () => listFaces(personId),
    enabled,
    retry: 1,
  })
}

export function useDeleteFace() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ personId, faceId }: { personId: string; faceId: string }) =>
      deleteFace(personId, faceId),
    onSuccess: (_result, { personId }) => {
      // Both: the gallery loses a thumbnail and the roster's `reference_faces` count
      // changes with it. Invalidating only one leaves the page contradicting itself.
      void queryClient.invalidateQueries({ queryKey: ['engine', 'people', personId, 'faces'] })
      void queryClient.invalidateQueries({ queryKey: ['engine', 'people'] })
    },
  })
}

export function useUpsertPerson() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ personId, person }: { personId: string; person: PersonRequest }) =>
      upsertPerson(personId, person),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['engine', 'people'] })
    },
  })
}

export function useDeletePerson() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (personId: string) => deletePerson(personId),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['engine', 'people'] })
    },
  })
}

export function useEnrollFace() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ personId, image }: { personId: string; image: File }) =>
      enrollFace(personId, image),
    onSuccess: (_result, { personId }) => {
      void queryClient.invalidateQueries({ queryKey: ['engine', 'people', personId, 'faces'] })
      void queryClient.invalidateQueries({ queryKey: ['engine', 'people'] })
    },
  })
}
