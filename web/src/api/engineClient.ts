import { ENGINE_BASE_URL } from '@/api/config'
import type { components } from '@/api/engine.types'

export type HealthResponse = components['schemas']['HealthResponse']
export type CamerasResponse = components['schemas']['CamerasResponse']
export type CameraStatus = components['schemas']['CameraStatus']
export type DescribeResponse = components['schemas']['DescribeResponse']
export type CameraEventsResponse = components['schemas']['CameraEventsResponse']
export type RecentEventEntry = components['schemas']['RecentEventEntry']
export type LatestDescriptionState = CameraEventsResponse['latest_description_state']
export type CameraEditRequest = components['schemas']['CameraEditRequest']
export type CameraEditResponse = components['schemas']['CameraEditResponse']
export type Zone = components['schemas']['Zone']
export type ZoneKind = components['schemas']['ZoneKind']
export type ConcernKind = components['schemas']['ConcernKind']
export type Confidence = components['schemas']['Confidence']
export type WelfareConcernEntry = components['schemas']['WelfareConcernEntry']
export type AlertEntry = components['schemas']['AlertEntry']
export type AlertsResponse = components['schemas']['AlertsResponse']
export type AlertState = components['schemas']['AlertState']
export type EventPriority = components['schemas']['EventPriority']
export type EscalationReason = components['schemas']['EscalationReason']
export type Capability = components['schemas']['Capability']
export type PersonEntry = components['schemas']['PersonEntry']
export type CameraCreateRequest = components['schemas']['CameraCreateRequest']
export type ProbeResponse = components['schemas']['ProbeResponse']
export type EnrolledFaceEntry = components['schemas']['EnrolledFaceEntry']
export type FacesResponse = components['schemas']['FacesResponse']
export type PeopleResponse = components['schemas']['PeopleResponse']
export type PersonRequest = components['schemas']['PersonRequest']
export type PersonStatus = components['schemas']['PersonStatus']

/** The engine did not answer at all — network failure, DNS, connection refused. */
export class EngineUnreachableError extends Error {
  constructor(cause: unknown) {
    super('The AI engine is unreachable.')
    this.name = 'EngineUnreachableError'
    this.cause = cause
  }
}

/** The engine answered with a non-2xx status. */
export class EngineHttpError extends Error {
  readonly status: number
  /**
   * The engine's own `detail` string, unprefixed. `message` is for a log line;
   * this is for showing an operator, because the engine's write endpoints
   * explain themselves at length and re-wording that in the console would only
   * make the two disagree.
   */
  readonly detail: string
  constructor(status: number, detail: string) {
    super(`AI engine returned ${status}: ${detail}`)
    this.name = 'EngineHttpError'
    this.status = status
    this.detail = detail
  }
}

/**
 * The engine's own `detail` from an error body, or the bare status text.
 *
 * The 204 endpoints cannot go through `request`, which always parses a body, so they
 * each need this on the error path — and the detail is what distinguishes "no such
 * person" from "this deployment does no face recognition at all". Shared rather than
 * copied so the two cannot drift.
 */
async function detailFrom(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json()
    if (body && typeof body === 'object' && 'detail' in body) {
      return String((body as { detail: unknown }).detail)
    }
  } catch {
    // Not JSON — fall through to the status text.
  }
  return response.statusText
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${ENGINE_BASE_URL}${path}`, {
      ...init,
      headers: { Accept: 'application/json', ...init?.headers },
    })
  } catch (cause) {
    throw new EngineUnreachableError(cause)
  }

  if (!response.ok) {
    let detail = response.statusText
    try {
      const body: unknown = await response.json()
      if (body && typeof body === 'object' && 'detail' in body) {
        detail = String((body as { detail: unknown }).detail)
      }
    } catch {
      // Body wasn't JSON — keep statusText.
    }
    throw new EngineHttpError(response.status, detail)
  }

  return (await response.json()) as T
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health')
}

export function listCameras(): Promise<CamerasResponse> {
  return request<CamerasResponse>('/cameras')
}

export function getCameraTelemetry(cameraId: string): Promise<CameraStatus> {
  return request<CameraStatus>(`/cameras/${encodeURIComponent(cameraId)}/telemetry`)
}

export function describeCameraNow(cameraId: string): Promise<DescribeResponse> {
  return request<DescribeResponse>(`/cameras/${encodeURIComponent(cameraId)}/describe`, {
    method: 'POST',
  })
}

/**
 * A volatile, bounded, in-memory ring of this camera's recent events (capped at
 * `capacity`, currently 200 — see `CameraEventsResponse`'s own description). Not
 * the event store: an absent event here means "not in the last `capacity` events
 * of this process run", never "did not happen". `latest` is `events.at(-1)` read
 * from the same snapshot, so the live panel and the chart cannot disagree.
 */
export function getCameraEvents(cameraId: string): Promise<CameraEventsResponse> {
  return request<CameraEventsResponse>(`/cameras/${encodeURIComponent(cameraId)}/events`)
}

/**
 * Edit a camera's `label`, `zone` and welfare notification policy, persisted to
 * the engine's `cameras.json` before the response is sent.
 *
 * Sends `edit` verbatim, because *which keys are present* is the instruction:
 * an omitted `zone` leaves the grouping alone and an explicit `zone: null`
 * ungroups the camera; an omitted `notify_on` leaves the routing alone while
 * `notify_on: []` mutes the camera. Build the body with `buildCameraEdit`
 * rather than assembling it at the call site.
 */
export function updateCamera(
  cameraId: string,
  edit: CameraEditRequest,
): Promise<CameraEditResponse> {
  return request<CameraEditResponse>(`/cameras/${encodeURIComponent(cameraId)}`, {
    method: 'PATCH',
    // `request` sets only `Accept`. Every other call is a GET or a bodyless
    // POST, so nothing has needed this before; a PATCH carrying JSON without
    // it is a 422 from FastAPI.
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(edit),
  })
}


/**
 * The operator's alert list — **episodes, not events**. Forty-five seconds of one
 * corridor produced eleven events in measurement and would produce eleven rows
 * without this; an alert folds those into one row with an occurrence count.
 *
 * Volatile, and more sharply than the event ring: the register lives in engine
 * memory, so **an acknowledgement does not survive a restart**.
 */
export function listAlerts(): Promise<AlertsResponse> {
  return request<AlertsResponse>('/alerts')
}

/**
 * Record that a person has seen an alert. `by` is a self-declared label rather
 * than an identity — this engine has no authentication — which is why the UI
 * asks for a name rather than assuming one.
 */
export function acknowledgeAlert(alertId: string, by: string): Promise<AlertEntry> {
  return request<AlertEntry>(`/alerts/${encodeURIComponent(alertId)}/acknowledge`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ by }),
  })
}

/** Record that a person has finished with an alert. Idempotent. */
export function resolveAlert(alertId: string): Promise<AlertEntry> {
  return request<AlertEntry>(`/alerts/${encodeURIComponent(alertId)}/resolve`, {
    method: 'POST',
  })
}

/**
 * The enrolled roster. Carries **no biometric data** — no embeddings, no vectors,
 * no reference images. `reference_faces` is a count, and it is the one biometric
 * fact worth showing: "1" is usually why somebody is not recognised at an angle.
 */
export function listPeople(): Promise<PeopleResponse> {
  return request<PeopleResponse>('/authorized-persons')
}

/** Create or replace an authorised person. `camera_ids: []` authorises nowhere. */
export function upsertPerson(personId: string, person: PersonRequest): Promise<PersonEntry> {
  return request<PersonEntry>(`/authorized-persons/${encodeURIComponent(personId)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(person),
  })
}

/**
 * Look at a stream before committing to it.
 *
 * `ok: false` arrives as a 200 with a reason, not an error status — a URL that does not
 * play and a request that was malformed are different things, and only the first is
 * something the operator can fix in the form they are looking at.
 */
export function probeSource(url: string): Promise<ProbeResponse> {
  return request<ProbeResponse>('/cameras/probe', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ url }),
  })
}

/** Add a camera. The engine persists it and starts watching immediately. */
export function createCamera(camera: CameraCreateRequest): Promise<CameraEditResponse> {
  return request<CameraEditResponse>('/cameras', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(camera),
  })
}

/** Stop a camera and remove it from the engine's file. */
export async function deleteCamera(cameraId: string): Promise<void> {
  const path = `/cameras/${encodeURIComponent(cameraId)}`
  let response: Response
  try {
    response = await fetch(`${ENGINE_BASE_URL}${path}`, { method: 'DELETE' })
  } catch (cause) {
    throw new EngineUnreachableError(cause)
  }
  if (!response.ok) {
    throw new EngineHttpError(response.status, await detailFrom(response))
  }
}

/**
 * The URL of a camera's latest decoded frame.
 *
 * The fallback for a camera with no mediamtx playlist — an EarthCam page, a local file.
 * `at` is a cache-buster: the engine sends `Cache-Control: no-store`, but a browser will
 * still reuse an `<img src>` it considers unchanged, and the whole value of this picture
 * is that it is current.
 */
export function snapshotUrl(cameraId: string, at: number): string {
  return `${ENGINE_BASE_URL}/cameras/${encodeURIComponent(cameraId)}/snapshot?at=${at}`
}

/**
 * Clear the whole alert list.
 *
 * Throws away triage state, never evidence — every event behind these alerts is already
 * published and still there. The console asks twice before calling it.
 */
export function clearAlerts(): Promise<{ cleared: number }> {
  return request<{ cleared: number }>('/alerts', { method: 'DELETE' })
}

/** The references on one person's record — metadata, never the pictures themselves. */
export function listFaces(personId: string): Promise<FacesResponse> {
  return request<FacesResponse>(`/authorized-persons/${encodeURIComponent(personId)}/faces`)
}

/**
 * The URL of one stored reference photograph.
 *
 * A URL rather than a fetched blob: `<img src>` lets the browser do the request,
 * which keeps the bytes out of JavaScript memory entirely. The engine answers it
 * with `Cache-Control: no-store`, so the picture is not left in a disk cache that
 * nothing here could clear when the person is deleted.
 */
export function faceImageUrl(personId: string, faceId: string): string {
  return (
    `${ENGINE_BASE_URL}/authorized-persons/${encodeURIComponent(personId)}` +
    `/faces/${encodeURIComponent(faceId)}/image`
  )
}

/**
 * Where to play one alert's recording.
 *
 * A URL rather than a fetch, because the consumer is a `<video src>` and going through
 * `request` would mean holding the whole clip in JS memory to hand it straight back to
 * the element that could have streamed it.
 *
 * `short` is the three-second notification copy — the length somebody watches while
 * triaging rather than scrolls past. The engine falls back to the full recording when
 * no short copy was made, so this never 404s for a reason the caller could have
 * avoided.
 */
export function alertClipUrl(alertId: string, { short = true }: { short?: boolean } = {}): string {
  return (
    `${ENGINE_BASE_URL}/alerts/${encodeURIComponent(alertId)}/clip` +
    `?short=${short ? 'true' : 'false'}`
  )
}

/** Remove one reference face — its embedding and its photograph together. */
export async function deleteFace(personId: string, faceId: string): Promise<void> {
  const path =
    `/authorized-persons/${encodeURIComponent(personId)}` +
    `/faces/${encodeURIComponent(faceId)}`
  let response: Response
  try {
    response = await fetch(`${ENGINE_BASE_URL}${path}`, { method: 'DELETE' })
  } catch (cause) {
    throw new EngineUnreachableError(cause)
  }
  if (!response.ok) {
    // Same shape as `deletePerson` below and for the same reason: a 204 endpoint
    // carries no body on success, so this cannot go through `request`, and the
    // engine's detail on the error path distinguishes "no such face" from "this
    // deployment does no face recognition at all".
    throw new EngineHttpError(response.status, await detailFrom(response))
  }
}

/**
 * Delete a person **and every face enrolled for them**. Not a soft delete — a
 * record that removed the name while keeping the vectors would keep exactly the
 * part that identifies somebody.
 */
export async function deletePerson(personId: string): Promise<void> {
  const path = `/authorized-persons/${encodeURIComponent(personId)}`
  let response: Response
  try {
    response = await fetch(`${ENGINE_BASE_URL}${path}`, { method: 'DELETE' })
  } catch (cause) {
    throw new EngineUnreachableError(cause)
  }
  if (!response.ok) {
    throw new EngineHttpError(response.status, await detailFrom(response))
  }
}

/**
 * Enrol one reference face from a photograph.
 *
 * The photograph itself is **not** stored. What is kept is the detector's own face
 * box with a margin, at most 320px on its longest side, sealed with the same key as
 * the embedding — enough for an operator to recognise somebody, and deliberately not
 * a copy of whatever else was in the frame.
 *
 * Enrol more than one: a single face-on photograph matches a corridor camera at an
 * angle poorly, and a second reference fixes that where a lower threshold would not.
 */
export async function enrollFace(personId: string, image: File): Promise<PersonEntry> {
  const body = new FormData()
  body.append('image', image)
  const path = `/authorized-persons/${encodeURIComponent(personId)}/faces`
  let response: Response
  try {
    // No `Content-Type` header: the browser sets it with the multipart boundary,
    // and setting it by hand produces a body the server cannot parse.
    response = await fetch(`${ENGINE_BASE_URL}${path}`, { method: 'POST', body })
  } catch (cause) {
    throw new EngineUnreachableError(cause)
  }
  if (!response.ok) {
    let detail = response.statusText
    try {
      const parsed: unknown = await response.json()
      if (parsed && typeof parsed === 'object' && 'detail' in parsed) {
        detail = String((parsed as { detail: unknown }).detail)
      }
    } catch {
      // Not JSON — keep statusText.
    }
    throw new EngineHttpError(response.status, detail)
  }
  return (await response.json()) as PersonEntry
}
