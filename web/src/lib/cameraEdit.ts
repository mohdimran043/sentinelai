import type { CameraEditRequest, ConcernKind, Confidence, Zone } from '@/api/engineClient'

/**
 * A camera record as this console holds it while it is being edited.
 *
 * The duration overrides are the operator's **raw text**, not numbers, because a
 * half-typed `1.` is a state a number cannot hold and clearing the box is a real
 * instruction ("revert to the default") that must survive as `''` rather than
 * collapsing into zero. `buildCameraEdit` compares them by parsed value, so
 * retyping `5` as `5.0` is not a change.
 */
export interface CameraRecordDraft {
  label: string
  zone: Zone | null
  notifyOn: readonly ConcernKind[]
  notifyMinConfidence: Confidence
  clipPrerollSeconds: string
  clipPostrollSeconds: string
  summaryIntervalSeconds: string
}

export const MAX_LABEL_LENGTH = 120

/**
 * Every zone the contract defines, in the order the console offers them.
 *
 * A runtime list is unavoidable — a TypeScript union cannot be enumerated — so
 * `zonesAreExhaustive` below fails the build if the contract gains a zone that
 * nobody added here. Silently offering an operator a short list is how a camera
 * ends up ungrouped because the zone it belonged in was not on the menu.
 */
export const ZONES = ['room', 'corridor', 'dayroom'] as const satisfies readonly Zone[]

type MissingZones = Exclude<Zone, (typeof ZONES)[number]>
const zonesAreExhaustive: MissingZones extends never ? true : never = true
void zonesAreExhaustive

/**
 * Every concern kind, guarded the same way and for a sharper reason: a kind
 * missing from this list is one the operator cannot route, on a screen that
 * looks complete — so the camera would go on not telling anyone about it while
 * the panel showed a full set of checkboxes.
 */
export const CONCERN_KINDS = [
  'collapse',
  'altercation',
  'self_harm',
  'medication',
  'distress',
  'other',
] as const satisfies readonly ConcernKind[]

type MissingKinds = Exclude<ConcernKind, (typeof CONCERN_KINDS)[number]>
const kindsAreExhaustive: MissingKinds extends never ? true : never = true
void kindsAreExhaustive

/**
 * The two confidence tiers, weakest first. There is no `certain` and there must
 * never be one here: a single still frame cannot earn it, so offering the tier
 * would be the console inventing a routing level the engine rejects.
 */
export const CONFIDENCE_TIERS = ['possible', 'likely'] as const satisfies readonly Confidence[]

type MissingTiers = Exclude<Confidence, (typeof CONFIDENCE_TIERS)[number]>
const tiersAreExhaustive: MissingTiers extends never ? true : never = true
void tiersAreExhaustive

export type DurationDraftKey =
  | 'clipPrerollSeconds'
  | 'clipPostrollSeconds'
  | 'summaryIntervalSeconds'

export interface DurationField {
  readonly draftKey: DurationDraftKey
  readonly wireKey: 'clip_preroll_seconds' | 'clip_postroll_seconds' | 'summary_interval_seconds'
  /** The form label, and the subject of every message about this field. */
  readonly label: string
  /**
   * Whether zero is a value or a refusal — the engine's `ge=0` against its
   * `gt=0`, mirrored so an operator is told here rather than by a 422.
   */
  readonly zeroAllowed: boolean
  /** What an empty box falls back to, in the operator's terms. */
  readonly fallback: string
  readonly hint: string
}

export const DURATION_FIELDS = [
  {
    draftKey: 'clipPrerollSeconds',
    wireKey: 'clip_preroll_seconds',
    label: 'Clip pre-roll (seconds)',
    zeroAllowed: true,
    fallback: 'engine default',
    hint: 'Buffered video kept before the keyframe. 0 means no lead-in at all.',
  },
  {
    draftKey: 'clipPostrollSeconds',
    wireKey: 'clip_postroll_seconds',
    label: 'Clip post-roll (seconds)',
    zeroAllowed: false,
    fallback: 'engine default',
    hint: 'Video kept after the keyframe. A notification cannot go out before this elapses, so a shorter post-roll is a faster alert and less evidence.',
  },
  {
    draftKey: 'summaryIntervalSeconds',
    wireKey: 'summary_interval_seconds',
    label: 'Summary interval (seconds)',
    zeroAllowed: false,
    fallback: 'profile default',
    hint: "How often this camera is looked at even when nothing has changed. Empty falls back to the camera profile's own interval, not to an engine-wide one.",
  },
] as const satisfies readonly DurationField[]

/**
 * A stored override as the operator's text: `null` is the empty box, because
 * "no override" and "zero seconds" are different answers and only one of them
 * is a number.
 */
export function durationText(value: number | null): string {
  return value === null ? '' : String(value)
}

/**
 * The message to show for a duration box, or null when it is acceptable.
 *
 * Mirrors the engine's own bounds — `ge=0` for the pre-roll, `gt=0` for the
 * other two, `allow_inf_nan=False` throughout — so the operator learns about a
 * bad value from the field rather than from a 422 after pressing save.
 */
export function durationError(raw: string, field: DurationField): string | null {
  const trimmed = raw.trim()
  if (trimmed.length === 0) return null

  const value = Number(trimmed)
  if (!Number.isFinite(value)) {
    return `${field.label} must be a number of seconds, or empty to use the ${field.fallback}.`
  }
  if (field.zeroAllowed) {
    if (value < 0) return `${field.label} cannot be negative.`
  } else if (value <= 0) {
    return `${field.label} must be greater than zero.`
  }
  return null
}

type ReadDuration = { ok: true; value: number | null } | { ok: false }

function readDuration(raw: string, field: DurationField): ReadDuration {
  if (durationError(raw, field) !== null) return { ok: false }
  const trimmed = raw.trim()
  return { ok: true, value: trimmed.length === 0 ? null : Number(trimmed) }
}

/** Sorted and deduplicated, matching the whole-list form the engine stores. */
function sortKinds(kinds: readonly ConcernKind[]): ConcernKind[] {
  return [...new Set(kinds)].sort()
}

/** Order is presentation; the set is what routes. */
export function sameKinds(a: readonly ConcernKind[], b: readonly ConcernKind[]): boolean {
  const left = sortKinds(a)
  const right = sortKinds(b)
  return left.length === right.length && left.every((kind, index) => kind === right[index])
}

/**
 * The request body for a draft, carrying **only** what actually changed.
 *
 * Which keys are present is the instruction, not an optimisation. On this
 * endpoint an omitted `zone` means "leave the grouping alone" and an explicit
 * `zone: null` means "ungroup", so a body that always carried both could not
 * express the difference. Omitting unchanged fields is also what stops one
 * operator's open window from reverting another's edit.
 *
 * The same rule governs every welfare field, where it bites harder:
 * `notify_on: []` is the mute switch and must be sent when the operator empties
 * the list, while an untouched `notify_on` must be left out entirely — restating
 * the full list is how a console silently un-mutes a camera somebody silenced.
 * A duration cleared to `''` sends `null` ("revert to the default"); one left
 * alone is omitted, so the camera is never pinned to a number nobody chose.
 *
 * A duration the operator has left unparseable is omitted rather than sent as
 * `NaN`. Save is blocked on it separately — this is the belt to that braces.
 *
 * An empty result is a request the engine answers 422 — see `isEmptyEdit`.
 */
export function buildCameraEdit(
  stored: CameraRecordDraft,
  draft: CameraRecordDraft,
): CameraEditRequest {
  const edit: CameraEditRequest = {}

  const label = draft.label.trim()
  if (label !== stored.label) {
    edit.label = label
  }
  if (draft.zone !== stored.zone) {
    edit.zone = draft.zone
  }
  if (!sameKinds(draft.notifyOn, stored.notifyOn)) {
    edit.notify_on = sortKinds(draft.notifyOn)
  }
  if (draft.notifyMinConfidence !== stored.notifyMinConfidence) {
    edit.notify_min_confidence = draft.notifyMinConfidence
  }
  for (const field of DURATION_FIELDS) {
    const next = readDuration(draft[field.draftKey], field)
    const base = readDuration(stored[field.draftKey], field)
    if (!next.ok || !base.ok) continue
    if (next.value !== base.value) {
      edit[field.wireKey] = next.value
    }
  }

  return edit
}

export function isEmptyEdit(edit: CameraEditRequest): boolean {
  return Object.keys(edit).length === 0
}

/**
 * Mirrors the engine's `CameraLabel` constraint: trim first, then require
 * non-empty and bounded. Trimming first is what makes `""` and `"   "` the same
 * answer — both are empty to an operator, only one is falsy to JavaScript.
 *
 * Returns the message to show, or null when the label is acceptable.
 */
export function labelError(label: string): string | null {
  const trimmed = label.trim()
  if (trimmed.length === 0) {
    return 'A camera label cannot be empty.'
  }
  if (trimmed.length > MAX_LABEL_LENGTH) {
    return `A camera label can be at most ${MAX_LABEL_LENGTH} characters (this one is ${trimmed.length}).`
  }
  return null
}
