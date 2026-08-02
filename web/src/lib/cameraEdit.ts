import type { CameraEditRequest, Zone } from '@/api/engineClient'

/** The two fields of a camera record this console can change. */
export interface CameraRecordDraft {
  label: string
  zone: Zone | null
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
 * The request body for a draft, carrying **only** what actually changed.
 *
 * Which keys are present is the instruction, not an optimisation. On this
 * endpoint an omitted `zone` means "leave the grouping alone" and an explicit
 * `zone: null` means "ungroup", so a body that always carried both could not
 * express the difference. Omitting unchanged fields is also what stops one
 * operator's open window from reverting another's edit.
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
