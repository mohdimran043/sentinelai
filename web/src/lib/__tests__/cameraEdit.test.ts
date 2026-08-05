import { describe, expect, it } from 'vitest'
import {
  CONCERN_KINDS,
  CONFIDENCE_TIERS,
  DURATION_FIELDS,
  MAX_LABEL_LENGTH,
  ZONES,
  buildCameraEdit,
  durationError,
  isEmptyEdit,
  labelError,
  type CameraRecordDraft,
} from '@/lib/cameraEdit'

/**
 * A camera as the engine echoes it back: every concern kind routed, the default
 * threshold, and no duration overrides. The duration fields are the operator's
 * raw text, so "no override" is the empty string rather than null.
 */
const stored: CameraRecordDraft = {
  label: 'Avenue entrance',
  zone: 'corridor',
  notifyOn: ['altercation', 'collapse', 'distress', 'medication', 'other', 'self_harm'],
  notifyMinConfidence: 'likely',
  clipPrerollSeconds: '',
  clipPostrollSeconds: '',
  summaryIntervalSeconds: '',
}

const draftOf = (patch: Partial<CameraRecordDraft>): CameraRecordDraft => ({ ...stored, ...patch })

describe('buildCameraEdit', () => {
  it('is empty when nothing changed, which the engine rejects as a 422', () => {
    expect(buildCameraEdit(stored, { ...stored })).toEqual({})
    expect(isEmptyEdit(buildCameraEdit(stored, { ...stored }))).toBe(true)
  })

  it('carries only the label when only the label changed', () => {
    const edit = buildCameraEdit(stored, draftOf({ label: 'East door' }))
    expect(edit).toEqual({ label: 'East door' })
    expect('zone' in edit).toBe(false)
  })

  it('carries only the zone when only the zone changed', () => {
    const edit = buildCameraEdit(stored, draftOf({ zone: 'room' }))
    expect(edit).toEqual({ zone: 'room' })
    expect('label' in edit).toBe(false)
  })

  it('carries an explicit null zone when ungrouping', () => {
    const edit = buildCameraEdit(stored, draftOf({ zone: null }))
    expect(edit).toEqual({ zone: null })
    expect('zone' in edit).toBe(true)
  })

  it('omits the zone entirely when it did not change, rather than re-asserting it', () => {
    // Re-asserting an unchanged zone is how one operator's open window reverts
    // another operator's regrouping.
    const edit = buildCameraEdit(
      draftOf({ label: 'a', zone: null }),
      draftOf({ label: 'b', zone: null }),
    )
    expect(edit).toEqual({ label: 'b' })
    expect('zone' in edit).toBe(false)
  })

  it('trims the label before comparing, so whitespace alone is not a change', () => {
    expect(buildCameraEdit(stored, draftOf({ label: '  Avenue entrance  ' }))).toEqual({})
  })

  it('sends the trimmed label, not the raw input', () => {
    expect(buildCameraEdit(stored, draftOf({ label: '  East door  ' }))).toEqual({
      label: 'East door',
    })
  })
})

describe('buildCameraEdit, notify_on', () => {
  it('omits an untouched notify_on rather than re-asserting the whole list', () => {
    // The same class of bug as the zone: re-asserting every kind is how one
    // console's open window restores a camera another operator just muted.
    const edit = buildCameraEdit(stored, draftOf({ label: 'East door' }))
    expect('notify_on' in edit).toBe(false)
  })

  it('sends [] when every kind is unchecked, because that is the mute switch', () => {
    // `[]` is a real instruction — never notify from this camera — and must not
    // be mistaken for "unchanged" and dropped from the body.
    const edit = buildCameraEdit(stored, draftOf({ notifyOn: [] }))
    expect(edit).toEqual({ notify_on: [] })
    expect('notify_on' in edit).toBe(true)
  })

  it('sends the whole remaining list when one kind is unchecked, never a diff', () => {
    const edit = buildCameraEdit(
      stored,
      draftOf({ notifyOn: ['altercation', 'collapse', 'distress', 'medication', 'other'] }),
    )
    expect(edit).toEqual({
      notify_on: ['altercation', 'collapse', 'distress', 'medication', 'other'],
    })
  })

  it('sends the list sorted, so two reads of the same record compare equal', () => {
    const edit = buildCameraEdit(stored, draftOf({ notifyOn: ['self_harm', 'collapse'] }))
    expect(edit).toEqual({ notify_on: ['collapse', 'self_harm'] })
  })

  it('treats a reordered list as unchanged, since the set is what routes', () => {
    const edit = buildCameraEdit(
      stored,
      draftOf({ notifyOn: ['other', 'self_harm', 'medication', 'distress', 'collapse', 'altercation'] }),
    )
    expect(edit).toEqual({})
  })

  it('carries a re-enabled kind away from [] rather than reading as unchanged', () => {
    const muted = draftOf({ notifyOn: [] })
    const edit = buildCameraEdit(muted, draftOf({ notifyOn: ['collapse'] }))
    expect(edit).toEqual({ notify_on: ['collapse'] })
  })
})

describe('buildCameraEdit, notify_min_confidence', () => {
  it('carries the threshold when it changed', () => {
    expect(buildCameraEdit(stored, draftOf({ notifyMinConfidence: 'possible' }))).toEqual({
      notify_min_confidence: 'possible',
    })
  })

  it('omits the threshold when it did not', () => {
    expect('notify_min_confidence' in buildCameraEdit(stored, draftOf({ label: 'x' }))).toBe(false)
  })
})

describe('buildCameraEdit, the duration overrides', () => {
  it('sends a number when an override is set on a camera that had none', () => {
    expect(buildCameraEdit(stored, draftOf({ clipPrerollSeconds: '4' }))).toEqual({
      clip_preroll_seconds: 4,
    })
  })

  it('sends an explicit null when a stored override is cleared', () => {
    // Clearing the box means "revert to the engine default", which only null
    // says. Omitting it would leave the camera pinned to the old number.
    const withOverride = draftOf({ clipPrerollSeconds: '4' })
    const edit = buildCameraEdit(withOverride, draftOf({ clipPrerollSeconds: '' }))
    expect(edit).toEqual({ clip_preroll_seconds: null })
    expect('clip_preroll_seconds' in edit).toBe(true)
  })

  it('omits an untouched override rather than re-asserting the stored number', () => {
    const withOverride = draftOf({ clipPostrollSeconds: '8' })
    const edit = buildCameraEdit(withOverride, { ...withOverride, label: 'East door' })
    expect(edit).toEqual({ label: 'East door' })
  })

  it('compares by value, so retyping 5 as 5.0 is not a change', () => {
    const withOverride = draftOf({ summaryIntervalSeconds: '5' })
    expect(buildCameraEdit(withOverride, draftOf({ summaryIntervalSeconds: '5.0' }))).toEqual({})
  })

  it('sends zero pre-roll as a real value, not as "cleared"', () => {
    // `0` is a legal pre-roll meaning no lead-in at all, and `''` means revert
    // to the default. Conflating them writes a number nobody chose.
    const edit = buildCameraEdit(stored, draftOf({ clipPrerollSeconds: '0' }))
    expect(edit).toEqual({ clip_preroll_seconds: 0 })
  })

  it('omits a field the operator has left unparseable rather than sending garbage', () => {
    // Save is blocked on this anyway; the body must still never carry NaN.
    const edit = buildCameraEdit(stored, draftOf({ clipPrerollSeconds: '1.' + 'x' }))
    expect(edit).toEqual({})
  })

  it('carries every duration at once when all three change', () => {
    expect(
      buildCameraEdit(
        stored,
        draftOf({
          clipPrerollSeconds: '3',
          clipPostrollSeconds: '9',
          summaryIntervalSeconds: '30',
        }),
      ),
    ).toEqual({
      clip_preroll_seconds: 3,
      clip_postroll_seconds: 9,
      summary_interval_seconds: 30,
    })
  })
})

describe('durationError', () => {
  const preroll = DURATION_FIELDS.find((field) => field.draftKey === 'clipPrerollSeconds')!
  const postroll = DURATION_FIELDS.find((field) => field.draftKey === 'clipPostrollSeconds')!
  const summary = DURATION_FIELDS.find((field) => field.draftKey === 'summaryIntervalSeconds')!

  it('accepts an empty box, which means "use the default"', () => {
    expect(durationError('', preroll)).toBeNull()
    expect(durationError('   ', postroll)).toBeNull()
  })

  it('accepts an ordinary duration', () => {
    expect(durationError('4.5', preroll)).toBeNull()
    expect(durationError('8', postroll)).toBeNull()
    expect(durationError('30', summary)).toBeNull()
  })

  it('rejects text that is not a number', () => {
    expect(durationError('soon', preroll)).toMatch(/number/i)
  })

  it('rejects a non-finite value, matching the engine\'s allow_inf_nan=False', () => {
    expect(durationError('Infinity', postroll)).toMatch(/number/i)
    expect(durationError('NaN', summary)).toMatch(/number/i)
  })

  it('allows a zero pre-roll but not a zero post-roll or interval', () => {
    // The engine's bounds exactly: pre-roll is `>= 0` because no lead-in is a
    // real choice; the other two are `> 0` because a zero-length clip and a
    // zero-second interval are not shorter, they are nothing.
    expect(durationError('0', preroll)).toBeNull()
    expect(durationError('0', postroll)).toMatch(/greater than zero/i)
    expect(durationError('0', summary)).toMatch(/greater than zero/i)
  })

  it('rejects a negative value on every field', () => {
    expect(durationError('-1', preroll)).toMatch(/negative/i)
    expect(durationError('-1', postroll)).toMatch(/greater than zero/i)
    expect(durationError('-1', summary)).toMatch(/greater than zero/i)
  })
})

describe('labelError', () => {
  it('accepts an ordinary label', () => {
    expect(labelError('East corridor, door end')).toBeNull()
  })

  it('rejects an empty label', () => {
    expect(labelError('')).toMatch(/empty/i)
  })

  it('rejects a whitespace-only label, which is empty to an operator', () => {
    expect(labelError('   ')).toMatch(/empty/i)
  })

  it(`rejects a label longer than ${MAX_LABEL_LENGTH} characters`, () => {
    expect(labelError('x'.repeat(MAX_LABEL_LENGTH))).toBeNull()
    expect(labelError('x'.repeat(MAX_LABEL_LENGTH + 1))).toMatch(/120/)
  })

  it('measures length after trimming, matching the engine', () => {
    expect(labelError(`  ${'x'.repeat(MAX_LABEL_LENGTH)}  `)).toBeNull()
  })
})

describe('ZONES', () => {
  it('lists every zone the contract defines', () => {
    expect([...ZONES]).toEqual(['room', 'corridor', 'dayroom'])
  })
})

describe('CONCERN_KINDS', () => {
  it('lists every concern kind the contract defines', () => {
    // A kind missing here is a concern the operator cannot route, on a screen
    // that looks complete. The `satisfies` guard in the module fails the build
    // if the contract gains one; this fails the suite if it loses one.
    expect([...CONCERN_KINDS]).toEqual([
      'collapse',
      'altercation',
      'self_harm',
      'medication',
      'distress',
      'other',
    ])
  })
})

describe('CONFIDENCE_TIERS', () => {
  it('offers exactly the two tiers, with no "certain"', () => {
    // There is no `certain`: one still frame cannot earn it. Offering one would
    // be the console inventing a tier the engine will 422.
    expect([...CONFIDENCE_TIERS]).toEqual(['possible', 'likely'])
  })
})
