import { describe, expect, it } from 'vitest'
import {
  MAX_LABEL_LENGTH,
  ZONES,
  buildCameraEdit,
  isEmptyEdit,
  labelError,
} from '@/lib/cameraEdit'

const stored = { label: 'Avenue entrance', zone: 'corridor' as const }

describe('buildCameraEdit', () => {
  it('is empty when nothing changed, which the engine rejects as a 422', () => {
    expect(buildCameraEdit(stored, { ...stored })).toEqual({})
    expect(isEmptyEdit(buildCameraEdit(stored, { ...stored }))).toBe(true)
  })

  it('carries only the label when only the label changed', () => {
    const edit = buildCameraEdit(stored, { label: 'East door', zone: 'corridor' })
    expect(edit).toEqual({ label: 'East door' })
    expect('zone' in edit).toBe(false)
  })

  it('carries only the zone when only the zone changed', () => {
    const edit = buildCameraEdit(stored, { label: 'Avenue entrance', zone: 'room' })
    expect(edit).toEqual({ zone: 'room' })
    expect('label' in edit).toBe(false)
  })

  it('carries an explicit null zone when ungrouping', () => {
    const edit = buildCameraEdit(stored, { label: 'Avenue entrance', zone: null })
    expect(edit).toEqual({ zone: null })
    expect('zone' in edit).toBe(true)
  })

  it('omits the zone entirely when it did not change, rather than re-asserting it', () => {
    // Re-asserting an unchanged zone is how one operator's open window reverts
    // another operator's regrouping.
    const edit = buildCameraEdit({ label: 'a', zone: null }, { label: 'b', zone: null })
    expect(edit).toEqual({ label: 'b' })
    expect('zone' in edit).toBe(false)
  })

  it('trims the label before comparing, so whitespace alone is not a change', () => {
    expect(buildCameraEdit(stored, { label: '  Avenue entrance  ', zone: 'corridor' })).toEqual({})
  })

  it('sends the trimmed label, not the raw input', () => {
    expect(buildCameraEdit(stored, { label: '  East door  ', zone: 'corridor' })).toEqual({
      label: 'East door',
    })
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
