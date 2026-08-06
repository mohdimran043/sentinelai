import { describe, expect, it } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import { WelfareConcerns } from '@/routes/camera/WelfareConcerns'
import type { WelfareConcernEntry } from '@/api/engineClient'

const concern = (overrides: Partial<WelfareConcernEntry> = {}): WelfareConcernEntry => ({
  kind: 'collapse',
  confidence: 'likely',
  evidence: 'A person is lying motionless by the door.',
  evidence_stated: true,
  ...overrides,
})

describe('WelfareConcerns', () => {
  it('renders nothing at all when the model reported no concerns', () => {
    // Not an empty box with a heading: "nothing to report" and "not assessed"
    // look identical to an operator, and only one of them is true here.
    const { container } = render(<WelfareConcerns concerns={[]} />)

    expect(container).toBeEmptyDOMElement()
  })

  it('names the concern kind in words rather than the wire value', () => {
    render(<WelfareConcerns concerns={[concern({ kind: 'self_harm' })]} />)

    expect(screen.getByTestId('welfare-concern')).toHaveTextContent('Self harm')
  })

  it("shows the model's own words as the evidence", () => {
    render(<WelfareConcerns concerns={[concern()]} />)

    expect(screen.getByTestId('welfare-concern')).toHaveTextContent(
      'A person is lying motionless by the door.',
    )
  })

  it('renders every concern, not just the first', () => {
    render(
      <WelfareConcerns
        concerns={[
          concern({ kind: 'collapse' }),
          concern({ kind: 'medication', confidence: 'possible' }),
          concern({ kind: 'distress' }),
        ]}
      />,
    )

    expect(screen.getAllByTestId('welfare-concern')).toHaveLength(3)
  })

  it('distinguishes a likely concern from a possible one', () => {
    // The two tiers are the entire confidence scale. Rendering them alike throws
    // away the only uncertainty signal the model gives.
    render(
      <WelfareConcerns
        concerns={[
          concern({ kind: 'collapse', confidence: 'likely' }),
          concern({ kind: 'distress', confidence: 'possible' }),
        ]}
      />,
    )

    const [likely, possible] = screen.getAllByTestId('welfare-concern')
    expect(likely).toHaveAttribute('data-confidence', 'likely')
    expect(possible).toHaveAttribute('data-confidence', 'possible')
    expect(likely).toHaveTextContent(/likely/i)
    expect(possible).toHaveTextContent(/possible/i)
  })

  it('says when the model named a concern but described nothing', () => {
    // `evidence` is a fixed placeholder in this case. Rendering it as prose would
    // present a stand-in as the model's observation.
    render(
      <WelfareConcerns
        concerns={[concern({ evidence: 'not stated', evidence_stated: false })]}
      />,
    )

    const row = screen.getByTestId('welfare-concern')
    expect(row).toHaveTextContent(/no reason given/i)
    expect(row).not.toHaveTextContent('not stated')
  })

  it('still shows an unevidenced concern rather than hiding it', () => {
    // A concern without a stated reason is not a concern that did not happen.
    render(
      <WelfareConcerns
        concerns={[concern({ kind: 'collapse', evidence: 'not stated', evidence_stated: false })]}
      />,
    )

    expect(screen.getByTestId('welfare-concern')).toHaveTextContent('Collapse')
  })

  it('says this is an opinion about one frame, not a detector reading', () => {
    render(<WelfareConcerns concerns={[concern()]} />)

    const note = screen.getByTestId('welfare-concerns-note')
    expect(note).toHaveTextContent(/single frame/i)
    expect(note).toHaveTextContent(/not a detector/i)
  })

  it('carries the caveat once, not once per concern', () => {
    render(<WelfareConcerns concerns={[concern(), concern({ kind: 'distress' })]} />)

    expect(screen.getAllByTestId('welfare-concerns-note')).toHaveLength(1)
  })

  it('keeps each concern kind with its own evidence', () => {
    render(
      <WelfareConcerns
        concerns={[
          concern({ kind: 'collapse', evidence: 'Lying by the door.' }),
          concern({ kind: 'medication', evidence: 'Tipping a bottle towards the mouth.' }),
        ]}
      />,
    )

    const rows = screen.getAllByTestId('welfare-concern')
    expect(within(rows[0]).getByText(/Lying by the door\./)).toBeInTheDocument()
    expect(within(rows[1]).getByText(/Tipping a bottle towards the mouth\./)).toBeInTheDocument()
  })
})
