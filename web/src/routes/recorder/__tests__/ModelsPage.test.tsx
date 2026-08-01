import { describe, expect, it } from 'vitest'
import { screen, within } from '@testing-library/react'
import { renderWithProviders } from '@/test/renderWithProviders'
import { server } from '@/test/mswServer'
import { ModelsPage } from '@/routes/recorder/ModelsPage'
import {
  recorderErrorHandler,
  recorderModelsHandler,
  recorderUnreachableHandler,
} from '@/recorder/mocks/handlers'
import { REAL_MODELS } from '@/recorder/mocks/fixtures'
import type { RecorderModelsResponse } from '@/recorder/recorder.types'

function withModels(patch: Partial<RecorderModelsResponse>) {
  server.use(recorderModelsHandler({ ...REAL_MODELS, ...patch }))
}

async function tableNamed(name: RegExp | string) {
  return screen.findByRole('table', { name })
}

describe('ModelsPage — status block', () => {
  it('leads with the fact that nothing is reading the footage', async () => {
    renderWithProviders(<ModelsPage />)

    expect(await screen.findByText('Nothing is reading the footage')).toBeInTheDocument()
    // The recorder's own sentence, verbatim — not a paraphrase.
    expect(
      screen.getByText('no model is configured, so no footage is being described'),
    ).toBeInTheDocument()
    expect(screen.getByText('no_model_configured')).toBeInTheDocument()
    // And it must not be mistaken for "the recorder stopped recording".
    expect(screen.getByText(/Recording continues; only description is off/i)).toBeInTheDocument()
  })

  it('does not claim nothing is reading the footage once a model is ready', async () => {
    withModels({
      status: { ...REAL_MODELS.status, ready: true, vision: true, cadence: 'interval' },
    })
    renderWithProviders(<ModelsPage />)

    expect(await screen.findByText(/footage is being described/i)).toBeInTheDocument()
    expect(screen.queryByText('Nothing is reading the footage')).not.toBeInTheDocument()
  })

  it('warns that a text-only model cannot see', async () => {
    renderWithProviders(<ModelsPage />)
    expect(await screen.findByText(/configured model cannot accept images/i)).toBeInTheDocument()
  })

  it('stops warning about images once the configured model can see them', async () => {
    withModels({ status: { ...REAL_MODELS.status, vision: true } })
    renderWithProviders(<ModelsPage />)

    // Wait for the payload to land before asserting on an absence.
    await screen.findByRole('table', { name: /Local models/ })
    expect(screen.queryByText(/configured model cannot accept images/i)).not.toBeInTheDocument()
  })

  it('renders the cadence as words and marks an unused interval as unused', async () => {
    renderWithProviders(<ModelsPage />)

    expect(await screen.findByText('Only when asked')).toBeInTheDocument()
    expect(screen.getByText(/not in use/)).toBeInTheDocument()
    expect(screen.queryByText('off')).not.toBeInTheDocument()
  })

  it('drops the "not in use" caveat when the timer cadence actually uses the interval', async () => {
    withModels({ status: { ...REAL_MODELS.status, cadence: 'interval' } })
    renderWithProviders(<ModelsPage />)

    expect(await screen.findByText('On a timer')).toBeInTheDocument()
    expect(screen.queryByText(/not in use/)).not.toBeInTheDocument()
  })

  it('reports frames staying on the machine when they do', async () => {
    renderWithProviders(<ModelsPage />)
    const tile = (await screen.findByText('frames leave this machine')).parentElement!
    expect(within(tile).getByText('No')).toBeInTheDocument()
    expect(within(tile).queryByText('Yes')).not.toBeInTheDocument()
  })

  it('reports frames leaving the machine when they do', async () => {
    withModels({ status: { ...REAL_MODELS.status, offsite: true } })
    renderWithProviders(<ModelsPage />)

    const tile = (await screen.findByText('frames leave this machine')).parentElement!
    expect(within(tile).getByText('Yes')).toBeInTheDocument()
    expect(within(tile).queryByText('No')).not.toBeInTheDocument()
  })
})

describe('ModelsPage — local stack', () => {
  it('lists every installed local model with its real size and family', async () => {
    renderWithProviders(<ModelsPage />)

    const table = await tableNamed(/Local models/)
    const rows = within(table).getAllByRole('row').slice(1)
    expect(rows).toHaveLength(3)

    const first = within(rows[0]!).getAllByRole('cell')
    expect(first[0]).toHaveTextContent('qwen2.5vl:3b')
    expect(first[1]).toHaveTextContent('3.2 GB')
    expect(first[1]).not.toHaveTextContent('3200627168')
    expect(first[2]).toHaveTextContent('sees images')
    expect(first[3]).toHaveTextContent('qwen25vl')
  })

  it('marks a text-only local model as such rather than letting it look usable', async () => {
    renderWithProviders(<ModelsPage />)

    const table = await tableNamed(/Local models/)
    const gemmaRow = within(table)
      .getAllByRole('row')
      .find((row) => row.textContent?.includes('gemma2:latest'))!
    expect(within(gemmaRow).getByText('text only')).toBeInTheDocument()
    expect(within(gemmaRow).queryByText('sees images')).not.toBeInTheDocument()
    // gemma2 is installed but not declared by this build's catalog.
    expect(within(gemmaRow).getByText('not declared')).toBeInTheDocument()
  })

  it('surfaces the measured runtime notes, including a recorded prompt violation', async () => {
    renderWithProviders(<ModelsPage />)

    await tableNamed(/Local models/)
    expect(screen.getByText(/MEASURED PROMPT VIOLATION/)).toBeInTheDocument()
    expect(screen.getByText(REAL_MODELS.catalog_note!)).toBeInTheDocument()
  })

  it('says so when no local runtime answers', async () => {
    withModels({ local: [], local_error: 'connection refused on 127.0.0.1:11434' })
    renderWithProviders(<ModelsPage />)

    expect(
      await screen.findByText(/connection refused on 127.0.0.1:11434/),
    ).toBeInTheDocument()
    expect(screen.getByText(/reports no local model installed/i)).toBeInTheDocument()
  })
})

describe('ModelsPage — offsite models', () => {
  it('shows per-MTok pricing for every frontier model', async () => {
    renderWithProviders(<ModelsPage />)

    const table = await tableNamed(/Offsite models/)
    const opusRow = within(table)
      .getAllByRole('row')
      .find((row) => row.textContent?.includes('claude-opus-5'))!
    const cells = within(opusRow).getAllByRole('cell')
    expect(cells[0]).toHaveTextContent('Claude Opus 5')
    expect(cells[2]).toHaveTextContent('$5.00')
    expect(cells[3]).toHaveTextContent('$25.00')
    expect(cells[4]).toHaveTextContent('strongest scene reading')
  })

  it('says the offsite path has no key, and marks each model unusable', async () => {
    renderWithProviders(<ModelsPage />)

    expect(await screen.findByText(/No API key is present/i)).toBeInTheDocument()
    expect(
      screen.getByText(/what they would cost, not what is being spent/i),
    ).toBeInTheDocument()
    const table = await tableNamed(/Offsite models/)
    expect(within(table).getAllByText('no key')).toHaveLength(REAL_MODELS.frontier.length)
  })

  it('changes the warning, and drops the per-row marks, once a key is present', async () => {
    withModels({ frontier_credentialed: true })
    renderWithProviders(<ModelsPage />)

    expect(await screen.findByText(/An API key is present/i)).toBeInTheDocument()
    expect(screen.queryByText(/No API key is present/i)).not.toBeInTheDocument()
    const table = await tableNamed(/Offsite models/)
    expect(within(table).queryByText('no key')).not.toBeInTheDocument()
  })

  it('renders the recorder’s own note about frames leaving the machine', async () => {
    renderWithProviders(<ModelsPage />)
    expect(await screen.findByText(REAL_MODELS.frontier_note)).toBeInTheDocument()
  })
})

describe('ModelsPage — perception tiers', () => {
  it('renders each tier with its real build status, none of them as nominal', async () => {
    renderWithProviders(<ModelsPage />)

    const table = await tableNamed(/Perception tiers/)
    const rows = within(table).getAllByRole('row').slice(1)
    expect(rows).toHaveLength(REAL_MODELS.tiers!.length)
    expect(within(table).getAllByText('NOT_BUILT')).toHaveLength(6)
    expect(within(table).getAllByText('PARTIAL')).toHaveLength(1)
    expect(
      within(table).getByText(/No detector runs. Nothing counts people/),
    ).toBeInTheDocument()
  })
})

describe('ModelsPage — a recorder that reports less', () => {
  it('still renders the parts it was given when the optional blocks are absent', async () => {
    server.use(
      recorderModelsHandler({
        status: REAL_MODELS.status,
        local: REAL_MODELS.local,
        frontier: REAL_MODELS.frontier,
        frontier_credentialed: false,
        frontier_note: REAL_MODELS.frontier_note,
      }),
    )
    renderWithProviders(<ModelsPage />)

    // The load-bearing parts survive...
    expect(await screen.findByText('Nothing is reading the footage')).toBeInTheDocument()
    expect(await tableNamed(/Local models/)).toBeInTheDocument()
    expect(await tableNamed(/Offsite models/)).toBeInTheDocument()
    // ...and the blocks with no data are simply absent, not invented.
    expect(screen.queryByRole('table', { name: /Perception tiers/ })).not.toBeInTheDocument()
    expect(screen.queryByText('where inference runs')).not.toBeInTheDocument()
    expect(screen.queryByText(/What the recorder has measured/)).not.toBeInTheDocument()
  })
})

describe('ModelsPage — recorder unavailable', () => {
  it('does not let a failed request read as an all-clear', async () => {
    server.use(recorderUnreachableHandler('/models'))
    renderWithProviders(<ModelsPage />)

    expect(
      await screen.findByText(/The recorder is unreachable/i, undefined, { timeout: 5000 }),
    ).toBeInTheDocument()
    expect(screen.getByText(/do not read the absence of a warning below as an all-clear/i)).toBeInTheDocument()
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
  })

  it('reports an error response in the recorder’s own words', async () => {
    server.use(recorderErrorHandler('/models', 503, 'model registry unavailable'))
    renderWithProviders(<ModelsPage />)

    expect(
      await screen.findByText(/model registry unavailable/i, undefined, { timeout: 5000 }),
    ).toBeInTheDocument()
  })
})
