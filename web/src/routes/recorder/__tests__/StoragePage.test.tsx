import { describe, expect, it } from 'vitest'
import { screen, within } from '@testing-library/react'
import { renderWithProviders } from '@/test/renderWithProviders'
import { server } from '@/test/mswServer'
import { StoragePage } from '@/routes/recorder/StoragePage'
import {
  recorderErrorHandler,
  recorderStorageHandler,
  recorderUnreachableHandler,
} from '@/recorder/mocks/handlers'
import { REAL_STORAGE } from '@/recorder/mocks/fixtures'
import type { RecorderUntrackedResponse } from '@/recorder/recorder.types'

function withStorage(body: RecorderUntrackedResponse) {
  server.use(recorderStorageHandler(body))
}

async function tableNamed(name: RegExp | string) {
  return screen.findByRole('table', { name })
}

describe('StoragePage — totals', () => {
  it('renders the real small-scale totals through formatBytes/formatCount, not raw digits', async () => {
    renderWithProviders(<StoragePage />)

    const totalTile = (await screen.findByText('untracked, total')).parentElement!
    expect(within(totalTile).getByText('84 B')).toBeInTheDocument()
    expect(within(totalTile).queryByText('84')).not.toBeInTheDocument()

    const countTile = screen.getByText('untracked files').parentElement!
    expect(within(countTile).getByText('3')).toBeInTheDocument()
  })

  it('renders sensibly at gigabyte scale, not as an unreadable byte count', async () => {
    withStorage({
      ...REAL_STORAGE,
      total_bytes: 5_234_000_000,
      cameras: REAL_STORAGE.cameras.map((camera) =>
        camera.camera_id === 'room_4b' ? { ...camera, total_bytes: 6_000_000_000 } : camera,
      ),
    })
    renderWithProviders(<StoragePage />)

    const totalTile = (await screen.findByText('untracked, total')).parentElement!
    expect(within(totalTile).getByText('5.2 GB')).toBeInTheDocument()
    const table = await tableNamed(/Untracked footage/)
    expect(within(table).getByText('6.0 GB')).toBeInTheDocument()
    expect(screen.queryByText('5234000000')).not.toBeInTheDocument()
  })

  it('counts cameras that actually have untracked footage, not just cameras reporting', async () => {
    renderWithProviders(<StoragePage />)

    // Only room_4b has count > 0 among the 4 real cameras.
    const tile = (await screen.findByText('cameras with untracked footage')).parentElement!
    expect(within(tile).getByText('1')).toBeInTheDocument()
  })
})

describe('StoragePage — per-camera table', () => {
  it('lists every camera, marking empty ones distinctly from the one with footage', async () => {
    renderWithProviders(<StoragePage />)

    const table = await tableNamed(/Untracked footage/)
    const rows = within(table).getAllByRole('row').slice(1)
    expect(rows).toHaveLength(4)

    const emptyRow = rows.find((row) => row.textContent?.includes('corridor_1'))!
    expect(within(emptyRow).getByText('none')).toBeInTheDocument()

    const populatedRow = rows.find((row) => row.textContent?.includes('room_4b'))!
    expect(within(populatedRow).queryByText('none')).not.toBeInTheDocument()
  })

  it('marks a camera whose sample is truncated as partial, not complete', async () => {
    withStorage({
      ...REAL_STORAGE,
      cameras: REAL_STORAGE.cameras.map((camera) =>
        camera.camera_id === 'room_4b' ? { ...camera, truncated: true } : camera,
      ),
    })
    renderWithProviders(<StoragePage />)

    const table = await tableNamed(/Untracked footage/)
    const row = within(table)
      .getAllByRole('row')
      .find((r) => r.textContent?.includes('room_4b'))!
    expect(within(row).getByText(/partial/)).toBeInTheDocument()
    expect(within(row).queryByText(/complete/)).not.toBeInTheDocument()
  })
})

describe('StoragePage — sample files', () => {
  it('lists the real sample files for the camera that has them', async () => {
    renderWithProviders(<StoragePage />)

    expect(await screen.findByText('room_4b', { selector: 'summary span' })).toBeInTheDocument()
    expect(
      screen.getByText(/room_4b_1785502196892950204\.mp4/),
    ).toBeInTheDocument()
  })

  it('says when more files exist beyond the shown sample', async () => {
    withStorage({
      ...REAL_STORAGE,
      cameras: REAL_STORAGE.cameras.map((camera) =>
        camera.camera_id === 'room_4b' ? { ...camera, truncated: true } : camera,
      ),
    })
    renderWithProviders(<StoragePage />)

    expect(await screen.findByText(/more exist/)).toBeInTheDocument()
  })

  it('renders no sample section at all when nothing has untracked footage', async () => {
    withStorage({
      cameras: REAL_STORAGE.cameras.map((camera) => ({
        ...camera,
        count: 0,
        total_bytes: 0,
        sample: null,
        truncated: false,
      })),
      note: REAL_STORAGE.note,
      total_bytes: 0,
      total_count: 0,
    })
    renderWithProviders(<StoragePage />)

    const tile = (await screen.findByText('untracked, total')).parentElement!
    expect(within(tile).getByText('0 B')).toBeInTheDocument()
    expect(screen.queryByText('sample files')).not.toBeInTheDocument()
  })
})

describe('StoragePage — recorder unavailable', () => {
  it('does not let a failed request read as an all-clear', async () => {
    server.use(recorderUnreachableHandler('/storage/untracked'))
    renderWithProviders(<StoragePage />)

    expect(
      await screen.findByText(/recorder appliance is not reachable/i, undefined, { timeout: 5000 }),
    ).toBeInTheDocument()
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
  })

  it('reports an error response in the recorder’s own words', async () => {
    server.use(recorderErrorHandler('/storage/untracked', 503, 'storage index unavailable'))
    renderWithProviders(<StoragePage />)

    expect(
      await screen.findByText(/storage index unavailable/i, undefined, { timeout: 5000 }),
    ).toBeInTheDocument()
  })
})
