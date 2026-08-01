import { setupServer } from 'msw/node'
import { recorderHandlers } from '@/recorder/mocks/handlers'

/**
 * One MSW node server for the whole suite, holding the default handlers for
 * every API the console talks to over MSW. The AI engine's own calls are
 * mocked at the `engineClient` module boundary per test instead (see
 * `routes/__tests__/CameraPage.test.tsx`), not through MSW — there used to be
 * a mocked events product here too (a stand-in for a Go service that didn't
 * exist), retired now that the AI engine's `/cameras/{id}/events` route is
 * real.
 *
 * Tests override per case with `server.use(...)`; `test/setup.ts` resets after
 * each one.
 */
export const server = setupServer(...recorderHandlers)
