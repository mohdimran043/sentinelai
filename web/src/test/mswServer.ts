import { setupServer } from 'msw/node'
import { eventHandlers } from '@/events/mocks/handlers'
import { recorderHandlers } from '@/recorder/mocks/handlers'

/**
 * One MSW node server for the whole suite, holding the default handlers for
 * every API the console talks to. It lives under `test/` rather than inside
 * either product's folder because it is shared plumbing — putting it in
 * `events/mocks/` (where it used to be) made it look like an events-only
 * concern the moment a second product's handlers arrived.
 *
 * Tests override per case with `server.use(...)`; `test/setup.ts` resets after
 * each one.
 */
export const server = setupServer(...eventHandlers, ...recorderHandlers)
