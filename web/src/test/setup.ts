import '@testing-library/jest-dom/vitest'
import { afterAll, afterEach, beforeAll } from 'vitest'
import { server } from '@/events/mocks/server'

// The mocked event API (MSW) runs for every test; the live engine API is not
// mocked globally on purpose — individual tests stub `fetch` (or leave it
// failing) so each test controls what "the engine is unreachable" looks like.
beforeAll(() => server.listen({ onUnhandledRequest: 'bypass' }))
afterEach(() => server.resetHandlers())
afterAll(() => server.close())
