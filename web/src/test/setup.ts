import '@testing-library/jest-dom/vitest'
import { afterAll, afterEach, beforeAll } from 'vitest'
import { server } from '@/test/mswServer'

// Timestamp formatting renders in the operator's local zone, so the suite pins
// TZ=UTC in `vite.config.ts` (`test.env`). Without that, an assertion on a
// formatted `at_ns` would pass or fail depending on where CI happens to be.

// The mocked event API and the recorder API (MSW) run for every test; the live
// AI engine API is not mocked globally on purpose — individual tests stub
// `fetch` (or leave it failing) so each test controls what "the engine is
// unreachable" looks like.
beforeAll(() => server.listen({ onUnhandledRequest: 'bypass' }))
afterEach(() => server.resetHandlers())
afterAll(() => server.close())
