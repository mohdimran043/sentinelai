import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import './index.css'
import { App } from '@/App'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Every screen decides its own polling/staleness; do not silently retry
      // network errors forever by default.
      retry: false,
      refetchOnWindowFocus: false,
    },
  },
})

/**
 * The event feed and history are mocked for the whole of this slice (Go is
 * Phase 1C — see events/eventClient.ts), so the MSW worker starts unconditionally
 * rather than only in dev. If `VITE_EVENTS_API_URL` is set, a real service is
 * assumed to be there instead and the worker is skipped.
 */
async function enableMocking(): Promise<void> {
  if (import.meta.env.VITE_EVENTS_API_URL) return
  const { worker } = await import('@/events/mocks/browser')
  await worker.start({
    onUnhandledRequest: 'bypass',
    quiet: true,
  })
}

void enableMocking().then(() => {
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <QueryClientProvider client={queryClient}>
        <App />
      </QueryClientProvider>
    </StrictMode>,
  )
})
