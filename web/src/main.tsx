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

// Anomaly events now come from the AI engine's own `/cameras/{id}/events`
// route (see `api/engineClient.ts`'s `getCameraEvents`) via the same
// `/engine` dev proxy every other engine call uses, so no MSW browser worker
// is needed here — the mocked stand-in for a not-yet-built Go events service
// this bootstrap used to start has been retired along with that service's
// need to exist.
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
)
