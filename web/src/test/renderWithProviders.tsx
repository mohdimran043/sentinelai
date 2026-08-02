import type { ReactElement, ReactNode } from 'react'
import { render } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

export function renderWithProviders(ui: ReactElement, { route = '/' }: { route?: string } = {}) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, refetchOnWindowFocus: false },
      mutations: { retry: false },
    },
  })

  // Providers go through RTL's `wrapper` option (rather than nesting them
  // directly around `ui`) so that `rerender(newUi)` re-applies them too.
  // Without this, calling the returned `rerender` with a bare element drops
  // the QueryClientProvider/MemoryRouter context instead of just swapping
  // props, which breaks any component under test that reads that context.
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={[route]}>{children}</MemoryRouter>
      </QueryClientProvider>
    )
  }

  return render(ui, { wrapper: Wrapper })
}
