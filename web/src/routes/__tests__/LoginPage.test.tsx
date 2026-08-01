import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { renderWithProviders } from '@/test/renderWithProviders'
import { LoginPage } from '@/routes/LoginPage'
import { useSessionStore } from '@/store/session'

describe('LoginPage', () => {
  beforeEach(() => {
    useSessionStore.setState({ operatorName: null })
    sessionStorage.clear()
  })

  afterEach(() => {
    useSessionStore.setState({ operatorName: null })
    sessionStorage.clear()
  })

  it('is honest that there is no real authentication yet', () => {
    renderWithProviders(<LoginPage />)
    expect(screen.getByRole('status')).toHaveTextContent(/there is no account system yet/i)
  })

  it('rejects an empty submission with a message, not a silent no-op', async () => {
    const user = userEvent.setup()
    renderWithProviders(<LoginPage />)
    await user.click(screen.getByRole('button', { name: /enter console/i }))
    expect(await screen.findByRole('alert')).toHaveTextContent(/enter a name/i)
    expect(useSessionStore.getState().operatorName).toBeNull()
  })

  it('signs in with a trimmed operator name and stores it', async () => {
    const user = userEvent.setup()
    renderWithProviders(<LoginPage />)
    await user.type(screen.getByLabelText(/operator name/i), '  J. Rivera  ')
    await user.click(screen.getByRole('button', { name: /enter console/i }))
    expect(useSessionStore.getState().operatorName).toBe('J. Rivera')
  })
})
