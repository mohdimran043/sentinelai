import { create } from 'zustand'
import { persist } from 'zustand/middleware'

/**
 * There is no authentication in this slice — JWT is Phase 1C (master-prompt §13,
 * spec §8). This store does not gate access to anything real; it only remembers
 * the operator label typed on the login screen so the rest of the shell has a
 * name to display, and lets the shell decide whether to show the login form or
 * the dashboard first. Nothing here should be read as a security boundary.
 */
interface SessionState {
  operatorName: string | null
  signIn: (operatorName: string) => void
  signOut: () => void
}

export const useSessionStore = create<SessionState>()(
  persist(
    (set) => ({
      operatorName: null,
      signIn: (operatorName: string) => set({ operatorName }),
      signOut: () => set({ operatorName: null }),
    }),
    {
      name: 'sentinelai.session-label',
      storage: {
        getItem: (name) => {
          const value = sessionStorage.getItem(name)
          return value ? JSON.parse(value) : null
        },
        setItem: (name, value) => sessionStorage.setItem(name, JSON.stringify(value)),
        removeItem: (name) => sessionStorage.removeItem(name),
      },
    },
  ),
)
