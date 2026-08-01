import type { ReactNode } from 'react'
import { Navigate, useLocation } from 'react-router-dom'
import { useSessionStore } from '@/store/session'

/**
 * A routing convenience, not a security boundary — there is no backend
 * authentication in this slice (see `store/session.ts`). This only decides which
 * screen to show first.
 */
export function RequireSession({ children }: { children: ReactNode }) {
  const operatorName = useSessionStore((state) => state.operatorName)
  const location = useLocation()

  if (!operatorName) {
    return <Navigate to="/login" replace state={{ from: location }} />
  }

  return <>{children}</>
}
