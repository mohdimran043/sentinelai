import { type FormEvent, useId, useState } from 'react'
import { Navigate, useLocation, useNavigate } from 'react-router-dom'
import { useSessionStore } from '@/store/session'
import { Panel } from '@/components/ui/Panel'
import { Label } from '@/components/ui/Label'
import { Input } from '@/components/ui/Input'
import { Button } from '@/components/ui/Button'
import { Notice } from '@/components/ui/Notice'

export function LoginPage() {
  const operatorName = useSessionStore((state) => state.operatorName)
  const signIn = useSessionStore((state) => state.signIn)
  const navigate = useNavigate()
  const location = useLocation()
  const inputId = useId()
  const [name, setName] = useState('')
  const [touched, setTouched] = useState(false)

  if (operatorName) {
    const redirectTo = (location.state as { from?: { pathname: string } } | null)?.from?.pathname
    return <Navigate to={redirectTo ?? '/dashboard'} replace />
  }

  const trimmed = name.trim()
  const error = touched && trimmed.length === 0 ? 'Enter a name to continue.' : null

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setTouched(true)
    if (trimmed.length === 0) return
    signIn(trimmed)
    navigate('/dashboard', { replace: true })
  }

  return (
    <div className="flex min-h-full items-center justify-center bg-void p-5">
      <div className="w-full max-w-[380px]">
        <div className="mb-6 text-center">
          <div className="font-mono text-[15px] font-semibold uppercase tracking-[0.18em]">
            <b className="text-signal">Sentinel</b>AI
          </div>
          <p className="lede mt-1.5 text-dim">Ingest &amp; response console</p>
        </div>

        <Panel>
          <h1>Sign in</h1>
          <p className="lede">Identify yourself to the console.</p>

          <Notice tone="inert" className="mb-1">
            There is no account system yet — authentication (JWT) is Phase 1C. This
            just labels who is watching the dashboard; it does not check a
            password or grant any permission the console wouldn&apos;t otherwise
            give you.
          </Notice>

          <form onSubmit={handleSubmit} noValidate>
            <Label htmlFor={inputId}>Operator name</Label>
            <Input
              id={inputId}
              name="operatorName"
              autoComplete="name"
              placeholder="e.g. J. Rivera"
              value={name}
              onChange={(event) => setName(event.target.value)}
              aria-invalid={error ? true : undefined}
              aria-describedby={error ? `${inputId}-error` : undefined}
            />
            {error ? (
              <p id={`${inputId}-error`} className="err mt-1.5" role="alert">
                {error}
              </p>
            ) : null}

            <Button type="submit" className="mt-4 w-full">
              Enter console
            </Button>
          </form>
        </Panel>
      </div>
    </div>
  )
}
