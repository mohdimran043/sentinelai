import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import { useSessionStore } from '@/store/session'
import { cn } from '@/lib/cn'
import { RECORDER_SECTIONS } from '@/routes/recorder/sections'

const navLinkClass = ({ isActive }: { isActive: boolean }) =>
  cn(
    'flex shrink-0 items-center gap-2 whitespace-nowrap border-l-2 px-[18px] py-[9px] text-left font-mono text-[12.5px] font-medium uppercase tracking-[0.08em]',
    isActive
      ? 'border-signal bg-[#0f1610] text-signal'
      : 'border-transparent text-dim hover:bg-panel-2 hover:text-fg',
  )

/**
 * Rail group heading. This console reads two entirely separate products — the
 * AI engine and the recorder appliance — and an operator needs to know which
 * one a number came from. Naming the groups is cheaper than badging every tile.
 */
function RailGroup({ children }: { children: string }) {
  return (
    <div className="eyebrow shrink-0 self-center px-[18px] py-2 md:mt-3 md:self-auto md:py-1.5">
      {children}
    </div>
  )
}

/** Marks a section whose route exists but whose screen is not built yet. */
function PendingTag() {
  return (
    <span className="rounded-sm border border-line-2 bg-panel-2 px-[5px] py-px font-mono text-[9px] font-normal tracking-[0.08em] text-dimmer">
      soon
    </span>
  )
}

export function AppShell() {
  const operatorName = useSessionStore((state) => state.operatorName)
  const signOut = useSessionStore((state) => state.signOut)
  const navigate = useNavigate()

  return (
    <div className="grid min-h-full grid-cols-1 md:grid-cols-[var(--spacing-rail)_1fr]">
      <nav
        aria-label="Sections"
        className="flex flex-row gap-0.5 overflow-x-auto border-b border-line bg-panel p-2.5 md:sticky md:top-0 md:h-screen md:flex-col md:overflow-x-visible md:overflow-y-auto md:border-b-0 md:border-r md:p-[18px_0]"
      >
        <div className="hidden px-[18px] pb-5 md:block">
          <div className="font-mono text-[13px] font-semibold uppercase tracking-[0.18em]">
            <b className="text-signal">Sentinel</b>AI
          </div>
          <div className="eyebrow mt-1.5">Ingest &amp; response</div>
        </div>

        <RailGroup>AI engine</RailGroup>
        <NavLink to="/dashboard" className={navLinkClass}>
          Dashboard
        </NavLink>
        <NavLink to="/site-map" className={navLinkClass}>
          Site map
        </NavLink>

        <RailGroup>Recorder</RailGroup>
        {RECORDER_SECTIONS.map((section) => (
          <NavLink key={section.path} to={`/recorder/${section.path}`} className={navLinkClass}>
            {section.label}
            {section.built ? null : <PendingTag />}
          </NavLink>
        ))}

        <div className="mt-auto hidden border-t border-line px-[18px] pt-4 md:block">
          <div className="eyebrow mb-1.5">Operator</div>
          <div className="text-[13px] text-fg">{operatorName ?? 'Unknown'}</div>
          <button
            type="button"
            onClick={() => {
              signOut()
              navigate('/login')
            }}
            className="mt-2 font-mono text-[11.5px] uppercase tracking-[0.07em] text-dim underline-offset-2 hover:text-fg hover:underline"
          >
            Sign out
          </button>
        </div>
      </nav>
      <main className="max-w-[1400px] p-5 pb-16 md:p-[28px_32px_64px]">
        <Outlet />
      </main>
    </div>
  )
}
