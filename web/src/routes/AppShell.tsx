import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import { useSessionStore } from '@/store/session'
import { cn } from '@/lib/cn'
import { RECORDER_SECTIONS } from '@/routes/recorder/sections'
import { useUnseenAlertCount } from '@/alerts/AlertRail'

const navLinkClass = ({ isActive }: { isActive: boolean }) =>
  cn(
    'flex shrink-0 items-center gap-2 whitespace-nowrap border-l-2 px-[18px] py-[9px] text-left font-mono text-[12.5px] font-medium uppercase tracking-[0.08em]',
    isActive
      ? 'border-signal bg-[#0f1610] text-signal'
      : 'border-transparent text-dim hover:bg-panel-2 hover:text-fg',
  )

/**
 * Rail group heading.
 *
 * There is one group. The rail used to split into "AI engine" and "Recorder"
 * because the console reads two separate products and an operator needed to
 * know which one a number came from — but the recorder sections that survived
 * (`routes/recorder/sections.ts`) are precisely the ones with *no* engine
 * equivalent, so the split was drawing a line the operator never had to act
 * on while making them cross it to get from "what did the engine see" to
 * "what did the recorder keep". The `/recorder/*` URL prefix still records
 * which system a screen talks to; see `App.tsx`.
 */
function RailGroup({ children }: { children: string }) {
  return (
    <div className="eyebrow shrink-0 self-center px-[18px] py-2 md:mt-3 md:self-auto md:py-1.5">
      {children}
    </div>
  )
}

/**
 * How many alerts nobody has looked at, on the nav rail.
 *
 * Counts `active` only. An alert somebody has acknowledged is being dealt with,
 * and keeping the badge lit through that is how a badge stops meaning anything.
 * Renders nothing at zero rather than a `0` — an empty badge is a thing to read
 * and dismiss on every glance.
 */
function UnseenBadge() {
  const count = useUnseenAlertCount()
  if (count === 0) return null
  return (
    <span
      className="ml-auto rounded-sm border border-[#5c231b] bg-[#170e0d] px-[5px] py-px font-mono text-[10px] text-breach"
      aria-label={`${count} alerts nobody has acknowledged`}
    >
      {count}
    </span>
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
          Command centre
        </NavLink>
        {/*
          "Alert queue", not "Alerts". The recorder's own alerts screen is gone
          (see `routes/recorder/sections.ts`), so this is the only alerts
          destination in the rail — but the name stays, because it is the more
          accurate one: this is a work queue whose rows are acknowledged and
          closed, not a list you read.
        */}
        <NavLink to="/alerts" className={navLinkClass}>
          Alert queue
          <UnseenBadge />
        </NavLink>
        <NavLink to="/people" className={navLinkClass}>
          People
        </NavLink>
        <NavLink to="/site-map" className={navLinkClass}>
          Site map
        </NavLink>
        {/* In the rail rather than only on the dashboard: adding a camera is the one
            action an operator takes before there is anything on the wall to click. */}
        <NavLink to="/cameras/new" className={navLinkClass}>
          Add a camera
        </NavLink>
        <NavLink to="/ai-system" className={navLinkClass}>
          AI system
        </NavLink>

        {/* Under the engine's own heading rather than a "Recorder" group of their
            own. They still read the recorder appliance, over `/recorder/*`. */}
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
