import type { ReactNode } from 'react'

export interface PageHeaderProps {
  /** Small mono uppercase kicker above the title, e.g. "delivered facts". */
  eyebrow: string
  title: string
  /** One or two sentences saying what this screen is, and what it is not. */
  lede?: ReactNode
}

/** The reference's page opening: eyebrow, h1, lede — in that order, on every screen. */
export function PageHeader({ eyebrow, title, lede }: PageHeaderProps) {
  return (
    <header>
      <div className="eyebrow">{eyebrow}</div>
      <h1>{title}</h1>
      {lede ? <p className="lede">{lede}</p> : null}
    </header>
  )
}
