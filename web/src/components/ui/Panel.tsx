import type { HTMLAttributes } from 'react'
import { cn } from '@/lib/cn'

export function Panel({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn('rounded-md border border-line bg-panel p-[18px]', className)}
      {...props}
    />
  )
}

export interface TileProps extends HTMLAttributes<HTMLDivElement> {
  /** Left accent bar tone, matching the reference's `.tile.ok/.caution/.breach`. */
  tone?: 'nominal' | 'caution' | 'breach' | 'inert'
}

const toneBar: Record<NonNullable<TileProps['tone']>, string> = {
  nominal: 'before:bg-signal',
  caution: 'before:bg-caution',
  breach: 'before:bg-breach',
  inert: 'before:bg-line-2',
}

export function Tile({ className, tone = 'inert', ...props }: TileProps) {
  return (
    <div
      className={cn(
        'relative rounded-md border border-line bg-panel p-[15px]',
        'before:absolute before:inset-y-0 before:left-0 before:w-[2px] before:content-[""]',
        toneBar[tone],
        className,
      )}
      {...props}
    />
  )
}
