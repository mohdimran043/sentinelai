import type { HTMLAttributes } from 'react'
import { cn } from '@/lib/cn'

export interface NoticeProps extends HTMLAttributes<HTMLDivElement> {
  tone?: 'caution' | 'breach' | 'inert'
}

const toneClass: Record<NonNullable<NoticeProps['tone']>, string> = {
  caution: 'border-caution bg-[#14100320] text-[#e9cf94]',
  breach: 'border-breach bg-[#1a0e0d] text-[#f0b3ab]',
  inert: 'border-line-2 bg-panel-2 text-dim',
}

export function Notice({ tone = 'caution', className, ...props }: NoticeProps) {
  return (
    <div
      role={tone === 'breach' ? 'alert' : 'status'}
      className={cn(
        'rounded-sm border-l-2 px-3.5 py-[11px] text-[12.5px]',
        toneClass[tone],
        className,
      )}
      {...props}
    />
  )
}
