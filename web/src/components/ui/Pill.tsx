import type { HTMLAttributes } from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '@/lib/cn'
import type { Tone } from '@/lib/severity'

const pillVariants = cva(
  'inline-block whitespace-nowrap rounded-sm border px-[9px] py-[2.5px] font-mono text-[10.5px] font-medium uppercase leading-[1.5] tracking-[0.1em]',
  {
    variants: {
      tone: {
        nominal: 'border-signal-d bg-[#0f1610] text-signal',
        caution: 'border-[#574000] bg-[#171304] text-caution',
        breach: 'border-[#5c231b] bg-[#170e0d] text-breach animate-throb',
        inert: 'border-line-2 bg-panel-2 text-dim',
      } satisfies Record<Tone, string>,
    },
    defaultVariants: {
      tone: 'inert',
    },
  },
)

export interface PillProps extends HTMLAttributes<HTMLSpanElement>, VariantProps<typeof pillVariants> {}

export function Pill({ className, tone, ...props }: PillProps) {
  return <span className={cn(pillVariants({ tone }), className)} {...props} />
}
