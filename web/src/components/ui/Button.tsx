import type { ButtonHTMLAttributes } from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '@/lib/cn'

const buttonVariants = cva(
  'inline-flex items-center justify-center gap-2 rounded-sm font-mono text-[12.5px] font-semibold uppercase tracking-[0.09em] transition-colors disabled:cursor-not-allowed disabled:opacity-45',
  {
    variants: {
      variant: {
        act: 'bg-signal text-void hover:brightness-110',
        quiet: 'border border-line-2 bg-transparent text-fg hover:bg-panel-2',
        live: 'bg-breach text-white hover:brightness-110',
      },
      size: {
        default: 'px-[17px] py-[9px]',
        small: 'px-[11px] py-[5px] text-[11px]',
      },
    },
    defaultVariants: {
      variant: 'act',
      size: 'default',
    },
  },
)

export interface ButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

export function Button({ className, variant, size, ...props }: ButtonProps) {
  return <button className={cn(buttonVariants({ variant, size }), className)} {...props} />
}
