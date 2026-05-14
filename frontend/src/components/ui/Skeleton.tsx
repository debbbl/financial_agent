import { clsx } from 'clsx'

const roundedMap = {
  sm: 'rounded',
  md: 'rounded-md',
  lg: 'rounded-lg',
} as const

export function Skeleton({
  className,
  rounded = 'md',
}: {
  className?: string
  rounded?: keyof typeof roundedMap
}) {
  return (
    <div
      className={clsx(
        'animate-pulse bg-bg-tertiary/80',
        roundedMap[rounded],
        className
      )}
    />
  )
}
