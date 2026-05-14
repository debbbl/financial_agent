import { clsx } from 'clsx'

import { useMarketData } from '../../hooks/useMarketData'
import { useAppStore } from '../../store/useAppStore'

export function LivePriceBadge({ ticker }: { ticker: string | null }) {
  const sym = ticker?.trim().toUpperCase() ?? null
  const chartPeriod = useAppStore((s) => s.chartPeriod)
  const { data, isLoading } = useMarketData(sym, chartPeriod)

  if (!sym) return null

  if (isLoading && data == null) {
    return (
      <span className="font-data text-xs tabular-nums text-text-muted">Loading…</span>
    )
  }

  const price = data?.current_price
  const pct = data?.price_change_pct

  if (typeof price !== 'number') {
    return (
      <span className="font-data text-xs text-text-muted" title={sym}>
        —
      </span>
    )
  }

  const pctLabel =
    typeof pct === 'number'
      ? `${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%`
      : null

  return (
    <div
      className="flex items-baseline gap-2 rounded-md border border-border bg-bg-surface/80 px-2 py-1"
      title={sym}
    >
      <span className="font-data text-sm font-medium tabular-nums text-text">
        {price.toFixed(2)}
      </span>
      {pctLabel ? (
        <span
          className={clsx(
            'font-data text-xs tabular-nums',
            (pct ?? 0) >= 0 ? 'text-bull' : 'text-bear'
          )}
        >
          {pctLabel}
        </span>
      ) : null}
    </div>
  )
}
