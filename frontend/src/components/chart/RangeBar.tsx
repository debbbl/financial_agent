import { clsx } from 'clsx'

import type { ChartContext } from '../../api/client'
import type { MarketNewsEvent, OhlcvBar } from '../../hooks/useMarketData'
import { useAppStore } from '../../store/useAppStore'

function impactRank(impact?: string): number {
  const s = (impact ?? '').toLowerCase()
  if (s === 'high') return 0
  if (s === 'medium') return 1
  return 2
}

/** Build chart_context from in-memory OHLCV/news (no extra fetch). */
export function buildChartContextForRange(
  ticker: string,
  start: string,
  end: string,
  ohlcv: OhlcvBar[],
  news: MarketNewsEvent[]
): ChartContext | null {
  const bars = ohlcv
    .filter((b) => b.date >= start && b.date <= end)
    .sort((a, b) => a.date.localeCompare(b.date))
  if (bars.length === 0) return null

  const open_price = bars[0].open
  const close_price = bars[bars.length - 1].close
  const high = Math.max(...bars.map((b) => b.high))
  const low = Math.min(...bars.map((b) => b.low))
  const price_change_pct =
    open_price !== 0 ? ((close_price - open_price) / open_price) * 100 : 0

  const inRange = news.filter((n) => n?.date && n.date >= start && n.date <= end)
  const sorted = [...inRange].sort(
    (a, b) => impactRank(a.impact) - impactRank(b.impact) || a.date.localeCompare(b.date)
  )
  const top_news: ChartContext['top_news'] = sorted.slice(0, 10).map((n) => ({
    date: n.date,
    title: n.title,
    sentiment: n.sentiment ?? null,
    impact: n.impact ?? null,
  }))

  return {
    ticker: ticker.toUpperCase(),
    from_date: start,
    to_date: end,
    price_change_pct,
    open_price,
    close_price,
    high,
    low,
    news_count: inRange.length,
    top_news,
  }
}

export function RangeBar(props: {
  tradingDays: number
  ohlcv: OhlcvBar[]
  news: MarketNewsEvent[]
}) {
  const ticker = useAppStore((s) => s.activeTicker)
  const { start, end } = useAppStore((s) => s.chartRange)
  const clearRange = useAppStore((s) => s.clearRange)
  const setPanelTab = useAppStore((s) => s.setPanelTab)
  const setChatDraft = useAppStore((s) => s.setChatDraft)
  const setPendingChartContext = useAppStore((s) => s.setPendingChartContext)

  if (!start || !end) return null

  const days = props.tradingDays

  return (
    <div className="sticky bottom-0 z-10 mt-3 rounded-xl border border-border-subtle bg-bg-tertiary/80 px-3 py-2.5 backdrop-blur-sm">
      <div className="flex flex-wrap items-center gap-2 text-[length:var(--text-13)]">
        <span className="font-data text-accent">{ticker ?? '—'}</span>
        <span className="text-text-muted">·</span>
        <span className={clsx('font-data', 'text-text-primary')}>
          <span className="text-accent">{start}</span> → <span className="text-accent">{end}</span>
        </span>
        <span className="text-text-muted">·</span>
        <span className="text-text-secondary">{days} trading days</span>

        <div className="ml-auto flex items-center gap-2">
          <button
            type="button"
            onClick={() => {
              const sym = ticker ?? ''
              const ctx =
                sym && start && end
                  ? buildChartContextForRange(sym, start, end, props.ohlcv, props.news)
                  : null
              setPendingChartContext(ctx)
              const prompt = `Analyze ${sym} from ${start} to ${end}.

- Summarize price action, trend, volatility.
- Identify key support/resistance and any breakout/breakdown.
- If there were notable news events, relate them to moves.
- Provide a concise outlook + risk notes.`
              setChatDraft(prompt.trim())
              setPanelTab('ai')
            }}
            className="rounded-lg border border-accent/30 bg-accent/10 px-3 py-1.5 text-[length:var(--text-12)] font-medium text-text-primary transition hover:glow-cyan"
          >
            Analyze with AI ↗
          </button>
          <button
            type="button"
            onClick={() => clearRange()}
            className="rounded-lg border border-border-subtle bg-transparent px-3 py-1.5 text-[length:var(--text-12)] text-text-secondary hover:bg-bg-secondary hover:text-text-primary"
          >
            ✕ Clear
          </button>
        </div>
      </div>
    </div>
  )
}
