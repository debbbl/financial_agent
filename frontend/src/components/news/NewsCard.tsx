import { clsx } from 'clsx'

import { SentimentBadge } from './SentimentBadge'

export type NewsItem = {
  date: string
  title: string
  source: string
  category: 'earnings' | 'product' | 'management' | 'policy' | 'competition' | 'market'
  sentiment: 'bullish' | 'bearish' | 'neutral'
  sentiment_score: number
  impact: 'high' | 'medium' | 'low'
  url: string
  summary?: string
}

function timeAgo(date: string) {
  const d = new Date(date)
  const diff = Date.now() - d.getTime()
  const s = Math.max(0, Math.floor(diff / 1000))
  if (s < 60) return `${s}s ago`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ago`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h ago`
  const days = Math.floor(h / 24)
  return `${days}d ago`
}

function impactColor(impact: NewsItem['impact']) {
  if (impact === 'high') return 'bg-bear'
  if (impact === 'medium') return 'bg-violet'
  return 'bg-neutral'
}

function categoryLabel(cat: NewsItem['category']) {
  return cat.toUpperCase()
}

export function NewsCard(props: { item: NewsItem; onDateClick?: (date: string) => void }) {
  const it = props.item
  const confidence = Math.min(1, Math.max(0, Math.abs(it.sentiment_score)))

  const barColor =
    it.sentiment === 'bullish'
      ? 'bg-bull'
      : it.sentiment === 'bearish'
        ? 'bg-bear'
        : 'bg-neutral'

  return (
    <article className="rounded-xl border border-border bg-bg-card p-4">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <SentimentBadge sentiment={it.sentiment} score={it.sentiment_score} showScore />

        <span className="rounded-full border border-border bg-bg-surface px-2 py-0.5 font-data text-[11px] text-text-muted">
          {categoryLabel(it.category)}
        </span>

        <span
          className="inline-flex items-center gap-1 text-text-muted"
          title={`Impact: ${it.impact}`}
        >
          <span className={clsx('h-2 w-2 rounded-full', impactColor(it.impact))} />
          <span className="font-data text-[11px]">{it.impact.toUpperCase()}</span>
        </span>

        <span className="ml-auto text-text-dim">{timeAgo(it.date)}</span>
      </div>

      <div className="mt-2">
        <a
          href={it.url}
          target="_blank"
          rel="noreferrer"
          className="text-sm font-semibold text-text hover:text-cyan"
        >
          {it.title}
        </a>
      </div>

      <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-text-muted">
        <button
          className="hover:text-cyan"
          onClick={() => props.onDateClick?.(it.date)}
          title="Jump to chart date"
        >
          {it.source}
        </button>

        <div className="flex flex-1 items-center gap-2">
          <div className="h-1 w-full overflow-hidden rounded bg-border/60">
            <div
              className={clsx('h-1', barColor)}
              style={{ width: `${Math.round(confidence * 100)}%` }}
              title={`FinBERT confidence: ${(confidence * 100).toFixed(0)}%`}
            />
          </div>
          <span className="font-data text-[11px] text-text-dim">
            {(confidence * 100).toFixed(0)}%
          </span>
        </div>
      </div>
    </article>
  )
}

