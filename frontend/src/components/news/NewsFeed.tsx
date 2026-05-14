import { useMemo, useState } from 'react'

import { useAppStore } from '../../store/useAppStore'
import { Skeleton } from '../ui/Skeleton'
import { useMarketData } from '../../hooks/useMarketData'
import { NewsCard, type NewsItem } from './NewsCard'
import { SentimentBadge } from './SentimentBadge'

type SortMode = 'recent_desc' | 'score_asc' | 'score_desc'
type SentimentFilter = 'all' | 'bullish' | 'bearish' | 'neutral'

const ALL_CATEGORIES: NewsItem['category'][] = [
  'earnings',
  'product',
  'management',
  'policy',
  'competition',
  'market',
]

function overallSentimentScore(items: NewsItem[]) {
  if (items.length === 0) return 0
  const sum = items.reduce((acc, it) => acc + (it.sentiment_score ?? 0), 0)
  return sum / items.length
}

function classifyOverall(score: number): NewsItem['sentiment'] {
  if (score > 0.15) return 'bullish'
  if (score < -0.15) return 'bearish'
  return 'neutral'
}

export function NewsFeed(props: { ticker: string }) {
  const setRange = useAppStore((s) => s.setRange)
  const setSidebarNav = useAppStore((s) => s.setSidebarNav)

  const ticker = props.ticker?.trim().toUpperCase()
  const { data, isLoading, error } = useMarketData(ticker || null, '3mo')

  const [sentiment, setSentiment] = useState<SentimentFilter>('all')
  const [sort, setSort] = useState<SortMode>('recent_desc')
  const [categoryOpen, setCategoryOpen] = useState(false)
  const [categories, setCategories] = useState<Set<NewsItem['category']>>(new Set())
  const [limit, setLimit] = useState(20)

  const items: NewsItem[] = useMemo(() => {
    const raw = (data?.news ?? []) as any[]
    return raw
      .filter(Boolean)
      .map((n) => ({
        date: n.date,
        title: n.title,
        source: n.source,
        category: n.category,
        sentiment: n.sentiment,
        sentiment_score: n.sentiment_score,
        impact: n.impact,
        url: n.url,
        summary: n.summary,
      })) as NewsItem[]
  }, [data?.news])

  const filtered = useMemo(() => {
    let out = items

    if (sentiment !== 'all') out = out.filter((i) => i.sentiment === sentiment)

    if (categories.size > 0) out = out.filter((i) => categories.has(i.category))

    out = [...out]
    if (sort === 'recent_desc') {
      out.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
    } else if (sort === 'score_asc') {
      out.sort((a, b) => a.sentiment_score - b.sentiment_score)
    } else if (sort === 'score_desc') {
      out.sort((a, b) => b.sentiment_score - a.sentiment_score)
    }

    return out
  }, [categories, items, sentiment, sort])

  const overall = useMemo(() => overallSentimentScore(filtered), [filtered])
  const overallClass = classifyOverall(overall)

  const shown = filtered.slice(0, limit)

  return (
    <div className="flex h-full flex-col">
      <div className="sticky top-0 z-10 mb-3 rounded-lg border border-border bg-bg-card p-3">
        <div className="flex flex-wrap items-center gap-2">
          <div className="font-data text-sm text-text">
            <span className="text-cyan">{ticker || '—'}</span>{' '}
            <span className="text-text-dim">News</span>
          </div>

          <div className="ml-auto flex flex-wrap items-center gap-2">
            <div className="flex items-center gap-1 rounded-md border border-border bg-bg-surface p-1">
              <button
                onClick={() => setSentiment('all')}
                className={
                  sentiment === 'all'
                    ? 'font-data rounded px-2 py-1 text-xs text-cyan glow-cyan'
                    : 'font-data rounded px-2 py-1 text-xs text-text-muted hover:bg-bg-card'
                }
              >
                All
              </button>
              <button
                onClick={() => setSentiment('bullish')}
                className={
                  sentiment === 'bullish'
                    ? 'font-data rounded px-2 py-1 text-xs text-bull glow-cyan'
                    : 'font-data rounded px-2 py-1 text-xs text-text-muted hover:bg-bg-card'
                }
              >
                Bull 📈
              </button>
              <button
                onClick={() => setSentiment('bearish')}
                className={
                  sentiment === 'bearish'
                    ? 'font-data rounded px-2 py-1 text-xs text-bear glow-cyan'
                    : 'font-data rounded px-2 py-1 text-xs text-text-muted hover:bg-bg-card'
                }
              >
                Bear 📉
              </button>
              <button
                onClick={() => setSentiment('neutral')}
                className={
                  sentiment === 'neutral'
                    ? 'font-data rounded px-2 py-1 text-xs text-neutral glow-cyan'
                    : 'font-data rounded px-2 py-1 text-xs text-text-muted hover:bg-bg-card'
                }
              >
                Neutral
              </button>
            </div>

            <div className="relative">
              <button
                onClick={() => setCategoryOpen((v) => !v)}
                className="rounded-md border border-border bg-bg-surface px-3 py-2 text-xs text-text-muted hover:text-text"
              >
                Category {categories.size ? `(${categories.size})` : '(All)'}
              </button>

              {categoryOpen ? (
                <div className="absolute right-0 mt-2 w-56 rounded-lg border border-border bg-bg-surface p-2">
                  <button
                    onClick={() => setCategories(new Set())}
                    className="mb-2 w-full rounded-md border border-border bg-bg-card px-2 py-1 text-left text-xs text-text-muted hover:text-text"
                  >
                    All categories
                  </button>

                  <div className="space-y-1">
                    {ALL_CATEGORIES.map((c) => {
                      const checked = categories.has(c)
                      return (
                        <label
                          key={c}
                          className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1 text-xs text-text-muted hover:bg-bg-card"
                        >
                          <input
                            type="checkbox"
                            checked={checked}
                            onChange={() => {
                              setCategories((prev) => {
                                const next = new Set(prev)
                                if (next.has(c)) next.delete(c)
                                else next.add(c)
                                return next
                              })
                            }}
                          />
                          <span className="font-data">{c.toUpperCase()}</span>
                        </label>
                      )
                    })}
                  </div>
                </div>
              ) : null}
            </div>

            <select
              value={sort}
              onChange={(e) => setSort(e.target.value as SortMode)}
              className="rounded-md border border-border bg-bg-surface px-3 py-2 text-xs text-text-muted outline-none focus:border-cyan"
            >
              <option value="recent_desc">Recent ↓</option>
              <option value="score_asc">Score ↑</option>
              <option value="score_desc">Score ↓</option>
            </select>
          </div>
        </div>

        <div className="mt-3 rounded-md border border-border bg-bg px-3 py-2 text-sm text-text-muted">
          {filtered.length} articles · Overall sentiment:{' '}
          <SentimentBadge
            sentiment={overallClass}
            score={overall}
            showScore
          />{' '}
          · <span className="text-text-dim">FinBERT powered 🤖</span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        {isLoading ? (
          <div className="space-y-3 rounded-xl border border-border-subtle bg-bg-secondary p-4">
            <Skeleton className="h-5 w-2/3" rounded="md" />
            <Skeleton className="h-24 w-full" rounded="lg" />
            <Skeleton className="h-24 w-full" rounded="lg" />
            <p className="text-center text-[length:var(--text-12)] text-text-muted">Loading news…</p>
          </div>
        ) : null}

        {error ? (
          <div className="rounded-lg border border-border bg-bg-card p-4 text-sm text-bear">
            {error.message}
          </div>
        ) : null}

        {!isLoading && !error && shown.length === 0 ? (
          <div className="rounded-lg border border-border bg-bg-card p-4 text-sm text-text-muted">
            No news found for {ticker || 'this ticker'}. Check your API keys.
          </div>
        ) : null}

        <div className="space-y-3">
          {shown.map((it, idx) => (
            <NewsCard
              key={`${it.url}-${idx}`}
              item={it}
              onDateClick={(date) => {
                setRange(date, date)
                setSidebarNav('dashboard')
              }}
            />
          ))}
        </div>

        {filtered.length > shown.length ? (
          <div className="mt-4 flex justify-center">
            <button
              onClick={() => setLimit((n) => n + 20)}
              className="rounded-md border border-border bg-bg-card px-4 py-2 text-sm text-text-muted hover:text-text hover:glow-cyan"
            >
              Load 20 more
            </button>
          </div>
        ) : null}
      </div>
    </div>
  )
}

