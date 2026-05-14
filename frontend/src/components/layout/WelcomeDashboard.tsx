import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Brain, LineChart, Sparkles } from 'lucide-react'
import { useCallback, useMemo } from 'react'

import { marketApi, sessionApi } from '../../api/client'
import { rememberTickerSelection, TickerSearchBar } from '../shared/TickerSearchBar'
import { useAppStore } from '../../store/useAppStore'

const POPULAR = [
  { ticker: 'AAPL', sector: 'Technology' },
  { ticker: 'NVDA', sector: 'Semiconductors' },
  { ticker: 'TSLA', sector: 'Automotive' },
  { ticker: 'MSFT', sector: 'Technology' },
  { ticker: 'META', sector: 'Communication' },
  { ticker: 'GOOGL', sector: 'Technology' },
  { ticker: 'AMZN', sector: 'Consumer' },
  { ticker: 'AMD', sector: 'Semiconductors' },
] as const

function greeting() {
  const h = new Date().getHours()
  if (h < 12) return 'morning'
  if (h < 17) return 'afternoon'
  return 'evening'
}

function relTime(ts: string | number | Date) {
  const d = new Date(ts)
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

export function WelcomeDashboard() {
  const qc = useQueryClient()
  const setTicker = useAppStore((s) => s.setTicker)
  const setSession = useAppStore((s) => s.setSession)
  const setTab = useAppStore((s) => s.setTab)
  const chartPeriod = useAppStore((s) => s.chartPeriod)

  const sessionsQuery = useQuery({
    queryKey: ['sessions'],
    queryFn: () => sessionApi.list(),
    staleTime: 10_000,
  })

  const sessions = useMemo(() => {
    const data = sessionsQuery.data
    type Row = {
      session_id?: string
      id?: string
      ticker?: string
      created_at?: string
      updated_at?: string
    }
    return Array.isArray(data) ? (data as Row[]) : ([] as Row[])
  }, [sessionsQuery.data])

  const recentFour = sessions.slice(0, 4)

  const selectTicker = useCallback(
    async (ticker: string) => {
      const sym = ticker.trim().toUpperCase()
      void qc.prefetchQuery({
        queryKey: ['market', sym, chartPeriod, null, true],
        queryFn: () => marketApi.getData(sym, chartPeriod, null, true),
      })
      setTicker(sym)
      setTab('chart')
      try {
        const data = await sessionApi.create(sym)
        setSession(data.session_id ?? data.id ?? null)
      } catch {
        /* session optional for UI */
      }
    },
    [chartPeriod, qc, setSession, setTab, setTicker]
  )

  async function resumeSession(row: (typeof sessions)[number]) {
    const id = (row.session_id ?? row.id) as string | undefined
    const ticker = (row.ticker ?? '').toUpperCase()
    if (!ticker) return
    void qc.prefetchQuery({
      queryKey: ['market', ticker, chartPeriod, null, true],
      queryFn: () => marketApi.getData(ticker, chartPeriod, null, true),
    })
    setTicker(ticker)
    setSession(id ?? null)
    setTab('chart')
  }

  async function selectPopular(ticker: string) {
    rememberTickerSelection(ticker)
    await selectTicker(ticker)
  }

  return (
    <div className="mx-auto max-w-5xl space-y-12 px-2 py-8 md:py-12">
      <header className="text-center">
        <h1 className="font-display text-3xl font-semibold tracking-tight text-text1 md:text-4xl">
          Good {greeting()}, Analyst.
        </h1>
        <p className="mt-3 text-sm text-text3 md:text-base">
          Select a ticker to begin AI-powered analysis
        </p>
        <div className="mt-8 flex justify-center">
          <TickerSearchBar
            size="large"
            onSelect={(ticker) => {
              void selectTicker(ticker)
            }}
          />
        </div>
      </header>

      <section>
        <h2 className="mb-4 text-center font-display text-lg font-semibold text-text1">Popular stocks</h2>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {POPULAR.map(({ ticker, sector }) => (
            <button
              key={ticker}
              type="button"
              onClick={() => void selectPopular(ticker)}
              className="card group flex flex-col items-start gap-2 rounded-xl border border-border p-4 text-left transition duration-250 ease-smooth hover:scale-[1.02] hover:shadow-hover"
            >
              <div className="flex w-full items-center gap-3">
                <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-surface2 font-mono text-xs font-semibold text-text3 ring-1 ring-border">
                  {ticker.slice(0, 2)}
                </span>
                <div className="min-w-0">
                  <div className="font-mono text-sm font-semibold text-text1">{ticker}</div>
                  <div className="truncate text-2xs text-text4">{sector}</div>
                </div>
              </div>
            </button>
          ))}
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-3">
        {[
          {
            icon: Sparkles,
            title: 'Multi-agent analysis',
            desc: 'Coordinated specialists debate fundamentals, risk, and positioning.',
          },
          {
            icon: Brain,
            title: 'FinBERT sentiment',
            desc: 'News and social tone scored so you see narrative vs. price.',
          },
          {
            icon: LineChart,
            title: 'Pattern memory',
            desc: 'Sessions and ranges carry forward for sharper follow-up questions.',
          },
        ].map(({ icon: Icon, title, desc }) => (
          <div
            key={title}
            className="card rounded-2xl border border-border bg-gradient-card p-5 shadow-card transition hover:shadow-md"
          >
            <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-lg bg-primary-light text-primary">
              <Icon className="h-5 w-5" aria-hidden />
            </div>
            <h3 className="font-display text-base font-semibold text-text1">{title}</h3>
            <p className="mt-2 text-sm leading-relaxed text-text3">{desc}</p>
          </div>
        ))}
      </section>

      {recentFour.length > 0 ? (
        <section>
          <h2 className="mb-4 font-display text-lg font-semibold text-text1">Continue where you left off</h2>
          <div className="grid gap-3 sm:grid-cols-2">
            {recentFour.map((s, idx) => {
              const id = (s.session_id ?? s.id ?? `s-${idx}`) as string
              const ticker = (s.ticker ?? '—').toUpperCase()
              const ts = s.created_at ?? s.updated_at ?? ''
              return (
                <button
                  key={id}
                  type="button"
                  onClick={() => void resumeSession(s)}
                  className="card flex flex-col items-start rounded-xl border border-border p-4 text-left transition hover:shadow-md"
                >
                  <span className="font-mono text-base font-semibold text-primary">{ticker}</span>
                  <span className="mt-1 text-xs text-text3">{ts ? relTime(ts) : 'Recent session'}</span>
                </button>
              )
            })}
          </div>
        </section>
      ) : null}
    </div>
  )
}
