import { useQuery } from '@tanstack/react-query'
import { clsx } from 'clsx'
import { Home, Layers, LineChart, Newspaper } from 'lucide-react'
import { useMemo } from 'react'

import { sessionApi } from '../../api/client'
import { useAppStore } from '../../store/useAppStore'

type SessionRow = {
  session_id?: string
  id?: string
  ticker?: string
  created_at?: string
  updated_at?: string
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

export function Sidebar() {
  const setTicker = useAppStore((s) => s.setTicker)
  const setSession = useAppStore((s) => s.setSession)
  const activeTab = useAppStore((s) => s.activeTab)
  const setTab = useAppStore((s) => s.setTab)

  const sessionsQuery = useQuery({
    queryKey: ['sessions'],
    queryFn: () => sessionApi.list(),
    refetchInterval: 30_000,
    staleTime: 10_000,
  })

  const sessionsItems = useMemo((): SessionRow[] => {
    const data = sessionsQuery.data
    return Array.isArray(data) ? (data as SessionRow[]) : []
  }, [sessionsQuery.data])

  const recentSessions = sessionsItems.slice(0, 10)

  return (
    <aside
      className={clsx(
        'sticky top-14 hidden h-[calc(100dvh-56px)] w-56 shrink-0 flex-col border-r border-border bg-bg-surface md:flex'
      )}
    >
      <div className="flex flex-col gap-1 border-b border-border p-3">
        <p className="text-xs font-medium text-text-muted">Navigate</p>
        <button
          type="button"
          onClick={() => {
            setSession(null)
            setTicker(null)
            setTab('chart')
          }}
          className={clsx(
            'flex items-center gap-2 rounded-lg px-2 py-2 text-left text-sm transition',
            activeTab === 'chart' || activeTab === 'dashboard'
              ? 'bg-cyan/10 text-cyan'
              : 'text-text-muted hover:bg-bg-card hover:text-text'
          )}
        >
          <Home className="h-4 w-4 shrink-0" />
          Home
        </button>
        <button
          type="button"
          onClick={() => setTab('portfolio')}
          className={clsx(
            'flex items-center gap-2 rounded-lg px-2 py-2 text-left text-sm transition',
            activeTab === 'portfolio'
              ? 'bg-cyan/10 text-cyan'
              : 'text-text-muted hover:bg-bg-card hover:text-text'
          )}
        >
          <Layers className="h-4 w-4 shrink-0" />
          Portfolio
        </button>
        <button
          type="button"
          onClick={() => setTab('chart')}
          className={clsx(
            'flex items-center gap-2 rounded-lg px-2 py-2 text-left text-sm transition',
            activeTab === 'chart' || activeTab === 'dashboard'
              ? 'bg-cyan/10 text-cyan'
              : 'text-text-muted hover:bg-bg-card hover:text-text'
          )}
        >
          <LineChart className="h-4 w-4 shrink-0" />
          Chart
        </button>
        <button
          type="button"
          onClick={() => setTab('news')}
          className={clsx(
            'flex items-center gap-2 rounded-lg px-2 py-2 text-left text-sm transition',
            activeTab === 'news'
              ? 'bg-cyan/10 text-cyan'
              : 'text-text-muted hover:bg-bg-card hover:text-text'
          )}
        >
          <Newspaper className="h-4 w-4 shrink-0" />
          News
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-3">
        <p className="mb-2 text-xs font-medium uppercase tracking-wide text-text-muted">
          Recent sessions
        </p>
        <div className="space-y-1">
          {recentSessions.map((s: SessionRow, idx: number) => {
            const id = (s.session_id ?? s.id ?? `row-${idx}`) as string
            const ticker = (s.ticker ?? '').toUpperCase()
            const ts = s.created_at ?? s.updated_at ?? ''
            return (
              <button
                key={id}
                type="button"
                onClick={() => {
                  if (ticker) setTicker(ticker)
                  setSession(id)
                  setTab('chart')
                }}
                className="w-full rounded-lg border border-transparent px-2 py-2 text-left text-sm text-text-muted transition-[filter,background-color,border-color,color] duration-200 hover:border-border hover:bg-bg-card hover:text-text hover:brightness-110"
              >
                <span className="font-data text-cyan">{ticker || '—'}</span>
                {ts ? (
                  <span className="ml-1 text-xs text-text-dim">· {relTime(ts)}</span>
                ) : null}
              </button>
            )
          })}
          {recentSessions.length === 0 ? (
            <p className="rounded-lg border border-border bg-bg-card px-2 py-3 text-xs text-text-muted">
              No sessions yet. Search for a ticker above.
            </p>
          ) : null}
        </div>
      </div>
    </aside>
  )
}
