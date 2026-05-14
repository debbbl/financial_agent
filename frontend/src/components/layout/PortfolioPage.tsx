import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Plus, Trash2 } from 'lucide-react'
import { useMemo, useState } from 'react'

import { portfolioApi, sessionApi } from '../../api/client'
import { useAppStore } from '../../store/useAppStore'

function isValidTicker(raw: string) {
  return /^[A-Z]{2,6}$/.test(raw)
}

export function PortfolioPage() {
  const [draft, setDraft] = useState('')
  const setTicker = useAppStore((s) => s.setTicker)
  const setSession = useAppStore((s) => s.setSession)
  const setTab = useAppStore((s) => s.setTab)
  const qc = useQueryClient()

  const q = useQuery({
    queryKey: ['portfolio'],
    queryFn: () => portfolioApi.list(),
    staleTime: 15_000,
  })

  const addMut = useMutation({
    mutationFn: (ticker: string) => portfolioApi.add(ticker),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['portfolio'] }),
  })

  const removeMut = useMutation({
    mutationFn: (ticker: string) => portfolioApi.remove(ticker),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['portfolio'] }),
  })

  const items = useMemo(() => {
    const data = q.data
    if (Array.isArray(data)) return data as Array<{ ticker: string; notes?: string }>
    if (data?.items && Array.isArray(data.items)) return data.items as Array<{ ticker: string }>
    return []
  }, [q.data])

  async function openTicker(ticker: string) {
    setTicker(ticker)
    const data = await sessionApi.create(ticker)
    setSession(data.session_id ?? data.id ?? null)
    setTab('chart')
  }

  const normalized = draft.toUpperCase().slice(0, 6)

  return (
    <div className="card max-w-3xl space-y-6 p-6">
      <div>
        <h1 className="font-display text-xl font-semibold text-text1">Portfolio</h1>
        <p className="mt-1 text-sm text-text3">Tickers you track across the workspace.</p>
      </div>

      <div className="flex flex-wrap gap-2">
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value.toUpperCase())}
          onKeyDown={(e) => {
            if (e.key !== 'Enter') return
            if (!isValidTicker(normalized)) return
            addMut.mutate(normalized)
            setDraft('')
          }}
          placeholder="Add ticker"
          className="min-w-[120px] flex-1 rounded-lg border border-border bg-surface2 px-3 py-2 font-mono text-sm text-text1 outline-none focus:border-primary focus:ring-2 focus:ring-primary/15"
          spellCheck={false}
        />
        <button
          type="button"
          disabled={!isValidTicker(normalized) || addMut.isPending}
          onClick={() => {
            if (!isValidTicker(normalized)) return
            addMut.mutate(normalized)
            setDraft('')
          }}
          className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-primary-dark disabled:opacity-40"
        >
          <Plus className="h-4 w-4" />
          Add
        </button>
      </div>

      <ul className="divide-y divide-border rounded-xl border border-border bg-surface">
        {items.length === 0 ? (
          <li className="px-4 py-8 text-center text-sm text-text3">No positions yet. Add a ticker above.</li>
        ) : (
          items.map((p) => (
            <li key={p.ticker} className="flex items-center justify-between gap-3 px-4 py-3">
              <button
                type="button"
                onClick={() => void openTicker(p.ticker)}
                className="font-mono text-sm font-semibold text-primary hover:underline"
              >
                {p.ticker}
              </button>
              <button
                type="button"
                title="Remove"
                onClick={() => removeMut.mutate(p.ticker)}
                className="rounded-lg p-2 text-text3 transition hover:bg-surface2 hover:text-bear"
              >
                <Trash2 className="h-4 w-4" />
              </button>
            </li>
          ))
        )}
      </ul>
    </div>
  )
}
