import { useQueryClient } from '@tanstack/react-query'

import { marketApi, sessionApi } from '../../api/client'
import { useAppStore } from '../../store/useAppStore'
import { rememberTickerSelection, TickerSearchBar } from '../shared/TickerSearchBar'

export function TickerInput() {
  const qc = useQueryClient()
  const chartPeriod = useAppStore((s) => s.chartPeriod)
  const setTicker = useAppStore((s) => s.setTicker)
  const setSession = useAppStore((s) => s.setSession)
  const setTab = useAppStore((s) => s.setTab)

  return (
    <div className="w-full max-w-md min-w-0">
      <TickerSearchBar
        onSelect={async (ticker) => {
          const sym = String(ticker).trim().toUpperCase()
          rememberTickerSelection(sym)
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
        }}
      />
    </div>
  )
}
