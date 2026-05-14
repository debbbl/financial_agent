import { clsx } from 'clsx'

import { LivePriceBadge } from '../shared/LivePriceBadge'
import { useAppStore } from '../../store/useAppStore'
import { TickerInput } from './TickerInput'

export function TopBar() {
  const activeTab = useAppStore((s) => s.activeTab)
  const setTab = useAppStore((s) => s.setTab)
  const activeTicker = useAppStore((s) => s.activeTicker)

  return (
    <header className="fixed inset-x-0 top-0 z-20 h-14 overflow-visible border-b border-border bg-bg/80 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-7xl items-center gap-3 px-4">
        <div className="flex items-center gap-2">
          <div className="font-data text-xs text-cyan">
            <span className="rounded-md border border-border bg-bg-surface px-2 py-1">
              FA
            </span>
          </div>
        </div>

        <div className="flex flex-1 items-center justify-center gap-3">
          <TickerInput />
          <LivePriceBadge ticker={activeTicker} />
        </div>

        <div className="flex items-center gap-2">
          <nav className="flex items-center gap-5">
            {(['chart', 'chat', 'news'] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={clsx(
                  'relative h-10 px-1 text-sm font-medium text-text-muted transition hover:text-text',
                  activeTab === t && 'text-text'
                )}
              >
                {t[0].toUpperCase() + t.slice(1)}
                <span
                  className={clsx(
                    'absolute -bottom-[7px] left-0 right-0 h-[2px] rounded-full bg-transparent transition',
                    activeTab === t && 'bg-cyan'
                  )}
                />
              </button>
            ))}
          </nav>
        </div>
      </div>
    </header>
  )
}

