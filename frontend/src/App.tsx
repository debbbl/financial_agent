import { useEffect, useRef } from 'react'

import { sessionApi } from './api/client'
import { MainContent } from './components/layout/MainContent'
import { Sidebar } from './components/layout/Sidebar'
import { TopBar } from './components/layout/TopBar'
import { useAppStore } from './store/useAppStore'

export function App() {
  const activeTicker = useAppStore((s) => s.activeTicker)
  const activeSessionId = useAppStore((s) => s.activeSessionId)
  const setSession = useAppStore((s) => s.setSession)

  const lastAutoTickerRef = useRef<string | null>(null)

  useEffect(() => {
    if (!activeTicker) return
    if (activeSessionId) return
    if (lastAutoTickerRef.current === activeTicker) return

    lastAutoTickerRef.current = activeTicker
    void sessionApi
      .create(activeTicker)
      .then((data) => setSession(data.session_id ?? data.id ?? null))
      .catch(() => {
        lastAutoTickerRef.current = null
      })
  }, [activeSessionId, activeTicker, setSession])

  return (
    <div className="min-h-screen bg-bg font-body text-text">
      <TopBar />

      <div className="flex pt-14">
        <Sidebar />

        <main className="min-h-[calc(100dvh-56px)] min-w-0 flex-1 overflow-y-auto">
          <div className="mx-auto max-w-7xl px-4 py-4">
            <MainContent />
          </div>
        </main>
      </div>
    </div>
  )
}
