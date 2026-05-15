import { lazy, Suspense, type ReactNode } from 'react'

import { AnalysisDashboard } from './AnalysisDashboard'
import { useAppStore } from '../../store/useAppStore'
import { SkeletonPage } from './SkeletonPage'
import { WelcomeDashboard } from './WelcomeDashboard'

const ChatFullPage = lazy(async () => {
  const m = await import('./ChatFullPage')
  return { default: m.ChatFullPage }
})

const NewsFullPage = lazy(async () => {
  const m = await import('./NewsFullPage')
  return { default: m.NewsFullPage }
})

const PortfolioPage = lazy(async () => {
  const m = await import('./PortfolioPage')
  return { default: m.PortfolioPage }
})

export function MainContent() {
  const activeTicker = useAppStore((s) => s.activeTicker)
  const activeTab = useAppStore((s) => s.activeTab)

  if (!activeTicker) {
    if (activeTab === 'portfolio') {
      return (
        <Suspense fallback={<SkeletonPage />}>
          <div className="animate-fade-in">
            <PortfolioPage />
          </div>
        </Suspense>
      )
    }
    if (activeTab === 'news') {
      return (
        <Suspense fallback={<SkeletonPage />}>
          <div className="animate-fade-in">
            <NewsFullPage />
          </div>
        </Suspense>
      )
    }
    if (activeTab === 'chat') {
      return (
        <Suspense fallback={<SkeletonPage />}>
          <div className="animate-fade-in">
            <ChatFullPage />
          </div>
        </Suspense>
      )
    }
    return (
      <div className="animate-fade-in">
        <WelcomeDashboard />
      </div>
    )
  }

  let panel: ReactNode
  switch (activeTab) {
    case 'chart':
    case 'dashboard':
      panel = <AnalysisDashboard />
      break
    case 'chat':
      panel = <ChatFullPage />
      break
    case 'news':
      panel = <NewsFullPage />
      break
    case 'portfolio':
      panel = <PortfolioPage />
      break
    default:
      panel = <AnalysisDashboard />
      break
  }

  const suspenseKey = `${activeTicker}-${activeTab}`

  return (
    <Suspense key={suspenseKey} fallback={<SkeletonPage />}>
      <div className="animate-fade-in min-h-0">{panel}</div>
    </Suspense>
  )
}
