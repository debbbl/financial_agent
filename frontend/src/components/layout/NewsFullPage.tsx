import { useAppStore } from '../../store/useAppStore'
import { NewsFeed } from '../news/NewsFeed'

export function NewsFullPage() {
  const ticker = useAppStore((s) => s.activeTicker)
  return (
    <section className="card min-h-[calc(100vh-8rem)] p-4">
      {ticker ? (
        <NewsFeed ticker={ticker} />
      ) : (
        <div className="flex min-h-[50vh] flex-col items-center justify-center gap-3 px-4 text-center">
          <p className="font-display text-lg font-medium text-text-primary">News</p>
          <p className="max-w-md text-sm text-text-muted">
            Choose a symbol from the search bar above (or the home screen) to load headlines and
            sentiment for that ticker.
          </p>
        </div>
      )}
    </section>
  )
}
