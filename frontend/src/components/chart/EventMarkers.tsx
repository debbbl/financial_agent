import type { SeriesMarker, Time } from 'lightweight-charts'

import type { MarketNewsEvent, OhlcvBar } from '../../hooks/useMarketData'

const CATEGORY_PRIORITY: Record<string, number> = {
  earnings: 100,
  product: 80,
  management: 60,
  policy: 40,
  competition: 20,
  market: 10,
}

function categoryRank(cat: string | undefined): number {
  if (!cat) return 0
  return CATEGORY_PRIORITY[cat.toLowerCase()] ?? 0
}

/** One marker per calendar day; highest-priority category wins. */
export function buildEventMarkers(
  news: MarketNewsEvent[],
  ohlcv: OhlcvBar[],
  barTime: (bar: OhlcvBar) => Time
): SeriesMarker<Time>[] {
  const available = new Set(ohlcv.map((b) => b.date))
  const byDate = new Map<string, MarketNewsEvent[]>()

  for (const n of news) {
    if (!n?.date || !available.has(n.date)) continue
    const arr = byDate.get(n.date) ?? []
    arr.push(n)
    byDate.set(n.date, arr)
  }

  const markers: SeriesMarker<Time>[] = []

  for (const [date, items] of byDate) {
    let best = items[0]
    for (const it of items) {
      if (categoryRank(it.category) > categoryRank(best.category)) best = it
    }

    const bar = ohlcv.find((b) => b.date === date)
    if (!bar) continue

    const cat = (best.category ?? 'market').toLowerCase()
    let shape: SeriesMarker<Time>['shape'] = 'circle'
    let color = '#9CA3AF'
    let size = 1

    switch (cat) {
      case 'earnings':
        shape = 'arrowUp'
        color = '#2563EB'
        size = 1
        break
      case 'product':
        shape = 'circle'
        color = '#10B981'
        break
      case 'management':
        shape = 'square'
        color = '#F59E0B'
        break
      case 'policy':
        shape = 'arrowDown'
        color = '#EA580C'
        break
      case 'competition':
        shape = 'circle'
        color = '#7C3AED'
        break
      case 'market':
      default:
        shape = 'square'
        color = '#94A3B8'
        break
    }

    markers.push({
      time: barTime(bar),
      position: 'belowBar',
      color,
      shape,
      size,
      text: '',
    })
  }

  markers.sort((a, b) => String(a.time).localeCompare(String(b.time)))
  return markers
}
