import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type MutableRefObject,
} from 'react'

import type { IChartApi, ISeriesApi, MouseEventParams, Time } from 'lightweight-charts'

import type { MarketNewsEvent, OhlcvBar } from '../../hooks/useMarketData'

export type StatsRow = { open: number; high: number; low: number; close: number; volume: number }

type Props = {
  chartRef: MutableRefObject<IChartApi | null>
  candleSeriesRef: MutableRefObject<ISeriesApi<'Candlestick'> | null>
  chartContainerRef: MutableRefObject<HTMLDivElement | null>
  /** Chart + tooltip wrapper — mouseleave uses this so moving onto the tooltip does not dismiss it */
  interactionRootRef?: MutableRefObject<HTMLDivElement | null>
  chartReady: boolean
  news: MarketNewsEvent[]
  ohlcv: OhlcvBar[]
  ticker: string | null
  onStatsChange: (row: StatsRow | null) => void
}

function isBusinessDay(t: Time): t is { year: number; month: number; day: number } {
  return typeof t === 'object' && t !== null && 'year' in t
}

function businessDayToIso(t: { year: number; month: number; day: number }) {
  const mm = String(t.month).padStart(2, '0')
  const dd = String(t.day).padStart(2, '0')
  return `${t.year}-${mm}-${dd}`
}

export function timeToIsoDay(t: Time): string | null {
  if (typeof t === 'number') {
    return new Date(t * 1000).toISOString().slice(0, 10)
  }
  if (isBusinessDay(t)) return businessDayToIso(t)
  if (typeof t === 'string') return t.slice(0, 10)
  return null
}

function barTime(bar: OhlcvBar): Time {
  if (typeof bar.t === 'number') return bar.t as Time
  return bar.date as Time
}

function addCalendarDaysIso(iso: string, delta: number): string {
  const [y, m, d] = iso.split('-').map(Number)
  const dt = new Date(Date.UTC(y, m - 1, d))
  dt.setUTCDate(dt.getUTCDate() + delta)
  return dt.toISOString().slice(0, 10)
}

function fmtUsd(n: number) {
  return `$${n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
}

function fmtVol(v: number) {
  if (v >= 1e9) return `${(v / 1e9).toFixed(2)}B`
  if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`
  if (v >= 1e3) return `${(v / 1e3).toFixed(1)}K`
  return String(Math.round(v))
}

function sentimentLabel(n: MarketNewsEvent): 'BULL' | 'BEAR' | 'NEUT' {
  if (n.sentiment === 'bullish') return 'BULL'
  if (n.sentiment === 'bearish') return 'BEAR'
  if (n.sentiment === 'neutral') return 'NEUT'
  const score = n.sentiment_score ?? n.score
  if (typeof score === 'number') {
    if (score > 0.15) return 'BULL'
    if (score < -0.15) return 'BEAR'
  }
  return 'NEUT'
}

function badgeClass(s: 'BULL' | 'BEAR' | 'NEUT') {
  if (s === 'BULL') return 'text-green-400 bg-green-400/10 px-1.5 py-0.5 rounded text-[10px] font-mono'
  if (s === 'BEAR') return 'text-red-400 bg-red-400/10 px-1.5 py-0.5 rounded text-[10px] font-mono'
  return 'text-slate-400 bg-slate-400/10 px-1.5 py-0.5 rounded text-[10px] font-mono'
}

function collectNewsWindow(iso: string, byDay: Map<string, MarketNewsEvent[]>): MarketNewsEvent[] {
  const keys = [addCalendarDaysIso(iso, -1), iso, addCalendarDaysIso(iso, 1)]
  const out: MarketNewsEvent[] = []
  const seen = new Set<string>()
  for (const k of keys) {
    for (const n of byDay.get(k) ?? []) {
      const id = `${n.date}|${n.title}|${n.url ?? ''}`
      if (seen.has(id)) continue
      seen.add(id)
      out.push(n)
    }
  }
  return out
}

function dedupeNewsIds(items: MarketNewsEvent[]): MarketNewsEvent[] {
  const seen = new Set<string>()
  const out: MarketNewsEvent[] = []
  for (const n of items) {
    const id = `${n.date}|${n.title}|${n.url ?? ''}`
    if (seen.has(id)) continue
    seen.add(id)
    out.push(n)
  }
  return out
}

/** Calendar distance in days between two YYYY-MM-DD strings. */
function calendarDayDistance(aIso: string, bIso: string): number {
  const t0 = Date.parse(`${aIso.slice(0, 10)}T12:00:00Z`)
  const t1 = Date.parse(`${bIso.slice(0, 10)}T12:00:00Z`)
  if (Number.isNaN(t0) || Number.isNaN(t1)) return 99999
  return Math.abs(Math.round((t0 - t1) / 86400000))
}

/**
 * Prefer articles published on hovered day ±1. If APIs only returned recent dates
 * (e.g. last 48h), that window is empty for older bars — fall back to chart-range
 * headlines sorted by proximity to the hovered bar, then move-scoring picks one.
 */
function newsCandidatePool(
  iso: string,
  byDay: Map<string, MarketNewsEvent[]>,
  allNews: MarketNewsEvent[],
  ohlcv: OhlcvBar[]
): MarketNewsEvent[] {
  const windowed = collectNewsWindow(iso, byDay)
  if (windowed.length > 0) return windowed

  if (allNews.length === 0) return []

  const minD = ohlcv[0]?.date
  const maxD = ohlcv[ohlcv.length - 1]?.date
  let base =
    minD && maxD ? allNews.filter((n) => n.date >= minD && n.date <= maxD) : allNews
  if (base.length === 0) base = allNews

  const sorted = [...base].sort(
    (a, b) => calendarDayDistance(a.date, iso) - calendarDayDistance(b.date, iso)
  )
  return dedupeNewsIds(sorted).slice(0, 40)
}

function numericSentiment(n: MarketNewsEvent): number {
  const v = n.sentiment_score ?? n.score
  if (typeof v === 'number' && !Number.isNaN(v)) return Math.max(-1, Math.min(1, v))
  if (n.sentiment === 'bullish') return 0.55
  if (n.sentiment === 'bearish') return -0.55
  return 0
}

function impactWeight(n: MarketNewsEvent): number {
  const i = (n.impact ?? '').toLowerCase()
  if (i === 'high') return 3.5
  if (i === 'medium') return 2
  return 1
}

function categoryMoveWeight(n: MarketNewsEvent): number {
  const c = (n.category ?? '').toLowerCase()
  if (c === 'earnings') return 2.8
  if (c === 'product' || c === 'management' || c === 'policy') return 1.6
  if (c === 'competition' || c === 'market') return 1.1
  return 1
}

function scoreNewsForBarMove(n: MarketNewsEvent, hoveredIso: string, ohlc: StatsRow): number {
  const dayRet = ohlc.open !== 0 ? (ohlc.close - ohlc.open) / ohlc.open : 0
  const moveSign = dayRet === 0 ? 0 : dayRet > 0 ? 1 : -1
  const s = numericSentiment(n)
  const sentSign = s > 0.04 ? 1 : s < -0.04 ? -1 : 0

  let alignment = 0.15
  if (moveSign !== 0 && sentSign !== 0) {
    alignment = sentSign === moveSign ? 1.2 + Math.abs(s) * 2.2 : 0.2 + Math.abs(s) * 0.35
  } else if (moveSign !== 0 && sentSign === 0) {
    alignment = 0.45
  } else {
    alignment = 0.35 + Math.abs(s) * 0.8
  }

  const moveMag = Math.min(2.2, Math.abs(dayRet) * 28)
  let recency = 0.75
  if (n.date === hoveredIso) recency = 1.35
  else if (n.date === addCalendarDaysIso(hoveredIso, -1) || n.date === addCalendarDaysIso(hoveredIso, 1))
    recency = 0.95

  return recency * (alignment * (0.55 + moveMag) + impactWeight(n) * 1.15 + categoryMoveWeight(n) * 0.55)
}

function selectMoveRelatedNews(
  candidates: MarketNewsEvent[],
  hoveredIso: string,
  ohlc: StatsRow
): MarketNewsEvent[] {
  if (candidates.length === 0) return []
  const scored = candidates
    .map((n) => ({ n, score: scoreNewsForBarMove(n, hoveredIso, ohlc) }))
    .sort((a, b) => b.score - a.score)
  return [scored[0]!.n]
}

function faviconUrlFromArticle(url?: string): string | null {
  if (!url || !/^https?:\/\//i.test(url)) return null
  try {
    const host = new URL(url).hostname
    if (!host) return null
    return `https://icons.duckduckgo.com/ip3/${encodeURIComponent(host)}.ico`
  } catch {
    return null
  }
}

/** Prefer provider article image; else favicon from article URL host. */
function articleThumbnailSrc(n: MarketNewsEvent): { src: string; isArticleImage: boolean } | null {
  const raw = n.image_url?.trim()
  if (raw && /^https?:\/\//i.test(raw)) {
    return { src: raw, isArticleImage: true }
  }
  const fav = faviconUrlFromArticle(n.url)
  return fav ? { src: fav, isArticleImage: false } : null
}

function findBarForTime(ohlcv: OhlcvBar[], t: Time | undefined): OhlcvBar | undefined {
  if (t === undefined || t === null) return undefined
  return ohlcv.find((b) => {
    const bt = barTime(b)
    return bt === t || String(bt) === String(t)
  })
}

function computeStatsFromCrosshair(
  candles: ISeriesApi<'Candlestick'>,
  ohlcv: OhlcvBar[],
  param: MouseEventParams<Time>
): StatsRow | null {
  const cd = param.seriesData?.get(candles) as
    | { open: number; high: number; low: number; close: number }
    | undefined

  const t = param.time
  let volRow: OhlcvBar | undefined
  if (t !== undefined && t !== null) {
    volRow = findBarForTime(ohlcv, t)
  }

  if (cd && typeof cd.open === 'number') {
    return {
      open: cd.open,
      high: cd.high,
      low: cd.low,
      close: cd.close,
      volume: volRow?.volume ?? 0,
    }
  }
  if (volRow) {
    return {
      open: volRow.open,
      high: volRow.high,
      low: volRow.low,
      close: volRow.close,
      volume: volRow.volume,
    }
  }
  if (ohlcv.length > 0) {
    const last = ohlcv[ohlcv.length - 1]
    return {
      open: last.open,
      high: last.high,
      low: last.low,
      close: last.close,
      volume: last.volume,
    }
  }
  return null
}

type TooltipPayload = {
  contentKey: string
  mode: 'full' | 'ohlc'
  headerDate: string
  ticker: string
  dayPct: string | null
  dayUp: boolean
  newsItems: MarketNewsEvent[]
  ohlc: StatsRow
  pctFromPrev: string | null
  prevUp: boolean
}

const GAP = 16
const EST_W = 320
const EST_H_FULL = 280
const EST_H_OHLC = 120

export function ChartNewsOverlay({
  chartRef,
  candleSeriesRef,
  chartContainerRef,
  interactionRootRef,
  chartReady,
  news,
  ohlcv,
  ticker,
  onStatsChange,
}: Props) {
  const tooltipRef = useRef<HTMLDivElement | null>(null)
  const latestParamRef = useRef<MouseEventParams<Time> | null>(null)
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  /** True while pointer is over the tooltip panel (chart crosshair often clears before this fires). */
  const pointerOverTooltipRef = useRef(false)
  const maybeHideTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const ohlcvRef = useRef(ohlcv)
  const contentKeyRef = useRef<string>('')
  const hideTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const onStatsChangeRef = useRef(onStatsChange)
  onStatsChangeRef.current = onStatsChange

  ohlcvRef.current = ohlcv

  const newsByDay = useMemo(() => {
    const m = new Map<string, MarketNewsEvent[]>()
    for (const n of news) {
      if (!n?.date) continue
      const arr = m.get(n.date) ?? []
      arr.push(n)
      m.set(n.date, arr)
    }
    return m
  }, [news])

  const [payload, setPayload] = useState<TooltipPayload | null>(null)

  const positionTooltip = useCallback(
    (param: MouseEventParams<Time>, mode: 'full' | 'ohlc') => {
      const el = tooltipRef.current
      const container = chartContainerRef.current
      if (!el || !container) return

      const point = param.point
      if (!point || param.time === undefined || param.time === null) {
        return
      }

      const rect = container.getBoundingClientRect()
      const vw = window.innerWidth
      const vh = window.innerHeight

      const estW = Math.min(340, Math.max(280, el.offsetWidth || EST_W))
      const estH = mode === 'full' ? EST_H_FULL : EST_H_OHLC

      let left = rect.left + point.x + GAP
      let top = rect.top + point.y + GAP

      if (left + estW > vw - 8) {
        left = rect.left + point.x - estW - GAP
      }
      if (top + estH > vh - 8) {
        top = rect.top + point.y - estH - GAP
      }

      left = Math.max(8, Math.min(left, vw - estW - 8))
      top = Math.max(8, Math.min(top, vh - estH - 8))

      el.style.left = `${left}px`
      el.style.top = `${top}px`
    },
    [chartContainerRef]
  )

  const showTooltipAnimated = useCallback(() => {
    const el = tooltipRef.current
    if (!el) return
    if (hideTimerRef.current) {
      clearTimeout(hideTimerRef.current)
      hideTimerRef.current = null
    }
    el.style.visibility = 'visible'
    el.style.transition = 'opacity 150ms ease-out'
    requestAnimationFrame(() => {
      el.style.opacity = '1'
    })
  }, [])

  const hideTooltipAnimated = useCallback(() => {
    const el = tooltipRef.current
    if (!el) return
    el.style.transition = 'opacity 100ms ease-in'
    el.style.opacity = '0'
    hideTimerRef.current = window.setTimeout(() => {
      el.style.visibility = 'hidden'
      hideTimerRef.current = null
    }, 100)
  }, [])

  const buildPayload = useCallback(
    (param: MouseEventParams<Time>): TooltipPayload | null => {
      const candles = candleSeriesRef.current
      if (!candles) return null

      const time = param.time
      if (time === undefined || time === null) return null

      let iso = timeToIsoDay(time)
      if (!iso && typeof time === 'string') iso = time.slice(0, 10)
      if (!iso) return null

      const cd = param.seriesData?.get(candles) as
        | { open: number; high: number; low: number; close: number }
        | undefined

      const volRow = findBarForTime(ohlcvRef.current, time)
      let ohlc: StatsRow | null = null
      if (cd && typeof cd.open === 'number') {
        ohlc = {
          open: cd.open,
          high: cd.high,
          low: cd.low,
          close: cd.close,
          volume: volRow?.volume ?? 0,
        }
      } else if (volRow) {
        ohlc = {
          open: volRow.open,
          high: volRow.high,
          low: volRow.low,
          close: volRow.close,
          volume: volRow.volume,
        }
      }
      if (!ohlc) return null

      const pool = newsCandidatePool(iso, newsByDay, news, ohlcvRef.current)
      const newsItems = selectMoveRelatedNews(pool, iso, ohlc)
      const mode: 'full' | 'ohlc' = newsItems.length > 0 ? 'full' : 'ohlc'

      const idx = ohlcvRef.current.findIndex((b) => {
        const bt = barTime(b)
        return bt === time || String(bt) === String(time)
      })
      let pctFromPrev: string | null = null
      let prevUp = true
      if (idx > 0) {
        const prev = ohlcvRef.current[idx - 1]
        if (prev && prev.close !== 0) {
          const p = ((ohlc.close - prev.close) / prev.close) * 100
          pctFromPrev = `${p >= 0 ? '+' : ''}${p.toFixed(2)}%`
          prevUp = p >= 0
        }
      }

      let dayPct: string | null = null
      let dayUp = true
      if (ohlc.open !== 0) {
        const d = ((ohlc.close - ohlc.open) / ohlc.open) * 100
        dayPct = `${d >= 0 ? '+' : ''}${d.toFixed(2)}%`
        dayUp = d >= 0
      }

      const headerDate = new Date(`${iso}T12:00:00Z`).toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
      })

      const contentKey = [
        mode,
        iso,
        String(time),
        ohlc.open,
        ohlc.high,
        ohlc.low,
        ohlc.close,
        ohlc.volume,
        newsItems.map((n) => n.title).join('|'),
      ].join('::')

      return {
        contentKey,
        mode,
        headerDate,
        ticker: ticker ?? '—',
        dayPct,
        dayUp,
        newsItems,
        ohlc,
        pctFromPrev,
        prevUp,
      }
    },
    [candleSeriesRef, news, newsByDay, ticker]
  )

  useEffect(() => {
    if (!chartReady) return
    const chart = chartRef.current
    const candles = candleSeriesRef.current
    if (!chart || !candles) return

    const flushTooltip = () => {
      const param = latestParamRef.current
      const el = tooltipRef.current
      if (!param || !el) return

      const time = param.time
      if (time === undefined || time === null) {
        hideTooltipAnimated()
        return
      }

      let layoutParam: MouseEventParams<Time> = param
      if (!param.point) {
        const x = chart.timeScale().timeToCoordinate(time as Time)
        const containerEl = chartContainerRef.current
        if (x != null && containerEl) {
          const h = containerEl.clientHeight
          layoutParam = { ...param, point: { x, y: Math.max(8, h * 0.25) } } as MouseEventParams<Time>
        }
      }
      if (!layoutParam.point) {
        hideTooltipAnimated()
        return
      }

      const next = buildPayload(param)
      if (!next) {
        hideTooltipAnimated()
        return
      }

      if (next.contentKey !== contentKeyRef.current) {
        contentKeyRef.current = next.contentKey
        setPayload(next)
      }

      showTooltipAnimated()
      window.setTimeout(() => positionTooltip(layoutParam, next.mode), 0)
    }

    const clearTooltipUi = () => {
      latestParamRef.current = null
      pointerOverTooltipRef.current = false
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current)
        debounceTimerRef.current = null
      }
      if (maybeHideTimerRef.current) {
        clearTimeout(maybeHideTimerRef.current)
        maybeHideTimerRef.current = null
      }
      hideTooltipAnimated()
      contentKeyRef.current = ''
      setPayload(null)
      if (ohlcvRef.current.length > 0) {
        const last = ohlcvRef.current[ohlcvRef.current.length - 1]
        onStatsChangeRef.current({
          open: last.open,
          high: last.high,
          low: last.low,
          close: last.close,
          volume: last.volume,
        })
      } else {
        onStatsChangeRef.current(null)
      }
    }

    const onCrosshairMove = (param: MouseEventParams<Time>) => {
      const stats = computeStatsFromCrosshair(candles, ohlcvRef.current, param)
      onStatsChangeRef.current(stats)

      if (param.time === undefined || param.time === null) {
        if (debounceTimerRef.current) {
          clearTimeout(debounceTimerRef.current)
          debounceTimerRef.current = null
        }
        if (pointerOverTooltipRef.current) {
          return
        }
        if (maybeHideTimerRef.current) clearTimeout(maybeHideTimerRef.current)
        maybeHideTimerRef.current = window.setTimeout(() => {
          maybeHideTimerRef.current = null
          if (pointerOverTooltipRef.current) return
          clearTooltipUi()
        }, 260)
        return
      }

      if (maybeHideTimerRef.current) {
        clearTimeout(maybeHideTimerRef.current)
        maybeHideTimerRef.current = null
      }

      latestParamRef.current = param

      if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current)
      debounceTimerRef.current = window.setTimeout(flushTooltip, 50)
    }

    chart.subscribeCrosshairMove(onCrosshairMove)

    const leaveEl = interactionRootRef?.current ?? chartContainerRef.current
    const onLeave = () => {
      clearTooltipUi()
    }

    leaveEl?.addEventListener('mouseleave', onLeave)

    return () => {
      chart.unsubscribeCrosshairMove(onCrosshairMove)
      leaveEl?.removeEventListener('mouseleave', onLeave)
      if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current)
      if (maybeHideTimerRef.current) clearTimeout(maybeHideTimerRef.current)
      if (hideTimerRef.current) clearTimeout(hideTimerRef.current)
    }
  }, [
    buildPayload,
    candleSeriesRef,
    chartContainerRef,
    chartReady,
    chartRef,
    hideTooltipAnimated,
    interactionRootRef,
    positionTooltip,
    showTooltipAnimated,
  ])

  useEffect(() => {
    const param = latestParamRef.current
    if (param && payload && tooltipRef.current?.style.visibility === 'visible') {
      positionTooltip(param, payload.mode)
    }
  }, [payload, positionTooltip])

  const panelStyle: CSSProperties = {
    background: 'rgba(15,23,42,0.94)',
    color: '#F8FAFC',
    borderRadius: 12,
    border: '1px solid rgba(255,255,255,0.1)',
    boxShadow: '0 20px 40px rgba(0,0,0,0.3)',
    minWidth: 280,
    maxWidth: 340,
    padding: 14,
    fontFamily: "'Geist', sans-serif",
    fontSize: 12,
    visibility: 'hidden',
    opacity: 0,
    zIndex: 50,
  }

  const panelDyn: CSSProperties =
    payload?.mode === 'ohlc'
      ? { minWidth: 240, maxWidth: 300, padding: 10 }
      : {}

  return (
    <div
      ref={tooltipRef}
      className="pointer-events-none fixed font-body"
      style={{ ...panelStyle, ...panelDyn }}
    >
      {payload ? (
        <div
          className="pointer-events-auto"
          onMouseEnter={() => {
            pointerOverTooltipRef.current = true
            if (maybeHideTimerRef.current) {
              clearTimeout(maybeHideTimerRef.current)
              maybeHideTimerRef.current = null
            }
          }}
          onMouseLeave={() => {
            pointerOverTooltipRef.current = false
          }}
        >
          <div className="mb-2 flex flex-wrap items-center gap-1 border-b border-white/10 pb-2 text-[11px] text-slate-200">
            <span className="text-slate-300">📅</span>
            <span className="font-medium">{payload.headerDate}</span>
            <span className="text-slate-500">·</span>
            <span className="font-mono text-indigo-300">{payload.ticker}</span>
            {payload.dayPct != null ? (
              <>
                <span className="text-slate-500">·</span>
                <span className={payload.dayUp ? 'text-green-400' : 'text-red-400'}>
                  {payload.dayPct} {payload.dayUp ? '▲' : '▼'}
                </span>
              </>
            ) : null}
          </div>

          {payload.mode === 'full' && payload.newsItems[0] ? (
            (() => {
              const n = payload.newsItems[0]
              const lab = sentimentLabel(n)
              const thumb = articleThumbnailSrc(n)
              const articleUrl = n.url?.trim()
              const canOpen = Boolean(articleUrl && /^https?:\/\//i.test(articleUrl))
              const body = (
                <>
                  <div className="relative h-14 w-14 shrink-0 overflow-hidden rounded-md bg-slate-800/90 ring-1 ring-white/10">
                    <span className="absolute inset-0 flex items-center justify-center text-[22px] leading-none text-slate-500">
                      📰
                    </span>
                    {thumb ? (
                      <img
                        src={thumb.src}
                        alt=""
                        className={`relative z-10 h-full w-full bg-slate-900/95 ${
                          thumb.isArticleImage ? 'object-cover' : 'object-contain p-1.5'
                        }`}
                        loading="lazy"
                        referrerPolicy="no-referrer"
                        onError={(e) => {
                          e.currentTarget.style.visibility = 'hidden'
                        }}
                      />
                    ) : null}
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-start gap-1.5">
                      <span className={badgeClass(lab)}>{lab}</span>
                      <span className="font-medium leading-snug text-slate-100">{n.title}</span>
                    </div>
                    {n.summary ? (
                      <p className="mt-1 text-[11px] leading-relaxed text-slate-400">{n.summary}</p>
                    ) : null}
                    <div className="mt-1 text-[10px] text-slate-500">
                      {n.source ? <span>{n.source}</span> : null}
                      {n.source && n.impact ? <span> · </span> : null}
                      {n.impact ? <span>{n.impact} impact</span> : null}
                      {canOpen ? (
                        <>
                          <span> · </span>
                          <span className="text-cyan-400/90">Open article ↗</span>
                        </>
                      ) : null}
                    </div>
                  </div>
                </>
              )
              return canOpen && articleUrl ? (
                <a
                  href={articleUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex gap-3 rounded-md outline-none ring-offset-2 ring-offset-slate-900 transition hover:bg-white/5 focus-visible:ring-2 focus-visible:ring-cyan-400/60"
                >
                  {body}
                </a>
              ) : (
                <div className="flex gap-3">{body}</div>
              )
            })()
          ) : null}

          <div
            className={
              payload.mode === 'full'
                ? 'mt-3 border-t border-white/15 pt-2'
                : 'mt-0 border-0 pt-0'
            }
          >
            {payload.mode === 'full' ? (
              <div className="mb-2 h-px w-full bg-gradient-to-r from-transparent via-white/25 to-transparent" />
            ) : null}
            <div className="font-mono text-[11px] leading-relaxed text-slate-300">
              O: {fmtUsd(payload.ohlc.open)} &nbsp; H: {fmtUsd(payload.ohlc.high)} &nbsp; L:{' '}
              {fmtUsd(payload.ohlc.low)} &nbsp; C: {fmtUsd(payload.ohlc.close)}
            </div>
            <div className="mt-1 text-[11px] text-slate-400">
              Volume: {fmtVol(payload.ohlc.volume)}
              {payload.pctFromPrev != null ? (
                <>
                  {' '}
                  ·{' '}
                  <span className={payload.prevUp ? 'text-green-400' : 'text-red-400'}>
                    {payload.pctFromPrev} from prev close
                  </span>
                </>
              ) : null}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}
