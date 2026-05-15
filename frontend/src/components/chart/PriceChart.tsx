import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import {
  createChart,
  ColorType,
  CrosshairMode,
  CandlestickSeries,
  HistogramSeries,
  type IChartApi,
  type ISeriesApi,
  type Time,
} from 'lightweight-charts'

import { useMarketData, type OhlcvBar } from '../../hooks/useMarketData'
import { useAppStore, type ChartPeriod } from '../../store/useAppStore'
import { Skeleton } from '../ui/Skeleton'

import { ChartNewsOverlay, timeToIsoDay, type StatsRow } from './ChartNewsOverlay'
import { buildEventMarkers } from './EventMarkers'
import { RangeBar } from './RangeBar'

const PERIODS: { label: string; value: ChartPeriod }[] = [
  { label: '1W', value: '5d' },
  { label: '2W', value: '10d' },
  { label: '1M', value: '1mo' },
  { label: '3M', value: '3mo' },
  { label: '6M', value: '6mo' },
  { label: '1Y', value: '1y' },
]

export function PriceChart() {
  const ticker = useAppStore((s) => s.activeTicker)
  const period = useAppStore((s) => s.chartPeriod)
  const setChartPeriod = useAppStore((s) => s.setChartPeriod)
  const { start, end } = useAppStore((s) => s.chartRange)
  const setRange = useAppStore((s) => s.setRange)
  const clearRange = useAppStore((s) => s.clearRange)

  const { data, isLoading, isFetching, error } = useMarketData(ticker, period)
  const [seriesError, setSeriesError] = useState<string | null>(null)
  const [chartReady, setChartReady] = useState(false)
  const [statsRow, setStatsRow] = useState<StatsRow | null>(null)
  const [selectMode, setSelectMode] = useState(false)
  const [dragOverlay, setDragOverlay] = useState<{ left: number; width: number } | null>(null)

  const handleStatsChange = useCallback((row: StatsRow | null) => {
    setStatsRow(row)
  }, [])

  useEffect(() => {
    setSeriesError(null)
    setStatsRow(null)
  }, [ticker])

  useEffect(() => {
    if (!start) setSelectMode(false)
  }, [start])

  const containerRef = useRef<HTMLDivElement | null>(null)
  const chartInteractionRef = useRef<HTMLDivElement | null>(null)

  const chartRef = useRef<IChartApi | null>(null)
  const candleRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const volumeRef = useRef<ISeriesApi<'Histogram'> | null>(null)

  const ohlcv = data?.ohlcv ?? []
  const news = data?.news ?? []

  const tradingDaysInRange = useMemo(() => {
    if (!start || !end) return 0
    return ohlcv.filter((b) => b.date >= start && b.date <= end).length
  }, [end, ohlcv, start])

  useEffect(() => {
    if (!containerRef.current) return

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#0a0e1a' },
        textColor: '#94a3b8',
      },
      grid: {
        vertLines: { color: 'rgba(255,255,255,0.06)' },
        horzLines: { color: 'rgba(255,255,255,0.06)' },
      },
      crosshair: {
        mode: CrosshairMode.Magnet,
      },
      rightPriceScale: {
        borderColor: 'rgba(255,255,255,0.08)',
      },
      timeScale: {
        borderColor: 'rgba(255,255,255,0.08)',
        rightOffset: 2,
      },
      handleScroll: true,
      handleScale: true,
    })

    // v5 API: chart.addSeries(CandlestickSeries, options)
    const candles = (chart as unknown as { addSeries: Function }).addSeries(
      CandlestickSeries,
      {
        upColor: '#10b981',
        downColor: '#ef4444',
        borderUpColor: '#10b981',
        borderDownColor: '#ef4444',
        wickUpColor: '#10b981',
        wickDownColor: '#ef4444',
        priceLineVisible: true,
        lastValueVisible: true,
      }
    ) as ISeriesApi<'Candlestick'>

    candles.priceScale().applyOptions({
      scaleMargins: { top: 0.08, bottom: 0.28 },
    })

    const volume = (chart as unknown as { addSeries: Function }).addSeries(
      HistogramSeries,
      {
        priceFormat: { type: 'volume' },
        priceScaleId: '',
      }
    ) as ISeriesApi<'Histogram'>

    volume.priceScale().applyOptions({
      scaleMargins: { top: 0.76, bottom: 0 },
    })

    chartRef.current = chart
    candleRef.current = candles
    volumeRef.current = volume
    setChartReady(true)

    const ro = new ResizeObserver(() => {
      if (!containerRef.current) return
      chart.applyOptions({
        width: containerRef.current.clientWidth,
        height: containerRef.current.clientHeight,
      })
    })
    ro.observe(containerRef.current)

    const syncChartSize = () => {
      const el = containerRef.current
      if (!el) return
      const rect = el.getBoundingClientRect()
      const w = Math.max(1, el.clientWidth || Math.floor(rect.width) || 320)
      const h = Math.max(1, el.clientHeight || Math.floor(rect.height) || 320)
      chart.applyOptions({ width: w, height: h })
    }
    syncChartSize()
    requestAnimationFrame(syncChartSize)

    return () => {
      ro.disconnect()
      chart.remove()
      chartRef.current = null
      candleRef.current = null
      volumeRef.current = null
      setChartReady(false)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- ticker must reset chart when symbol changes
  }, [ticker])

  useEffect(() => {
    const chart = chartRef.current
    const candles = candleRef.current
    const volume = volumeRef.current
    if (!chart || !candles || !volume) return

    try {
      const candleData = ohlcv.map((b) => ({
        time: b.date as unknown as Time,
        open: b.open,
        high: b.high,
        low: b.low,
        close: b.close,
      }))

      candles.setData(candleData)

      const volumeData = ohlcv.map((b) => {
        const up = b.close >= b.open
        return {
          time: b.date as unknown as Time,
          value: b.volume,
          color: up ? '#10b981' : '#ef4444',
        }
      })
      volume.setData(volumeData)

      const bt = (bar: OhlcvBar): Time => (typeof bar.t === 'number' ? bar.t as Time : bar.date as Time)
      const markers = buildEventMarkers(news, ohlcv, bt)
      ;(candles as any).setMarkers?.(markers)

      chart.timeScale().fitContent()
      setSeriesError(null)
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Chart update failed'
      setSeriesError(msg)
    }
  }, [news, ohlcv, ticker])

  useEffect(() => {
    const chart = chartRef.current
    if (!chart) return
    if (!start || !end) return
    if (start !== end) return

    try {
      chart.timeScale().setVisibleRange({
        from: start as unknown as Time,
        to: end as unknown as Time,
      })
    } catch {
      // ignore if time not in series yet
    }
  }, [end, start])

  useEffect(() => {
    if (!selectMode || !chartReady) {
      setDragOverlay(null)
      return
    }
    const el = containerRef.current
    const chart = chartRef.current
    if (!el || !chart) return

    const ts = chart.timeScale()
    let drag: { pointerId: number; originX: number } | null = null

    const localX = (e: PointerEvent) => {
      const rect = el.getBoundingClientRect()
      return e.clientX - rect.left
    }

    const onWindowMove = (e: PointerEvent) => {
      if (!drag || e.pointerId !== drag.pointerId) return
      const x = localX(e)
      setDragOverlay({ left: Math.min(drag.originX, x), width: Math.abs(x - drag.originX) })
    }

    const finishWindowListeners = () => {
      window.removeEventListener('pointermove', onWindowMove)
      window.removeEventListener('pointerup', onWindowUp)
      window.removeEventListener('pointercancel', onWindowUp)
    }

    const coordinateToTimeInPlot = (rawX: number): Time | null => {
      const w = el.clientWidth
      if (w < 1) return null
      const hi = w - 1
      const x0 = Math.max(0, Math.min(hi, Math.round(rawX)))
      for (let d = 0; d <= 40; d++) {
        const xLo = Math.max(0, x0 - d)
        const xHi = Math.min(hi, x0 + d)
        const tLo = ts.coordinateToTime(xLo)
        if (tLo != null) return tLo as Time
        if (xLo !== xHi) {
          const tHi = ts.coordinateToTime(xHi)
          if (tHi != null) return tHi as Time
        }
      }
      return null
    }

    const onWindowUp = (e: PointerEvent) => {
      if (!drag || e.pointerId !== drag.pointerId) return
      const originX = drag.originX
      const endX = localX(e)
      drag = null
      setDragOverlay(null)
      finishWindowListeners()

      const xLeft = Math.min(originX, endX)
      const xRight = Math.max(originX, endX)
      if (xRight - xLeft < 2) return

      const tStart = coordinateToTimeInPlot(xLeft)
      const tEnd = coordinateToTimeInPlot(xRight)
      const isoStart = tStart != null ? timeToIsoDay(tStart) : null
      const isoEnd = tEnd != null ? timeToIsoDay(tEnd) : null
      if (isoStart && isoEnd) {
        const [from, to] = isoStart <= isoEnd ? [isoStart, isoEnd] : [isoEnd, isoStart]
        setRange(from, to)
        setSelectMode(false)
      }
    }

    const onPointerDown = (e: PointerEvent) => {
      if (e.button !== 0) return
      e.preventDefault()
      e.stopPropagation()
      const originX = localX(e)
      drag = { pointerId: e.pointerId, originX }
      setDragOverlay({ left: originX, width: 0 })
      window.addEventListener('pointermove', onWindowMove)
      window.addEventListener('pointerup', onWindowUp)
      window.addEventListener('pointercancel', onWindowUp)
    }

    el.addEventListener('pointerdown', onPointerDown, true)

    return () => {
      el.removeEventListener('pointerdown', onPointerDown, true)
      finishWindowListeners()
      drag = null
      setDragOverlay(null)
    }
  }, [chartReady, selectMode, setRange])

  const displayError = error ?? (seriesError ? new Error(seriesError) : null)

  const headerPrice = statsRow != null ? statsRow.close : data?.current_price
  const headerChangePct =
    statsRow != null
      ? statsRow.open !== 0
        ? ((statsRow.close - statsRow.open) / statsRow.open) * 100
        : null
      : data?.price_change_pct

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="mb-3 flex items-center justify-between gap-2">
        <div className="font-display min-w-0 text-[length:var(--text-14)] font-medium text-text-primary">
          {ticker ? (
            <>
              <span className="font-data text-accent">{ticker}</span>
              {headerPrice != null ? (
                <span className="ml-2 text-text-secondary">
                  <span className="font-data">{headerPrice}</span>
                  {headerChangePct != null ? (
                    <span
                      className={`font-data ml-1 ${headerChangePct >= 0 ? 'text-accent-green' : 'text-accent-red'}`}
                    >
                      ({headerChangePct >= 0 ? '+' : ''}
                      {headerChangePct.toFixed(2)}%)
                    </span>
                  ) : null}
                </span>
              ) : null}
            </>
          ) : (
            <span className="text-text-secondary">Select a ticker to load the book</span>
          )}
        </div>
        <div className="flex shrink-0 flex-wrap items-center justify-end gap-2">
          <button
            type="button"
            onClick={() => {
              setSelectMode((prev) => {
                if (prev) clearRange()
                return !prev
              })
            }}
            className={`font-data shrink-0 rounded-full border text-[length:var(--text-11)] px-2 py-1 transition ${
              selectMode
                ? 'border-accent bg-accent/10 text-accent ring-2 ring-accent/40'
                : 'border-border-subtle text-text-muted'
            }`}
          >
            {selectMode ? '✕ Cancel' : '⊹ Select Range'}
          </button>
          <div className="flex flex-wrap items-center justify-end gap-1">
          {PERIODS.map(({ label, value }) => {
            const active = period === value
            return (
              <button
                key={value}
                type="button"
                onClick={() => setChartPeriod(value)}
                className={`font-data rounded-full border text-[length:var(--text-11)] px-2 py-1 transition ${
                  active
                    ? 'border-accent bg-accent/10 text-accent'
                    : 'border-border-subtle text-text-muted'
                }`}
              >
                {label}
              </button>
            )
          })}
          </div>
        </div>
      </div>

      <div ref={chartInteractionRef} className="relative min-h-0 flex-1">
        <div
          ref={containerRef}
          className={`h-full min-h-0 w-full rounded-xl border border-border-subtle bg-bg-primary${
            selectMode ? ' cursor-crosshair' : ''
          }`}
        />

        {dragOverlay ? (
          <div
            className="pointer-events-none absolute z-[4] rounded-xl bg-accent/15 ring-1 ring-inset ring-accent/30"
            style={{
              left: dragOverlay.left,
              width: dragOverlay.width,
              top: 0,
              bottom: 0,
            }}
          />
        ) : null}

        <ChartNewsOverlay
          chartRef={chartRef}
          candleSeriesRef={candleRef}
          chartContainerRef={containerRef}
          interactionRootRef={chartInteractionRef}
          chartReady={chartReady}
          news={news}
          ohlcv={ohlcv}
          ticker={ticker}
          onStatsChange={handleStatsChange}
        />

        {(isLoading || (isFetching && !data)) && !displayError ? (
          <div className="pointer-events-none absolute inset-0 z-[1] flex flex-col justify-center gap-2 rounded-xl border border-transparent bg-bg-primary/60 p-6 backdrop-blur-[1px]">
            <Skeleton className="h-8 w-3/4 max-w-md" rounded="md" />
            <Skeleton className="h-[55%] min-h-[220px] w-full flex-1" rounded="lg" />
            <div className="mx-auto text-[length:var(--text-12)] text-text-muted">Loading quotes…</div>
          </div>
        ) : null}

        {displayError ? (
          <div className="absolute inset-0 z-[2] flex items-center justify-center overflow-auto rounded-xl border border-accent-red/40 bg-bg-primary/95 p-4 text-center">
            <div className="max-w-md text-sm text-accent-red">
              <p className="font-medium text-text-primary">Could not load market data</p>
              <p className="mt-2 text-text-muted">{displayError.message}</p>
              <p className="mt-3 text-xs text-text-dim">
                Is the API running on port 8000? In PowerShell, from the{' '}
                <span className="font-mono text-text-muted">backend</span> folder run:{' '}
                <code className="mt-1 block rounded bg-bg-tertiary px-2 py-1.5 font-mono text-[11px] text-cyan">
                  python -m uvicorn main:app --host 127.0.0.1 --port 8000
                </code>
              </p>
            </div>
          </div>
        ) : null}

        {ticker &&
        !displayError &&
        !isLoading &&
        !(isFetching && !data) &&
        data &&
        ohlcv.length === 0 ? (
          <div className="absolute inset-0 z-[1] flex items-center justify-center rounded-xl border border-border-subtle bg-bg-primary/80 p-4 text-center text-sm text-text-muted">
            No OHLCV bars returned for this range. Try another period or check the ticker symbol.
          </div>
        ) : null}
      </div>

      <RangeBar tradingDays={tradingDaysInRange} ohlcv={ohlcv} news={news} />
    </div>
  )
}

