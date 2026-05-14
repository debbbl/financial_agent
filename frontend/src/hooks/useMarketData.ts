import { useQuery } from '@tanstack/react-query'

import { marketApi } from '../api/client'

export type OhlcvBar = {
  date: string
  /** Unix seconds (UTC) for intraday / precise chart time */
  t?: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export type MarketNewsEvent = {
  date: string
  title: string
  sentiment?: 'bullish' | 'bearish' | 'neutral'
  sentiment_score?: number
  score?: number
  source?: string
  category?: string
  impact?: string
  summary?: string
  url?: string
}

export type MarketDataResponse = {
  ticker: string
  current_price?: number
  price_change_pct?: number
  ohlcv: OhlcvBar[]
  news?: MarketNewsEvent[]
  news_count?: number
}

export type UseMarketDataOptions = {
  /** When false, backend skips news + FinBERT (OHLCV only). Default true. */
  includeNews?: boolean
}

export function useMarketData(
  ticker: string | null,
  period: string,
  interval: string | null = null,
  options?: UseMarketDataOptions
) {
  const includeNews = options?.includeNews ?? true
  const query = useQuery<MarketDataResponse, Error>({
    queryKey: ['market', ticker, period, interval, includeNews],
    queryFn: async () => {
      if (!ticker) throw new Error('Select a ticker to load market data.')
      return await marketApi.getData(ticker, period, interval, includeNews)
    },
    enabled: !!ticker,
    staleTime: 60 * 1000,
    retry: 1,
  })

  return {
    data: query.data,
    isLoading: query.isLoading,
    isFetching: query.isFetching,
    error: query.error ? new Error(query.error.message || 'Failed to load market data.') : null,
    refetch: query.refetch,
  }
}

