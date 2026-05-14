import axios from 'axios'

export const api = axios.create({
  baseURL: '/api/v1',
  timeout: 30000,
})

api.interceptors.response.use(
  (res) => res,
  (err) => {
    const detail = err.response?.data?.detail
    let message: string
    if (typeof detail === 'string') {
      message = detail
    } else if (Array.isArray(detail)) {
      message = detail
        .map((x: unknown) =>
          typeof x === 'object' && x != null && 'msg' in x
            ? String((x as { msg: string }).msg)
            : JSON.stringify(x)
        )
        .join('; ')
    } else if (detail && typeof detail === 'object' && 'message' in detail) {
      message = String((detail as { message: string }).message)
    } else {
      message = err.message || 'Request failed'
    }
    return Promise.reject(new Error(message))
  }
)

// Typed API calls that map to our FastAPI endpoints
export const marketApi = {
  getData: (
    ticker: string,
    period = '3mo',
    interval?: string | null,
    includeNews = true
  ) => {
    const params = new URLSearchParams({ period })
    if (interval) params.set('interval', interval)
    if (!includeNews) params.set('include_news', 'false')
    return api.get(`/market/${ticker}?${params.toString()}`).then((r) => r.data)
  },
  getSentiment: (ticker: string) =>
    api.get(`/market/${ticker}/sentiment`).then((r) => r.data),
}

function sessionsFromListPayload(data: unknown): unknown[] {
  if (Array.isArray(data)) return data
  if (data && typeof data === 'object') {
    const o = data as Record<string, unknown>
    if (Array.isArray(o.items)) return o.items
    if (Array.isArray(o.sessions)) return o.sessions
  }
  return []
}

export const sessionApi = {
  list: () => api.get('/sessions').then((r) => sessionsFromListPayload(r.data)),
  create: (ticker?: string, period = '3mo') =>
    api.post('/sessions', { ticker, period }).then((r) => r.data),
  get: (id: string) => api.get(`/sessions/${id}`).then((r) => r.data),
  delete: (id: string) => api.delete(`/sessions/${id}`).then((r) => r.data),
}

export const portfolioApi = {
  list: () => api.get('/portfolio').then((r) => r.data),
  add: (ticker: string, notes = '') =>
    api.post('/portfolio', { ticker, notes }).then((r) => r.data),
  remove: (ticker: string) =>
    api.delete(`/portfolio/${ticker}`).then((r) => r.data),
}

/** Payload for POST /chat/stream (optional chart_context). */
export type NewsContextItem = {
  date: string
  title: string
  sentiment: string | null
  impact: string | null
}

export type ChartContext = {
  ticker: string
  from_date: string
  to_date: string
  price_change_pct: number
  open_price: number
  close_price: number
  high: number
  low: number
  news_count: number
  top_news: NewsContextItem[]
}

export type ChatStreamRequestBody = {
  session_id: string
  ticker: string
  message: string
  chart_context?: ChartContext
}

export function buildChatStreamBody(
  sessionId: string,
  ticker: string | null,
  message: string,
  chartContext?: ChartContext | null
): ChatStreamRequestBody {
  const body: ChatStreamRequestBody = {
    session_id: sessionId,
    ticker: (ticker ?? '').toUpperCase(),
    message,
  }
  if (chartContext) body.chart_context = chartContext
  return body
}

export const chatApi = {
  streamPath: '/chat/stream' as const,
  buildStreamBody: buildChatStreamBody,
}
