import { clsx } from 'clsx'
import { Clock, Search } from 'lucide-react'
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
} from 'react'

export interface TickerOption {
  symbol: string
  name: string
  sector: string
  sectorColor: string
}

const POPULAR_GRID_SYMBOLS = [
  'AAPL',
  'NVDA',
  'TSLA',
  'MSFT',
  'META',
  'GOOGL',
  'AMZN',
  'AMD',
  'JPM',
  'NFLX',
  'V',
  'WMT',
] as const

export const POPULAR_TICKERS: TickerOption[] = [
  { symbol: 'AAPL', name: 'Apple Inc.', sector: 'Technology', sectorColor: '#4F46E5' },
  { symbol: 'NVDA', name: 'NVIDIA Corporation', sector: 'Technology', sectorColor: '#4F46E5' },
  { symbol: 'TSLA', name: 'Tesla Inc.', sector: 'Automotive', sectorColor: '#059669' },
  { symbol: 'MSFT', name: 'Microsoft Corporation', sector: 'Technology', sectorColor: '#4F46E5' },
  { symbol: 'META', name: 'Meta Platforms', sector: 'Technology', sectorColor: '#4F46E5' },
  { symbol: 'GOOGL', name: 'Alphabet Inc.', sector: 'Technology', sectorColor: '#4F46E5' },
  { symbol: 'AMZN', name: 'Amazon.com Inc.', sector: 'E-Commerce', sectorColor: '#D97706' },
  { symbol: 'AMD', name: 'Advanced Micro Devices', sector: 'Semiconductors', sectorColor: '#7C3AED' },
  { symbol: 'JPM', name: 'JPMorgan Chase', sector: 'Finance', sectorColor: '#0891B2' },
  { symbol: 'NFLX', name: 'Netflix Inc.', sector: 'Media', sectorColor: '#DC2626' },
  { symbol: 'V', name: 'Visa Inc.', sector: 'Finance', sectorColor: '#0891B2' },
  { symbol: 'WMT', name: 'Walmart Inc.', sector: 'Retail', sectorColor: '#D97706' },
  { symbol: 'BABA', name: 'Alibaba Group', sector: 'E-Commerce', sectorColor: '#D97706' },
  { symbol: 'INTC', name: 'Intel Corporation', sector: 'Semiconductors', sectorColor: '#7C3AED' },
  { symbol: 'DIS', name: 'Walt Disney Co.', sector: 'Entertainment', sectorColor: '#DC2626' },
  { symbol: 'PYPL', name: 'PayPal Holdings', sector: 'Finance', sectorColor: '#0891B2' },
  { symbol: 'CRM', name: 'Salesforce Inc.', sector: 'Software', sectorColor: '#4F46E5' },
  { symbol: 'UBER', name: 'Uber Technologies', sector: 'Transport', sectorColor: '#334155' },
]

const STORAGE_KEY = 'fa_recent_tickers'

interface RecentEntry {
  symbol: string
  name: string
  viewedAt: number
}

function loadRecent(): RecentEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw) as unknown
    if (!Array.isArray(parsed)) return []
    return parsed
      .filter(
        (r): r is RecentEntry =>
          r != null &&
          typeof r === 'object' &&
          typeof (r as RecentEntry).symbol === 'string' &&
          typeof (r as RecentEntry).name === 'string' &&
          typeof (r as RecentEntry).viewedAt === 'number'
      )
      .slice(0, 5)
  } catch {
    return []
  }
}

function saveRecent(symbol: string, name: string) {
  const prev = loadRecent().filter((r) => r.symbol !== symbol)
  const next: RecentEntry[] = [{ symbol, name, viewedAt: Date.now() }, ...prev].slice(0, 5)
  localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
  window.dispatchEvent(new CustomEvent('fa-recent-tickers-changed'))
}

/** Call when selecting a ticker outside this component (e.g. welcome grid chips). */
export function rememberTickerSelection(symbol: string) {
  const upper = symbol.toUpperCase().slice(0, 6)
  const meta = optionBySymbol(upper)
  saveRecent(upper, meta?.name ?? upper)
}

function optionBySymbol(symbol: string): TickerOption | undefined {
  return POPULAR_TICKERS.find((o) => o.symbol === symbol)
}

function isValidTickerSymbol(s: string) {
  return /^[A-Z]{2,6}$/.test(s)
}

function relShort(ts: number) {
  const diff = Date.now() - ts
  const s = Math.max(0, Math.floor(diff / 1000))
  if (s < 60) return `${s}s ago`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ago`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h ago`
  const d = Math.floor(h / 24)
  return `${d}d ago`
}

function filterTickers(query: string): TickerOption[] {
  const q = query.trim().toLowerCase()
  if (!q) return []
  return POPULAR_TICKERS.filter(
    (o) => o.symbol.toLowerCase().includes(q) || o.name.toLowerCase().includes(q)
  )
}

function Highlight({ text, query }: { text: string; query: string }) {
  const q = query.trim()
  if (!q) return <>{text}</>
  const lower = text.toLowerCase()
  const qLower = q.toLowerCase()
  const idx = lower.indexOf(qLower)
  if (idx === -1) return <>{text}</>
  return (
    <>
      {text.slice(0, idx)}
      <strong className="font-semibold text-text1">{text.slice(idx, idx + q.length)}</strong>
      {text.slice(idx + q.length)}
    </>
  )
}

type NavItem =
  | { kind: 'popular'; symbol: string }
  | { kind: 'recent'; symbol: string; name: string; viewedAt: number }
  | { kind: 'result'; option: TickerOption }
  | { kind: 'enter'; symbol: string }

export interface TickerSearchBarProps {
  onSelect: (ticker: string) => void
  size?: 'default' | 'large'
  placeholder?: string
}

export function TickerSearchBar({
  onSelect,
  size = 'default',
  placeholder = 'Search ticker or company name...',
}: TickerSearchBarProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const [query, setQuery] = useState('')
  const [isOpen, setIsOpen] = useState(false)
  const [isFocused, setIsFocused] = useState(false)
  const [focusedIndex, setFocusedIndex] = useState(-1)
  const [recentTickers, setRecentTickers] = useState<RecentEntry[]>(() =>
    typeof window === 'undefined' ? [] : loadRecent()
  )

  const normalizedQuery = useMemo(() => query.toUpperCase().replace(/[^A-Z]/g, '').slice(0, 6), [query])

  const popularGridOptions = useMemo(() => {
    return POPULAR_GRID_SYMBOLS.map((sym) => optionBySymbol(sym)).filter(
      (o): o is TickerOption => o != null
    )
  }, [])

  const searchResults = useMemo(() => filterTickers(query), [query])

  const showSearchSection = query.trim().length >= 1

  const symbolInCatalog = useMemo(
    () => POPULAR_TICKERS.some((o) => o.symbol === normalizedQuery),
    [normalizedQuery]
  )

  const showEnterRow =
    showSearchSection &&
    normalizedQuery.length >= 2 &&
    isValidTickerSymbol(normalizedQuery) &&
    !symbolInCatalog

  const navItems = useMemo((): NavItem[] => {
    const items: NavItem[] = []
    if (!showSearchSection) {
      for (const sym of popularGridOptions.map((o) => o.symbol)) {
        items.push({ kind: 'popular', symbol: sym })
      }
      for (const r of recentTickers) {
        items.push({
          kind: 'recent',
          symbol: r.symbol,
          name: r.name,
          viewedAt: r.viewedAt,
        })
      }
      return items
    }
    for (const opt of searchResults) {
      items.push({ kind: 'result', option: opt })
    }
    if (showEnterRow) {
      items.push({ kind: 'enter', symbol: normalizedQuery })
    }
    return items
  }, [
    normalizedQuery,
    popularGridOptions,
    recentTickers,
    searchResults,
    showEnterRow,
    showSearchSection,
  ])

  useLayoutEffect(() => {
    if (focusedIndex >= navItems.length) {
      setFocusedIndex(navItems.length > 0 ? navItems.length - 1 : -1)
    }
  }, [focusedIndex, navItems.length])

  const close = useCallback(() => {
    setIsOpen(false)
    setFocusedIndex(-1)
  }, [])

  const commitSelect = useCallback(
    (symbol: string) => {
      const upper = symbol.toUpperCase().slice(0, 6)
      if (!isValidTickerSymbol(upper)) return
      const meta = optionBySymbol(upper)
      saveRecent(upper, meta?.name ?? upper)
      setRecentTickers(loadRecent())
      onSelect(upper)
      setQuery('')
      close()
      inputRef.current?.blur()
    },
    [close, onSelect]
  )

  useEffect(() => {
    if (!isOpen) return
    const onDoc = (e: MouseEvent) => {
      const el = containerRef.current
      if (!el || !(e.target instanceof Node)) return
      if (!el.contains(e.target)) close()
    }
    document.addEventListener('mousedown', onDoc)
    return () => document.removeEventListener('mousedown', onDoc)
  }, [close, isOpen])

  useEffect(() => {
    setRecentTickers(loadRecent())
  }, [])

  useEffect(() => {
    const onRecent = () => setRecentTickers(loadRecent())
    window.addEventListener('fa-recent-tickers-changed', onRecent)
    return () => window.removeEventListener('fa-recent-tickers-changed', onRecent)
  }, [])

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Escape') {
      e.preventDefault()
      close()
      inputRef.current?.blur()
      return
    }

    if (!isOpen && (e.key === 'ArrowDown' || e.key === 'ArrowUp')) {
      e.preventDefault()
      setIsOpen(true)
      setFocusedIndex(0)
      return
    }

    if (!isOpen) {
      if (e.key === 'Enter') {
        const q = normalizedQuery
        if (isValidTickerSymbol(q)) {
          e.preventDefault()
          commitSelect(q)
        }
      }
      return
    }

    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setFocusedIndex((i) => (navItems.length === 0 ? -1 : (i + 1) % navItems.length))
      return
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault()
      setFocusedIndex((i) =>
        navItems.length === 0 ? -1 : i <= 0 ? navItems.length - 1 : i - 1
      )
      return
    }

    if (e.key === 'Enter') {
      e.preventDefault()
      if (focusedIndex >= 0 && focusedIndex < navItems.length) {
        const it = navItems[focusedIndex]
        if (it.kind === 'popular' || it.kind === 'enter') commitSelect(it.symbol)
        else if (it.kind === 'recent') commitSelect(it.symbol)
        else if (it.kind === 'result') commitSelect(it.option.symbol)
        return
      }
      if (isValidTickerSymbol(normalizedQuery)) commitSelect(normalizedQuery)
    }
  }

  const large = size === 'large'

  return (
    <div
      ref={containerRef}
      className={clsx(
        'relative w-full max-w-xl origin-top transition-transform duration-150 ease-smooth',
        isFocused && 'scale-[1.01]'
      )}
    >
      <div
        className={clsx(
          'flex w-full items-center gap-2 rounded-xl border border-border bg-bg-surface px-3 transition duration-150 ease-smooth',
          large ? 'h-[52px] px-4' : 'h-11',
          isFocused && 'border-primary glow-cyan'
        )}
      >
        <Search className="h-4 w-4 shrink-0 text-text3" aria-hidden />
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value)
            setIsOpen(true)
            setFocusedIndex(0)
          }}
          onFocus={() => {
            setIsFocused(true)
            setIsOpen(true)
            setFocusedIndex(navItems.length > 0 ? 0 : -1)
          }}
          onBlur={() => setIsFocused(false)}
          onKeyDown={onKeyDown}
          placeholder={placeholder}
          autoComplete="off"
          spellCheck={false}
          className={clsx(
            'min-w-0 flex-1 bg-transparent text-sm text-text1 outline-none placeholder:text-text4',
            large && 'text-base'
          )}
          aria-autocomplete="list"
          aria-expanded={isOpen}
          aria-controls="ticker-search-listbox"
        />
        {isFocused ? (
          <span className="hidden shrink-0 items-center gap-1 rounded-md border border-border bg-surface2 px-2 py-0.5 font-mono text-2xs text-text3 sm:inline-flex">
            ↵ Enter
          </span>
        ) : null}
      </div>

      {isOpen ? (
        <div
          id="ticker-search-listbox"
          role="listbox"
          className="animate-fade-in absolute left-0 right-0 z-[100] mt-2 max-h-[360px] overflow-y-auto rounded-xl border border-border bg-bg-surface shadow-lg"
        >
          {!showSearchSection ? (
            <>
              <div className="border-b border-border px-3 py-2">
                <p className="text-2xs font-semibold uppercase tracking-widest text-text4">Popular</p>
                <div className="mt-2 grid grid-cols-4 gap-2">
                  {popularGridOptions.map((opt, idx) => {
                    const active = focusedIndex === idx
                    return (
                      <button
                        key={opt.symbol}
                        type="button"
                        role="option"
                        aria-selected={active}
                        className={clsx(
                          'flex flex-col items-center gap-1 rounded-lg border border-transparent px-1 py-2 text-center transition duration-150 ease-smooth',
                          active ? 'border-primary/30 bg-primary-light/50' : 'hover:bg-surface2'
                        )}
                        onMouseDown={(e) => e.preventDefault()}
                        onClick={() => commitSelect(opt.symbol)}
                      >
                        <span
                          className="h-2 w-2 rounded-full ring-2 ring-white"
                          style={{ backgroundColor: opt.sectorColor }}
                          title={opt.sector}
                        />
                        <span className="font-mono text-xs font-semibold text-text1">{opt.symbol}</span>
                      </button>
                    )
                  })}
                </div>
              </div>

              {recentTickers.length > 0 ? (
                <div className="px-2 py-2">
                  <p className="px-1 pb-2 text-2xs font-semibold uppercase tracking-widest text-text4">
                    Recently viewed
                  </p>
                  <ul className="space-y-0.5">
                    {recentTickers.map((r, i) => {
                      const flatIdx = popularGridOptions.length + i
                      const active = focusedIndex === flatIdx
                      return (
                        <li key={r.symbol}>
                          <button
                            type="button"
                            role="option"
                            aria-selected={active}
                            className={clsx(
                              'flex w-full items-center gap-2 rounded-lg px-2 py-2 text-left text-sm transition duration-150 ease-smooth',
                              active ? 'bg-primary-light/50' : 'hover:bg-surface2'
                            )}
                            onMouseDown={(e) => e.preventDefault()}
                            onClick={() => commitSelect(r.symbol)}
                          >
                            <Clock className="h-3.5 w-3.5 shrink-0 text-text4" aria-hidden />
                            <span className="font-mono font-medium text-text1">{r.symbol}</span>
                            <span className="min-w-0 flex-1 truncate text-text3">{r.name}</span>
                            <span className="shrink-0 text-2xs text-text4">{relShort(r.viewedAt)}</span>
                          </button>
                        </li>
                      )
                    })}
                  </ul>
                </div>
              ) : null}
            </>
          ) : (
            <div className="py-2">
              <p className="px-3 pb-1 text-2xs font-semibold uppercase tracking-widest text-text4">
                Results
              </p>
              {searchResults.length === 0 && !showEnterRow ? (
                <p className="px-3 py-4 text-sm text-text3">No matches. Try another name or symbol.</p>
              ) : (
                <ul className="space-y-0.5 px-2">
                  {searchResults.map((opt, i) => {
                    const active = focusedIndex === i
                    return (
                      <li key={opt.symbol}>
                        <button
                          type="button"
                          role="option"
                          aria-selected={active}
                          className={clsx(
                            'flex w-full items-center gap-3 rounded-lg px-2 py-2 text-left text-sm transition duration-150 ease-smooth',
                            active ? 'bg-primary-light/50' : 'hover:bg-surface2'
                          )}
                          onMouseDown={(e) => e.preventDefault()}
                          onClick={() => commitSelect(opt.symbol)}
                        >
                          <span
                            className="h-2.5 w-2.5 shrink-0 rounded-full"
                            style={{ backgroundColor: opt.sectorColor }}
                          />
                          <span className="w-14 shrink-0 font-mono text-sm font-semibold text-text1">
                            <Highlight text={opt.symbol} query={query} />
                          </span>
                          <span className="min-w-0 flex-1 truncate text-text-secondary">
                            <Highlight text={opt.name} query={query} />
                          </span>
                          <span className="hidden shrink-0 text-2xs text-text4 sm:inline">{opt.sector}</span>
                        </button>
                      </li>
                    )
                  })}
                  {showEnterRow ? (
                    <li>
                      <button
                        type="button"
                        role="option"
                        aria-selected={focusedIndex === searchResults.length}
                        className={clsx(
                          'mt-1 flex w-full items-center rounded-lg border border-dashed border-border px-2 py-2.5 text-left text-sm transition duration-150 ease-smooth',
                          focusedIndex === searchResults.length
                            ? 'border-primary/40 bg-primary-light/40'
                            : 'hover:bg-surface2'
                        )}
                        onMouseDown={(e) => e.preventDefault()}
                        onClick={() => commitSelect(normalizedQuery)}
                      >
                        Press Enter to search <span className="font-mono font-semibold">{normalizedQuery}</span>
                      </button>
                    </li>
                  ) : null}
                </ul>
              )}
            </div>
          )}
        </div>
      ) : null}
    </div>
  )
}
