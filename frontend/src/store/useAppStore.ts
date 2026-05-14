import { create } from 'zustand'

import type { ChartContext } from '../api/client'

export type ChartPeriod = '5d' | '10d' | '1mo' | '3mo' | '6mo' | '1y' | '2y'

export type SidebarNavId =
  | 'dashboard'
  | 'portfolio'
  | 'markets'
  | 'screener'
  | 'news'
  | 'settings'

export type DashboardPanelTab = 'news' | 'events' | 'ai' | 'technical'

/** Primary workspace tab (TopBar + MainContent). */
export type MainTab = 'chart' | 'chat' | 'news' | 'portfolio' | 'dashboard'

interface AppStore {
  activeTicker: string | null
  activeSessionId: string | null
  chatDraft: string
  chartRange: { start: string | null; end: string | null }
  /** Set by RangeBar when opening AI from a chart range; consumed on next chat send. */
  pendingChartContext: ChartContext | null

  chartPeriod: ChartPeriod
  /** Filters popular ticker chip row + sector bar above chart */
  chartSectorFilter: string
  activeSidebarNav: SidebarNavId
  activePanelTab: DashboardPanelTab
  activeTab: MainTab

  setTicker: (ticker: string | null) => void
  setSession: (id: string | null) => void
  setChatDraft: (draft: string) => void
  setPendingChartContext: (ctx: ChartContext | null) => void
  setRange: (start: string, end: string) => void
  clearRange: () => void

  setChartPeriod: (p: ChartPeriod) => void
  setChartSectorFilter: (s: string) => void
  setSidebarNav: (nav: SidebarNavId) => void
  setPanelTab: (tab: DashboardPanelTab) => void
  setTab: (tab: MainTab) => void
}

export const useAppStore = create<AppStore>((set) => ({
  activeTicker: null,
  activeSessionId: null,
  chatDraft: '',
  chartRange: { start: null, end: null },
  pendingChartContext: null,

  chartPeriod: '3mo',
  chartSectorFilter: 'All',
  activeSidebarNav: 'dashboard',
  activePanelTab: 'news',
  activeTab: 'chart',

  setTicker: (ticker) =>
    set({
      activeTicker:
        ticker == null || String(ticker).trim() === ''
          ? null
          : String(ticker).trim().toUpperCase(),
    }),
  setSession: (id) => set({ activeSessionId: id }),
  setChatDraft: (draft) => set({ chatDraft: draft }),
  setPendingChartContext: (ctx) => set({ pendingChartContext: ctx }),
  setRange: (start, end) => set({ chartRange: { start, end } }),
  clearRange: () => set({ chartRange: { start: null, end: null }, pendingChartContext: null }),

  setChartPeriod: (period) => set({ chartPeriod: period }),
  setChartSectorFilter: (chartSectorFilter) => set({ chartSectorFilter }),
  setSidebarNav: (nav) =>
    set((s) => ({
      activeSidebarNav: nav,
      activePanelTab: nav === 'news' ? 'news' : s.activePanelTab,
    })),
  setPanelTab: (tab) => set({ activePanelTab: tab }),
  setTab: (tab) => set({ activeTab: tab }),
}))
