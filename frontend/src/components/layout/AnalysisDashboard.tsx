import { PriceChart } from '../chart/PriceChart'
import { ChatPanel } from '../chat/ChatPanel'

/** Fixed viewport height: top chrome + padding; chart/AI columns never resize with chat content. Lower the subtractor (e.g. 120→88) for a taller chat column. */
const DASHBOARD_VIEWPORT_H = 'h-[calc(100dvh-88px)]'

export function AnalysisDashboard() {
  return (
    <div
      className={`flex min-h-0 flex-col gap-4 overflow-hidden ${DASHBOARD_VIEWPORT_H} lg:grid lg:grid-cols-2 lg:grid-rows-1 lg:gap-5`}
    >
      <section
        className={`card flex min-h-0 flex-1 flex-col overflow-hidden p-4 lg:h-full lg:min-h-0`}
      >
        <PriceChart />
      </section>
      <section
        className={`card flex min-h-0 flex-1 flex-col overflow-hidden p-4 lg:h-full lg:min-h-0`}
      >
        <ChatPanel />
      </section>
    </div>
  )
}
