import { PriceChart } from '../chart/PriceChart'
import { ChatPanel } from '../chat/ChatPanel'

export function AnalysisDashboard() {
  return (
    <div className="grid min-h-0 flex-1 gap-4 lg:grid-cols-2 lg:gap-6">
      <section className="card flex min-h-[320px] flex-col overflow-hidden p-4 lg:min-h-[480px]">
        <PriceChart />
      </section>
      <section className="card flex min-h-[320px] flex-col overflow-hidden p-4 lg:min-h-[480px]">
        <ChatPanel />
      </section>
    </div>
  )
}
