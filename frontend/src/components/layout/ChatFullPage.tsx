import { ChatPanel } from '../chat/ChatPanel'

export function ChatFullPage() {
  return (
    <section className="card flex min-h-[calc(100vh-8rem)] flex-col p-4">
      <ChatPanel />
    </section>
  )
}
