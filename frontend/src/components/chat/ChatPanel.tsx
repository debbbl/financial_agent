import { clsx } from 'clsx'
import { useEffect, useMemo, useRef, useState } from 'react'

import { useAppStore } from '../../store/useAppStore'
import { useSSEChat } from '../../hooks/useSSEChat'
import { AgentStageBar } from './AgentStageBar'
import { MessageBubble } from './MessageBubble'

export function ChatPanel() {
  const sessionId = useAppStore((s) => s.activeSessionId)
  const ticker = useAppStore((s) => s.activeTicker)
  const { start, end } = useAppStore((s) => s.chartRange)
  const clearRange = useAppStore((s) => s.clearRange)
  const draft = useAppStore((s) => s.chatDraft)
  const setChatDraft = useAppStore((s) => s.setChatDraft)

  const { messages, stages, debateTurns, isStreaming, sendMessage, clearMessages } = useSSEChat(
    sessionId,
    ticker
  )

  const [text, setText] = useState('')
  const maxChars = 2000

  useEffect(() => {
    if (!draft) return
    setText(draft)
    setChatDraft('')
  }, [draft, setChatDraft])

  const listRef = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    const el = listRef.current
    if (!el) return
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
  }, [messages, isStreaming])

  const contextChip = useMemo(() => {
    if (!ticker || !start || !end) return null
    return `${ticker} · ${start} → ${end}`
  }, [end, start, ticker])

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden">
      <div className="shrink-0">
        <AgentStageBar
          stages={stages}
          debate={debateTurns}
          isStreaming={isStreaming || messages.length > 0}
        />
      </div>

      <div
        ref={listRef}
        className="min-h-0 flex-1 overflow-y-auto scroll-smooth rounded-lg border border-border bg-bg px-3 py-3"
      >
        <div className="space-y-3">
          {messages.length === 0 ? (
            <div className="rounded-lg border border-border bg-bg-surface px-4 py-3 text-sm text-text-muted">
              Ask about {ticker ?? 'a ticker'} — or select a chart range and click “Analyze with
              AI”.
            </div>
          ) : null}

          {messages.map((m) => (
            <MessageBubble key={m.id} message={m} isGlobalStreaming={isStreaming} />
          ))}
        </div>
      </div>

      <div className="mt-3 shrink-0 rounded-lg border border-border bg-bg-card p-3">
        <div className="mb-2 flex items-center gap-2">
          {contextChip ? (
            <button
              onClick={() => clearRange()}
              className="rounded-full border border-border bg-bg-surface px-3 py-1 text-xs text-text-muted hover:text-text"
              title="Remove range context"
            >
              [{contextChip}] <span className="ml-1 text-text-dim">×</span>
            </button>
          ) : null}

          <div className="ml-auto flex items-center gap-2">
            <button
              onClick={() => clearMessages()}
              className="rounded-md border border-border bg-transparent px-3 py-1 text-xs text-text-muted hover:bg-bg-surface hover:text-text"
              disabled={isStreaming}
            >
              Clear
            </button>
          </div>
        </div>

        <div className="flex gap-2">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value.slice(0, maxChars))}
            rows={3}
            placeholder="Ask the agent…"
            className={clsx(
              'w-full resize-none rounded-md border border-border bg-bg-surface px-3 py-2 text-sm text-text outline-none',
              'placeholder:text-text-dim focus:border-cyan focus:glow-cyan'
            )}
            onKeyDown={(e) => {
              if (e.key !== 'Enter') return
              if (e.shiftKey) return
              e.preventDefault()
              if (isStreaming) return
              void sendMessage(text)
              setText('')
            }}
          />

          <div className="flex flex-col items-end justify-between">
            <button
              onClick={() => {
                if (isStreaming) return
                void sendMessage(text)
                setText('')
              }}
              disabled={isStreaming || !text.trim()}
              className={clsx(
                'h-9 rounded-md border px-4 text-sm font-medium',
                isStreaming
                  ? 'border-border bg-bg-surface text-text-muted'
                  : 'border-cyan/40 bg-cyan/10 text-cyan hover:glow-cyan',
                (!text.trim() || isStreaming) && 'opacity-60'
              )}
            >
              {isStreaming ? 'Sending…' : 'Send'}
            </button>

            <div className="mt-2 text-xs text-text-dim">
              {text.length}/{maxChars}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

