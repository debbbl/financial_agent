import { useCallback, useMemo, useRef, useState } from 'react'

import { buildChatStreamBody } from '../api/client'
import { useAppStore } from '../store/useAppStore'

export interface AgentStage {
  name: 'researcher' | 'analyst' | 'risk' | 'synthesis'
  label: string
  icon: string
  status: 'idle' | 'running' | 'done' | 'error'
  summary?: string
}

export interface DebateTurn {
  id: string
  speaker: 'analyst' | 'risk'
  content: string
  turn: number
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  isStreaming?: boolean
  timestamp: Date
}

type StreamEvent =
  | { type: 'stage_start'; stage: AgentStage['name'] }
  | { type: 'stage_complete'; stage: AgentStage['name']; summary?: string }
  | { type: 'debate_turn'; speaker: DebateTurn['speaker']; content: string; turn: number }
  | { type: 'synthesis_token'; token: string }
  | { type: 'done'; session_id?: string }
  | { type: 'error'; message: string }

const DEFAULT_STAGES: AgentStage[] = [
  { name: 'researcher', label: 'Researcher', icon: '🔍', status: 'idle' },
  { name: 'analyst', label: 'Analyst', icon: '📊', status: 'idle' },
  { name: 'risk', label: 'Risk', icon: '⚠️', status: 'idle' },
  { name: 'synthesis', label: 'Synthesis', icon: '✍️', status: 'idle' },
]

function uid() {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) return crypto.randomUUID()
  return `${Date.now()}_${Math.random().toString(16).slice(2)}`
}

export function useSSEChat(sessionId: string | null, ticker: string | null) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [stages, setStages] = useState<AgentStage[]>(DEFAULT_STAGES)
  const [debateTurns, setDebateTurns] = useState<DebateTurn[]>([])
  const [isStreaming, setIsStreaming] = useState(false)

  const abortRef = useRef<AbortController | null>(null)

  const clearMessages = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
    setIsStreaming(false)
    setMessages([])
    setStages(DEFAULT_STAGES)
    setDebateTurns([])
  }, [])

  const stageIndex = useMemo(() => {
    const m = new Map<AgentStage['name'], number>()
    DEFAULT_STAGES.forEach((s, i) => m.set(s.name, i))
    return m
  }, [])

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim()
      if (!trimmed) return

      // cancel any previous stream
      abortRef.current?.abort()
      const ac = new AbortController()
      abortRef.current = ac

      const userMsg: ChatMessage = {
        id: uid(),
        role: 'user',
        content: trimmed,
        timestamp: new Date(),
      }

      const assistantId = uid()
      const assistantMsg: ChatMessage = {
        id: assistantId,
        role: 'assistant',
        content: '',
        isStreaming: true,
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, userMsg, assistantMsg])
      setStages(DEFAULT_STAGES)
      setDebateTurns([])
      setIsStreaming(true)

      try {
        const pending = useAppStore.getState().pendingChartContext
        if (pending) useAppStore.getState().setPendingChartContext(null)

        const res = await fetch('/api/v1/chat/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(buildChatStreamBody(sessionId!, ticker, trimmed, pending)),
          signal: ac.signal,
        })

        if (!res.ok || !res.body) {
          throw new Error(`Request failed (${res.status})`)
        }

        const reader = res.body.getReader()
        const decoder = new TextDecoder('utf-8')
        let buffer = ''

        const handleEvent = (evt: StreamEvent) => {
          if (evt.type === 'stage_start') {
            setStages((prev) => {
              const idx = stageIndex.get(evt.stage)
              if (idx == null) return prev
              const next = [...prev]
              next[idx] = { ...next[idx], status: 'running' }
              return next
            })
          }

          if (evt.type === 'stage_complete') {
            setStages((prev) => {
              const idx = stageIndex.get(evt.stage)
              if (idx == null) return prev
              const next = [...prev]
              next[idx] = {
                ...next[idx],
                status: 'done',
                summary: evt.summary ?? next[idx].summary,
              }
              return next
            })
          }

          if (evt.type === 'debate_turn') {
            setDebateTurns((prev) => [
              ...prev,
              {
                id: uid(),
                speaker: evt.speaker,
                content: evt.content,
                turn: evt.turn,
              },
            ])
          }

          if (evt.type === 'synthesis_token') {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, content: m.content + evt.token, isStreaming: true }
                  : m
              )
            )
          }

          if (evt.type === 'done') {
            setIsStreaming(false)
            setMessages((prev) =>
              prev.map((m) => (m.id === assistantId ? { ...m, isStreaming: false } : m))
            )
          }

          if (evt.type === 'error') {
            setIsStreaming(false)
            setStages((prev) => prev.map((s) => (s.status === 'running' ? { ...s, status: 'error' } : s)))
            setMessages((prev) => [
              ...prev.map((m) => (m.id === assistantId ? { ...m, isStreaming: false } : m)),
              {
                id: uid(),
                role: 'assistant',
                content: `Error: ${evt.message}`,
                timestamp: new Date(),
              },
            ])
          }
        }

        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })

          // SSE frames are line-based; we only care about `data: ...`
          const lines = buffer.split(/\r?\n/)
          buffer = lines.pop() ?? ''

          for (const line of lines) {
            const trimmedLine = line.trim()
            if (!trimmedLine.startsWith('data:')) continue
            const payload = trimmedLine.slice(5).trim()
            if (!payload) continue
            try {
              const evt = JSON.parse(payload) as StreamEvent
              handleEvent(evt)
            } catch {
              // ignore malformed JSON line
            }
          }
        }

        setIsStreaming(false)
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? { ...m, isStreaming: false } : m))
        )
      } catch (e: any) {
        if (e?.name === 'AbortError') return
        setIsStreaming(false)
        setMessages((prev) => [
          ...prev.map((m) => (m.id === assistantId ? { ...m, isStreaming: false } : m)),
          {
            id: uid(),
            role: 'assistant',
            content: `Error: ${e?.message ?? 'Chat request failed'}`,
            timestamp: new Date(),
          },
        ])
      } finally {
        abortRef.current = null
      }
    },
    [sessionId, stageIndex, ticker]
  )

  return { messages, stages, debateTurns, isStreaming, sendMessage, clearMessages }
}

