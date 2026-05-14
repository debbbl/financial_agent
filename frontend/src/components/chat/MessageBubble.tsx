import ReactMarkdown from 'react-markdown'
import { clsx } from 'clsx'

import type { ChatMessage } from '../../hooks/useSSEChat'

export function MessageBubble(props: { message: ChatMessage; isGlobalStreaming: boolean }) {
  const m = props.message
  const streamingCursor = m.role === 'assistant' && m.isStreaming && props.isGlobalStreaming

  if (m.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-2xl border border-cyan/30 bg-cyan/10 px-4 py-2 text-sm text-text">
          <div className="whitespace-pre-wrap">{m.content}</div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[80%] rounded-2xl bg-bg-card px-4 py-3 text-sm text-text">
        <div className="prose prose-invert max-w-none">
          <ReactMarkdown
            components={{
              h2: ({ children }) => (
                <h2 className="mb-2 border-b border-border pb-1 text-base font-bold text-cyan">
                  {children}
                </h2>
              ),
              h3: ({ children }) => (
                <h3 className="mt-3 text-sm font-semibold text-text">{children}</h3>
              ),
              strong: ({ children }) => (
                <strong className="font-semibold text-cyan/90">{children}</strong>
              ),
              ul: ({ children }) => <ul className="pl-4 marker:text-cyan">{children}</ul>,
              li: ({ children }) => <li className="my-1">{children}</li>,
              p: ({ children }) => (
                <p className="leading-relaxed text-text">{children}</p>
              ),
            }}
          >
            {m.content}
          </ReactMarkdown>

          {streamingCursor ? (
            <span className={clsx('font-data ml-1', 'animate-pulse-cyan')}>▊</span>
          ) : null}
        </div>
      </div>
    </div>
  )
}

