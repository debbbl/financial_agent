import { clsx } from 'clsx'
import { Fragment, useState } from 'react'

import type { AgentStage, DebateTurn } from '../../hooks/useSSEChat'

function StageStatusIcon(props: { status: AgentStage['status'] }) {
  if (props.status === 'done') return <span className="text-bull">✓</span>
  if (props.status === 'error') return <span className="text-bear">×</span>
  if (props.status === 'running')
    return <span className="inline-block h-2 w-2 rounded-full bg-cyan animate-pulse-cyan" />
  return <span className="inline-block h-2 w-2 rounded-full bg-text-dim" />
}

function DebatePill(props: {
  count: number
  expanded: boolean
  onToggle: () => void
}) {
  return (
    <button
      onClick={props.onToggle}
      className={clsx(
        'flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[11px] font-semibold tracking-wider transition-colors',
        'border-violet/40 bg-violet/10 text-violet hover:bg-violet/20'
      )}
      title={props.expanded ? 'Hide debate' : 'Show debate'}
    >
      <span>⚡</span>
      <span>DEBATE</span>
      <span className="rounded-full bg-violet/30 px-1.5 text-[10px] text-text">
        {props.count}
      </span>
      <span className="text-text-muted">{props.expanded ? '▾' : '▸'}</span>
    </button>
  )
}

function DebateBubble(props: { turn: DebateTurn }) {
  const isAnalyst = props.turn.speaker === 'analyst'
  return (
    <div
      className={clsx(
        'flex w-full animate-fade-in',
        isAnalyst ? 'justify-end' : 'justify-start'
      )}
    >
      <div
        className={clsx(
          'max-w-[85%] rounded-2xl border px-3 py-2 text-xs leading-relaxed',
          isAnalyst
            ? 'border-cyan/30 bg-cyan/10 text-cyan'
            : 'border-amber-400/40 bg-amber-400/10 text-amber-300'
        )}
      >
        <div
          className={clsx(
            'mb-1 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-wider',
            isAnalyst ? 'justify-end text-cyan/80' : 'justify-start text-amber-200/80'
          )}
        >
          <span>{isAnalyst ? '📊 Analyst' : '⚠️ Risk'}</span>
          <span className="rounded bg-bg-surface px-1.5 py-0.5 text-text-muted">
            turn {props.turn.turn}
          </span>
        </div>
        <pre className="whitespace-pre-wrap break-words font-sans text-text">
          {props.turn.content}
        </pre>
      </div>
    </div>
  )
}

export function AgentStageBar(props: {
  stages: AgentStage[]
  isStreaming: boolean
  debate?: DebateTurn[]
}) {
  const [open, setOpen] = useState<Record<string, boolean>>({})
  const [debateOpen, setDebateOpen] = useState(false)

  const debate = props.debate ?? []
  const riskDone = props.stages.find((s) => s.name === 'risk')?.status === 'done'
  const showDebatePill = riskDone && debate.length > 0

  if (!props.isStreaming && props.stages.every((s) => s.status === 'idle')) return null

  return (
    <div className="mb-3 rounded-lg border border-border bg-bg-card px-3 py-2">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        {props.stages.map((s, idx) => (
          <Fragment key={s.name}>
            {s.name === 'risk' && showDebatePill ? (
              <>
                <DebatePill
                  count={debate.length}
                  expanded={debateOpen}
                  onToggle={() => setDebateOpen((v) => !v)}
                />
                <span className="text-text-dim">→</span>
              </>
            ) : null}

            <div className="flex items-center gap-2">
              <div
                className={clsx('flex items-center gap-2', s.status === 'running' && 'text-text')}
              >
                <span className="font-data text-xs">{s.icon}</span>
                <span
                  className={clsx('text-xs', s.status === 'idle' ? 'text-text-dim' : 'text-text')}
                >
                  {s.label}
                  {s.status === 'running' ? <span className="ml-1 text-text-muted">...</span> : null}
                </span>
                <StageStatusIcon status={s.status} />
                {s.status === 'done' && s.summary ? (
                  <button
                    onClick={() => setOpen((p) => ({ ...p, [s.name]: !p[s.name] }))}
                    className="ml-1 rounded-md border border-border bg-bg-surface px-2 py-0.5 text-[11px] text-text-muted hover:text-text"
                  >
                    {open[s.name] ? 'Hide' : 'Summary'}
                  </button>
                ) : null}
              </div>

              {idx < props.stages.length - 1 ? (
                <span className="text-text-dim">→</span>
              ) : null}
            </div>
          </Fragment>
        ))}
      </div>

      {props.stages.map((s) =>
        s.status === 'done' && s.summary && open[s.name] ? (
          <div
            key={`${s.name}-summary`}
            className="mt-2 rounded-md border border-border bg-bg-surface px-3 py-2"
          >
            <div className="mb-1 text-[11px] font-semibold tracking-wider text-text-dim">
              {s.label.toUpperCase()} SUMMARY
            </div>
            <pre className="whitespace-pre-wrap font-data text-xs text-text-muted">
              {s.summary}
            </pre>
          </div>
        ) : null
      )}

      {showDebatePill && debateOpen ? (
        <div className="mt-2 animate-fade-in rounded-md border border-violet/30 bg-bg-surface px-3 py-3">
          <div className="mb-2 flex items-center justify-between">
            <div className="text-[11px] font-semibold tracking-wider text-violet">
              ANALYST ↔ RISK DEBATE
            </div>
            <span className="text-[10px] text-text-muted">
              {debate.length} turn{debate.length === 1 ? '' : 's'}
            </span>
          </div>
          <div className="flex flex-col gap-2">
            {debate.map((turn) => (
              <DebateBubble key={turn.id} turn={turn} />
            ))}
          </div>
        </div>
      ) : null}
    </div>
  )
}
