import { clsx } from 'clsx'

export function SentimentBadge(props: {
  sentiment: string
  score: number
  showScore?: boolean
}) {
  const s = props.sentiment
  const score = props.score

  const { label, colorClass } =
    s === 'bullish'
      ? { label: 'BULL ▲', colorClass: 'bg-bull/15 text-bull border-bull/30' }
      : s === 'bearish'
        ? { label: 'BEAR ▼', colorClass: 'bg-bear/15 text-bear border-bear/30' }
        : { label: 'NEUT ─', colorClass: 'bg-neutral/15 text-neutral border-neutral/30' }

  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1 rounded-full border px-2 py-0.5 font-data text-[11px]',
        colorClass
      )}
      title={`FinBERT score: ${score.toFixed(2)}`}
    >
      <span>{label}</span>
      {props.showScore && s !== 'neutral' ? (
        <span className="opacity-80">{score.toFixed(2)}</span>
      ) : null}
    </span>
  )
}

