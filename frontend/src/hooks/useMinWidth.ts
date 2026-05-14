import { useEffect, useState } from 'react'

export function useMinWidth(px: number) {
  const [matches, setMatches] = useState(() => {
    if (typeof window === 'undefined') return false
    return window.matchMedia(`(min-width: ${px}px)`).matches
  })

  useEffect(() => {
    const mq = window.matchMedia(`(min-width: ${px}px)`)
    const onChange = () => setMatches(mq.matches)
    onChange()
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [px])

  return matches
}
