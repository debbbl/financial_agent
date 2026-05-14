/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      animation: {
        'pulse-cyan': 'pulse-cyan 1.5s ease-in-out infinite',
        'skeleton-shimmer': 'skeleton-shimmer 1.2s ease-in-out infinite',
        'slide-in-left': 'slide-in-left 200ms cubic-bezier(0.25, 0.46, 0.45, 0.94) forwards',
        'fade-rise': 'fade-rise 0.45s cubic-bezier(0.16, 1, 0.3, 1) forwards',
      },
      keyframes: {
        'pulse-cyan': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.35' },
        },
        'skeleton-shimmer': {
          '0%': { backgroundPosition: '200% 0' },
          '100%': { backgroundPosition: '-200% 0' },
        },
        'slide-in-left': {
          from: { opacity: '0', transform: 'translateX(-12px)' },
          to: { opacity: '1', transform: 'translateX(0)' },
        },
        'fade-rise': {
          from: { opacity: '0', transform: 'translateY(10px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
}
