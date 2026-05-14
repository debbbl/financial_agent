export function SkeletonPage() {
  return (
    <div className="animate-fade-in space-y-4 p-6">
      <div className="h-8 w-1/3 max-w-xs animate-pulse rounded-lg bg-bg-tertiary" />
      <div className="h-4 w-full max-w-2xl animate-pulse rounded-md bg-bg-tertiary" />
      <div className="h-4 w-5/6 max-w-xl animate-pulse rounded-md bg-bg-tertiary" />
      <div className="mt-8 grid gap-4 md:grid-cols-2">
        <div className="h-64 animate-pulse rounded-xl bg-bg-tertiary" />
        <div className="h-64 animate-pulse rounded-xl bg-bg-tertiary" />
      </div>
    </div>
  )
}
