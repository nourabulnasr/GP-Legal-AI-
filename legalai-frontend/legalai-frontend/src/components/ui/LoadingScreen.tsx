/**
 * Full-screen loading state for app bootstrap (e.g. while checking auth).
 * Uses the boxes loader animation (see index.css .loader-boxes).
 */
export default function LoadingScreen() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center gap-6 bg-background">
      <div className="flex flex-col items-center gap-6">
        <div className="loader-boxes" aria-hidden>
          <div className="loader-box">
            <div /><div /><div />
          </div>
          <div className="loader-box">
            <div /><div /><div />
          </div>
          <div className="loader-box">
            <div /><div /><div />
          </div>
          <div className="loader-box">
            <div /><div /><div />
          </div>
        </div>
        <p className="text-sm text-muted-foreground">Loading Legato...</p>
      </div>
    </div>
  );
}
