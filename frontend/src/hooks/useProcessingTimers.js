import { useEffect, useState } from "react";

export function useElapsedSeconds(active) {
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  useEffect(() => {
    if (!active) {
      setElapsedSeconds(0);
      return undefined;
    }

    const startedAt = Date.now();
    setElapsedSeconds(0);
    const timer = window.setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000));
    }, 1000);

    return () => window.clearInterval(timer);
  }, [active]);

  return elapsedSeconds;
}

export function usePollWhen(active, pollFn, intervalMs = 2000) {
  useEffect(() => {
    if (!active) return undefined;

    const timer = window.setInterval(() => {
      pollFn();
    }, intervalMs);

    return () => window.clearInterval(timer);
  }, [active, pollFn, intervalMs]);
}
