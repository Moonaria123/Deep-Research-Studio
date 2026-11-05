import { useCallback, useEffect, useRef, useState } from "react";
import type { KeyboardEvent, PointerEvent as ReactPointerEvent } from "react";

const DEFAULT_RATIO = 0.48;
const MIN_RATIO = 0.25;
const MAX_RATIO = 0.75;
const STEP = 0.02;
const STORAGE_PREFIX = "deepresearch-session-thread-split-ratio";

function clampRatio(value: number) {
  if (!Number.isFinite(value)) return DEFAULT_RATIO;
  return Math.min(MAX_RATIO, Math.max(MIN_RATIO, value));
}

function storageKey(sessionId?: string) {
  const id = sessionId?.trim();
  return id ? `${STORAGE_PREFIX}:${id}` : "";
}

function readStoredRatio(sessionId?: string) {
  const key = storageKey(sessionId);
  if (!key || typeof window === "undefined") return DEFAULT_RATIO;
  const value = Number.parseFloat(window.localStorage.getItem(key) ?? "");
  return Number.isFinite(value) ? clampRatio(value) : DEFAULT_RATIO;
}

function persistRatio(sessionId: string | undefined, ratio: number) {
  const key = storageKey(sessionId);
  if (!key || typeof window === "undefined") return;
  window.localStorage.setItem(key, String(clampRatio(ratio)));
}

export function useSplitPaneRatio(sessionId?: string) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [ratio, setRatio] = useState(DEFAULT_RATIO);

  useEffect(() => {
    setRatio(readStoredRatio(sessionId));
  }, [sessionId]);

  const updateRatio = useCallback(
    (next: number) => {
      const clamped = clampRatio(next);
      setRatio(clamped);
      persistRatio(sessionId, clamped);
      return clamped;
    },
    [sessionId],
  );

  const ratioFromPointer = useCallback((clientX: number) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0) return null;
    return clampRatio((clientX - rect.left) / rect.width);
  }, []);

  const onPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLElement>) => {
      event.preventDefault();
      event.currentTarget.setPointerCapture?.(event.pointerId);
      const previousUserSelect = document.body.style.userSelect;
      document.body.style.userSelect = "none";
      const initial = ratioFromPointer(event.clientX);
      if (initial !== null) updateRatio(initial);

      const onMove = (moveEvent: PointerEvent) => {
        const next = ratioFromPointer(moveEvent.clientX);
        if (next !== null) updateRatio(next);
      };
      const cleanup = () => {
        document.body.style.userSelect = previousUserSelect;
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", cleanup);
        window.removeEventListener("pointercancel", cleanup);
      };

      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", cleanup, { once: true });
      window.addEventListener("pointercancel", cleanup, { once: true });
    },
    [ratioFromPointer, updateRatio],
  );

  const onKeyDown = useCallback(
    (event: KeyboardEvent<HTMLElement>) => {
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        updateRatio(ratio - STEP);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        updateRatio(ratio + STEP);
      } else if (event.key === "Home") {
        event.preventDefault();
        updateRatio(MIN_RATIO);
      } else if (event.key === "End") {
        event.preventDefault();
        updateRatio(MAX_RATIO);
      }
    },
    [ratio, updateRatio],
  );

  return {
    containerRef,
    ratio,
    separatorProps: {
      role: "separator",
      "aria-orientation": "vertical" as const,
      "aria-valuemin": MIN_RATIO,
      "aria-valuemax": MAX_RATIO,
      "aria-valuenow": Number(ratio.toFixed(2)),
      tabIndex: 0,
      onPointerDown,
      onKeyDown,
    },
  };
}
