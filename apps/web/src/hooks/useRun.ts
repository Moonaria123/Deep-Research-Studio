import { useEffect, useRef, useState } from "react";
import { api, ApiError, streamRun } from "@/lib/api";
import type { Run, RunEvent } from "@/lib/types";

export interface RunState {
  run?: Run;
  events: RunEvent[];
  status: "streaming" | "pausing" | "paused" | "completed" | "error" | "cancelling" | "cancelled";
  error?: string;
  errorCode?: string;
  draft?: string;
  generationId?: string;
  lastSeq: number;
}

type InternalRunState = RunState & { subscriptionRunId?: string };

/** Subscribe to a run's SSE stream and load the final result when it ends. */
export function useRun(runId: string | undefined): RunState & { cancel: () => void; pause: () => void } {
  const [state, setState] = useState<InternalRunState>({
    events: [],
    status: "streaming",
    lastSeq: 0,
    subscriptionRunId: runId,
  });
  const abortRef = useRef<(() => void) | undefined>(undefined);
  const lastSeqRef = useRef(0);
  const terminalErrorRef = useRef(false);

  useEffect(() => {
    if (!runId) return;
    let disposed = false;
    let reconnectTimer: number | undefined;
    lastSeqRef.current = 0;
    terminalErrorRef.current = false;
    setState({
      events: [],
      status: "streaming",
      lastSeq: 0,
      subscriptionRunId: runId,
    });

    // Load the persisted run snapshot immediately so controls that depend on
    // mode/state (notably Deep Pause) are available while SSE is still open.
    // A stale running snapshot must never demote a newer client-observed
    // pausing/cancelling/terminal state.
    api
      .getRun(runId)
      .then((run) => {
        if (disposed) return;
        setState((current) => {
          const promotedStatus: RunState["status"] =
            run.status === "paused"
              ? "paused"
              : run.status === "pausing"
                ? "pausing"
                : run.status === "cancelled"
                  ? "cancelled"
                  : run.status === "cancelling"
                    ? "cancelling"
                    : run.status === "error"
                      ? "error"
                      : ["completed", "completed_degraded", "partial_timeout", "succeeded"].includes(
                            run.status,
                          )
                        ? "completed"
                        : current.status;
          return {
            ...current,
            run,
            status: promotedStatus,
            error: run.error?.error?.message ?? current.error,
            errorCode: run.error?.error?.code ?? current.errorCode,
          };
        });
      })
      .catch(() => undefined);

    const finalize = async (status: RunState["status"]) => {
      try {
        const run = await api.getRun(runId);
        if (disposed) return;
        // The stream can end without a terminal event (idle timeout, proxy drop,
        // network blip) while the run is still executing. Reconnect instead of
        // misreporting "completed" with an empty answer (DR-ITER-058). Reconnect
        // from the last persisted sequence without clearing the visible draft.
        if (
          status !== "error" &&
          (run.status === "queued" || run.status === "running" || run.status === "pausing")
        ) {
          reconnectTimer = window.setTimeout(connect, 2000);
          return;
        }
        const resolved =
          run.status === "cancelled"
            ? "cancelled"
            : run.status === "paused"
              ? "paused"
            : status === "error" || run.status === "error"
            ? "error"
            : run.status === "cancelling"
              ? "cancelling"
              : "completed";
        setState((s) => ({
          ...s,
          run,
          status: resolved,
          error: run.error?.error?.message,
          errorCode: run.error?.error?.code,
          draft: undefined,
          generationId: undefined,
        }));
      } catch (e) {
        if (disposed) return;
        const code = e instanceof ApiError ? e.code : undefined;
        setState((s) => ({
          ...s,
          status: "error",
          error: (e as Error).message,
          errorCode: code,
        }));
      }
    };

    const onStreamError = (e: Error) => {
      if (disposed) return;
      if (e instanceof ApiError) {
        setState((s) => ({ ...s, status: "error", error: e.message, errorCode: e.code, draft: undefined }));
        return;
      }
      void finalize("streaming");
    };

    const connect = () => {
      if (disposed) return;
      setState((s) => ({ ...s, status: "streaming", error: undefined, errorCode: undefined }));
      abortRef.current = streamRun(
        runId,
        (event) => {
          if (event.seq <= lastSeqRef.current) return;
          lastSeqRef.current = event.seq;
          if (event.type === "error") terminalErrorRef.current = true;
          setState((s) => {
            const next: InternalRunState = {
              ...s,
              events: [...s.events, event],
              lastSeq: event.seq,
            };
            if (event.type === "answer_delta") {
              const delta = String(event.payload.delta ?? "");
              const generationId = String(event.payload.generation_id ?? "");
              next.draft = s.generationId === generationId ? `${s.draft ?? ""}${delta}` : delta;
              next.generationId = generationId;
            } else if (event.type === "answer_reset") {
              const generationId = String(event.payload.generation_id ?? "");
              if (!s.generationId || s.generationId === generationId) {
                next.draft = undefined;
                next.generationId = undefined;
              }
            } else if (event.type === "pause_requested") {
              next.status = "pausing";
            } else if (event.type === "error") {
              const envelope = event.payload.error as { code?: string; message?: string } | undefined;
              next.status = "error";
              next.error = envelope?.message;
              next.errorCode = envelope?.code;
              next.draft = undefined;
              next.generationId = undefined;
            }
            return next;
          });
        },
        () => finalize(terminalErrorRef.current ? "error" : "completed"),
        onStreamError,
        lastSeqRef.current,
      );
    };

    connect();

    return () => {
      disposed = true;
      if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer);
      abortRef.current?.();
    };
  }, [runId]);

  const cancel = () => {
    if (runId) api.cancelRun(runId).catch(() => undefined);
    setState((s) => ({ ...s, status: "cancelling" }));
  };

  const pause = () => {
    if (!runId) return;
    api.pauseRun(runId).catch(() => undefined);
    setState((s) => ({ ...s, status: "pausing" }));
  };

  const { subscriptionRunId, ...publicState } = state;
  const visibleState: RunState =
    subscriptionRunId === runId
      ? publicState
      : { events: [], status: "streaming", lastSeq: 0 };
  return { ...visibleState, cancel, pause };
}
