/** useRun reconnect semantics: a dropped stream on a still-running run must
 * reconnect instead of misreporting "completed" (DR-ITER-058). */
import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

interface StreamHandlers {
  onEvent: (event: { seq: number; type: string; payload: Record<string, unknown> }) => void;
  onDone?: () => void;
  onError?: (error: Error) => void;
  afterSeq: number;
}

const streamCalls: StreamHandlers[] = [];
let runStatus = "running";

vi.mock("@/lib/api", () => ({
  ApiError: class ApiError extends Error {
    code: string;
    constructor(code: string, message: string) {
      super(message);
      this.code = code;
    }
  },
  api: {
    getRun: vi.fn(async (runId: string) => ({ id: runId, status: runStatus })),
    cancelRun: vi.fn(async () => ({})),
    pauseRun: vi.fn(async () => ({ status: "pausing" })),
  },
  streamRun: vi.fn(
    (
      _runId: string,
      onEvent: StreamHandlers["onEvent"],
      onDone?: () => void,
      onError?: (error: Error) => void,
      afterSeq = 0,
    ) => {
      streamCalls.push({ onEvent, onDone, onError, afterSeq });
      return () => undefined;
    },
  ),
}));

import { api } from "@/lib/api";
import { useRun } from "./useRun";

describe("useRun", () => {
  beforeEach(() => {
    streamCalls.length = 0;
    runStatus = "running";
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("reconnects after ~2s when the stream ends while the run is still running", async () => {
    const { result } = renderHook(() => useRun("r1"));
    expect(streamCalls).toHaveLength(1);

    // Stream drops without a terminal event; the run is still executing.
    await act(async () => {
      streamCalls[0].onDone?.();
    });
    expect(result.current.status).toBe("streaming");
    expect(streamCalls).toHaveLength(1);

    runStatus = "completed";
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    expect(streamCalls).toHaveLength(2);

    // The reconnected stream replays and finishes normally.
    await act(async () => {
      streamCalls[1].onEvent({ seq: 1, type: "workflow_end", payload: { status: "completed" } });
      streamCalls[1].onDone?.();
    });
    expect(result.current.status).toBe("completed");
  });

  it("finalizes immediately when the run is already terminal", async () => {
    runStatus = "completed";
    const { result } = renderHook(() => useRun("r1"));
    await act(async () => {
      streamCalls[0].onEvent({ seq: 1, type: "workflow_end", payload: { status: "completed" } });
      streamCalls[0].onDone?.();
    });
    expect(streamCalls).toHaveLength(1);
    expect(result.current.status).toBe("completed");
  });

  it("treats partial_timeout as terminal while preserving the run status", async () => {
    runStatus = "partial_timeout";
    const { result } = renderHook(() => useRun("r1"));
    await act(async () => {
      streamCalls[0].onEvent({ seq: 1, type: "workflow_end", payload: { status: "partial_timeout", code: "RUN_TIMEOUT" } });
      streamCalls[0].onDone?.();
    });

    expect(streamCalls).toHaveLength(1);
    expect(result.current.status).toBe("completed");
    expect(result.current.run?.status).toBe("partial_timeout");
  });

  it("treats completed_degraded as terminal while preserving the run status", async () => {
    runStatus = "completed_degraded";
    const { result } = renderHook(() => useRun("r1"));
    await act(async () => {
      streamCalls[0].onEvent({ seq: 1, type: "workflow_end", payload: { status: "completed_degraded" } });
      streamCalls[0].onDone?.();
    });

    expect(streamCalls).toHaveLength(1);
    expect(result.current.status).toBe("completed");
    expect(result.current.run?.status).toBe("completed_degraded");
  });

  it("tracks pause_requested and resolves a paused run as a terminal state", async () => {
    const { result } = renderHook(() => useRun("r1"));

    act(() => {
      streamCalls[0].onEvent({ seq: 1, type: "pause_requested", payload: {} });
    });
    expect(result.current.status).toBe("pausing");

    runStatus = "paused";
    await act(async () => {
      streamCalls[0].onEvent({ seq: 2, type: "workflow_end", payload: { status: "paused" } });
      streamCalls[0].onDone?.();
    });

    expect(result.current.status).toBe("paused");
    expect(result.current.run?.status).toBe("paused");
  });

  it("does not leak a paused parent state into a newly selected child run", async () => {
    runStatus = "paused";
    const { result, rerender } = renderHook(
      ({ runId }) => useRun(runId),
      { initialProps: { runId: "parent" } },
    );
    await act(async () => {
      await Promise.resolve();
    });
    expect(result.current.status).toBe("paused");

    runStatus = "running";
    rerender({ runId: "child" });

    expect(result.current.status).toBe("streaming");
    expect(result.current.run).toBeUndefined();
    await act(async () => {
      await Promise.resolve();
    });
  });

  it("requests a persistent pause and immediately reflects pausing", async () => {
    const { result } = renderHook(() => useRun("r1"));
    await act(async () => {
      await Promise.resolve();
    });

    act(() => {
      result.current.pause();
    });

    expect(result.current.status).toBe("pausing");
    expect(api.pauseRun).toHaveBeenCalledWith("r1");
  });

  it("does not reconnect after unmount", async () => {
    const { unmount } = renderHook(() => useRun("r1"));
    await act(async () => {
      streamCalls[0].onDone?.();
    });
    unmount();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000);
    });
    expect(streamCalls).toHaveLength(1);
  });

  it("accumulates ordered deltas, deduplicates replay, and replaces a reset generation", async () => {
    const { result } = renderHook(() => useRun("r1"));
    await act(async () => {
      await Promise.resolve();
    });

    act(() => {
      streamCalls[0].onEvent({ seq: 2, type: "answer_delta", payload: { delta: "First ", delta_seq: 1, generation_id: "g1" } });
      streamCalls[0].onEvent({ seq: 3, type: "answer_delta", payload: { delta: "draft", delta_seq: 2, generation_id: "g1" } });
      streamCalls[0].onEvent({ seq: 3, type: "answer_delta", payload: { delta: " duplicate", delta_seq: 2, generation_id: "g1" } });
    });
    expect(result.current.draft).toBe("First draft");
    expect(result.current.lastSeq).toBe(3);

    act(() => {
      streamCalls[0].onEvent({ seq: 4, type: "answer_reset", payload: { generation_id: "g1", reason: "council_judge_fallback" } });
      streamCalls[0].onEvent({ seq: 5, type: "answer_delta", payload: { delta: "Grounded replacement", delta_seq: 1, generation_id: "g2" } });
    });
    expect(result.current.draft).toBe("Grounded replacement");
    expect(result.current.generationId).toBe("g2");
  });

  it("clears an ungrounded draft when the deadline reset precedes a timeout error", async () => {
    const { result } = renderHook(() => useRun("r1"));
    await act(async () => {
      await Promise.resolve();
    });

    act(() => {
      streamCalls[0].onEvent({
        seq: 1,
        type: "answer_delta",
        payload: { delta: "Unverified draft", delta_seq: 1, generation_id: "g1" },
      });
      streamCalls[0].onEvent({
        seq: 2,
        type: "answer_reset",
        payload: { generation_id: "g1", reason: "deadline_exceeded" },
      });
    });

    expect(result.current.draft).toBeUndefined();
    expect(result.current.generationId).toBeUndefined();

    act(() => {
      streamCalls[0].onEvent({
        seq: 3,
        type: "error",
        payload: { error: { code: "RUN_TIMEOUT", message: "deadline exceeded" } },
      });
    });

    expect(result.current.status).toBe("error");
    expect(result.current.errorCode).toBe("RUN_TIMEOUT");
    expect(result.current.draft).toBeUndefined();
    expect(result.current.lastSeq).toBe(3);
  });

  it("keeps the draft on reconnect and resumes strictly after the persisted cursor", async () => {
    const { result } = renderHook(() => useRun("r1"));
    act(() => {
      streamCalls[0].onEvent({ seq: 7, type: "answer_delta", payload: { delta: "Persisted", delta_seq: 1, generation_id: "g1" } });
    });
    await act(async () => {
      streamCalls[0].onDone?.();
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });

    expect(streamCalls).toHaveLength(2);
    expect(streamCalls[1].afterSeq).toBe(7);
    expect(result.current.draft).toBe("Persisted");
  });

  it("drops the ungrounded draft when the authoritative terminal run arrives", async () => {
    runStatus = "completed";
    const { result } = renderHook(() => useRun("r1"));
    await act(async () => {
      streamCalls[0].onEvent({ seq: 1, type: "answer_delta", payload: { delta: "Draft", delta_seq: 1, generation_id: "g1" } });
      streamCalls[0].onEvent({ seq: 2, type: "workflow_end", payload: { status: "completed" } });
      streamCalls[0].onDone?.();
    });

    expect(result.current.status).toBe("completed");
    expect(result.current.draft).toBeUndefined();
  });

  it("preserves the cancelled terminal state when cancellation arrives as an error event", async () => {
    runStatus = "cancelled";
    const { result } = renderHook(() => useRun("r1"));
    await act(async () => {
      streamCalls[0].onEvent({
        seq: 1,
        type: "error",
        payload: { error: { code: "RUN_CANCELLED", message: "run was cancelled" } },
      });
      streamCalls[0].onDone?.();
    });

    expect(result.current.status).toBe("cancelled");
    expect(result.current.run?.status).toBe("cancelled");
    expect(result.current.draft).toBeUndefined();
  });
});
