import { afterEach, describe, expect, it, vi } from "vitest";
import { streamRun } from "./api";
import type { AnswerResetPayload } from "./types";

describe("streamRun persisted cursor", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("requests after_seq and uses SSE id as the authoritative event sequence", async () => {
    const encoder = new TextEncoder();
    const chunks = [
      encoder.encode('id: 8\nevent: answer_delta\ndata: {"delta":"A","delta_seq":1,"generation_id":"g1"}\n\n'),
      encoder.encode('id: 9\nevent: answer_reset\ndata: {"generation_id":"g1","reason":"deadline_exceeded"}\n\n'),
      encoder.encode('id: 10\nevent: workflow_end\ndata: {"status":"error"}\n\n'),
    ];
    let index = 0;
    const fetchMock = vi.fn(async (_input: RequestInfo | URL) => ({
      ok: true,
      body: {
        getReader: () => ({
          read: async () => (index < chunks.length ? { done: false, value: chunks[index++] } : { done: true, value: undefined }),
        }),
      },
    }));
    vi.stubGlobal("fetch", fetchMock);

    const events: { seq: number; type: string; payload: Record<string, unknown> }[] = [];
    await new Promise<void>((resolve, reject) => {
      streamRun(
        "r1",
        (event) => events.push({ seq: event.seq, type: event.type, payload: event.payload }),
        resolve,
        reject,
        7,
      );
    });

    expect(String(fetchMock.mock.calls[0][0])).toContain("/runs/r1/stream?after_seq=7");
    expect(events.map(({ seq, type }) => ({ seq, type }))).toEqual([
      { seq: 8, type: "answer_delta" },
      { seq: 9, type: "answer_reset" },
      { seq: 10, type: "workflow_end" },
    ]);
    expect(events[1].payload as unknown as AnswerResetPayload).toEqual({
      generation_id: "g1",
      reason: "deadline_exceeded",
    });
  });
});
