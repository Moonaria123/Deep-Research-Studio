import { describe, expect, it, vi } from "vitest";
import { api, ApiError, streamRun } from "./demoApi";

describe("public demo API", () => {
  it("serves four real, sanitized report fixtures without network access", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    for (const mode of ["fast", "multiple", "deep", "council"] as const) {
      const run = await api.getRun(`demo-${mode}`);
      expect(run.mode).toBe(mode);
      expect(run.result.answer.text.length).toBeGreaterThan(200);
      expect(run.result.answer.citations.length).toBeGreaterThan(0);
      expect(run.result.verification.status).toBe("completed");
    }
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("fails closed for configuration mutations", async () => {
    await expect(api.updateProviders({})).rejects.toMatchObject({ code: "DEMO_READ_ONLY" } satisfies Partial<ApiError>);
  });

  it("fails closed for API reads without an explicit preview fixture", async () => {
    await expect(api.unmappedPreviewRead()).rejects.toMatchObject({ code: "DEMO_UNSUPPORTED" } satisfies Partial<ApiError>);
  });

  it("replays a bounded in-memory run stream", async () => {
    vi.useFakeTimers();
    const events: string[] = [];
    const done = vi.fn();
    streamRun("demo-fast", (event) => events.push(event.type), done);
    await vi.advanceTimersByTimeAsync(900);
    expect(events).toEqual(["status", "status", "done"]);
    expect(done).toHaveBeenCalledOnce();
    vi.useRealTimers();
  });
});
