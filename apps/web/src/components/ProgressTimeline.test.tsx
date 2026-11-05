import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ProgressTimeline } from "./ProgressTimeline";
import type { RunEvent } from "@/lib/types";

const event = (seq: number, node: string, message?: string): RunEvent => ({
  seq,
  type: "node_status",
  payload: { node, ...(message ? { message } : {}) },
});

describe("ProgressTimeline", () => {
  it("prefers the safe user-facing recovery message", () => {
    render(
      <ProgressTimeline
        events={[event(1, "search"), event(2, "source_recovery", "正在验证更合适的来源…")]}
        done={false}
      />,
    );

    expect(screen.getByText("正在验证更合适的来源…")).toBeInTheDocument();
    expect(screen.queryByText("source recovery")).toBeNull();
  });

  it("deduplicates a repeated recovery status while retaining the latest message", () => {
    render(
      <ProgressTimeline
        events={[
          event(1, "source_recovery", "Checking sources…"),
          event(2, "source_recovery", "Verifying better sources…"),
        ]}
        done={false}
      />,
    );

    expect(screen.queryByText("Checking sources…")).toBeNull();
    expect(screen.getAllByText("Verifying better sources…")).toHaveLength(1);
  });
});
