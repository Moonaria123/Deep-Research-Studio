import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { RunBudgetStrip, latestBudgetSnapshot } from "./RunBudgetStrip";
import type { RunEvent } from "@/lib/types";

const ev = (seq: number, type: string, payload: Record<string, unknown>): RunEvent => ({
  seq,
  type,
  payload,
});

describe("latestBudgetSnapshot", () => {
  it("returns the most recent budget_snapshot payload", () => {
    const events = [
      ev(1, "node_status", { node: "search" }),
      ev(2, "budget_snapshot", { mode: "deep", searches_used: 1, searches_max: 6, stage: "section_1" }),
      ev(3, "budget_snapshot", { mode: "deep", searches_used: 3, searches_max: 6, stage: "gap_1", gap_rounds: 1 }),
    ];
    expect(latestBudgetSnapshot(events)?.searches_used).toBe(3);
    expect(latestBudgetSnapshot([])).toBeUndefined();
  });
});

describe("RunBudgetStrip", () => {
  it("renders nothing without a snapshot or budget", () => {
    const { container } = render(<RunBudgetStrip events={[]} mode="fast" />);
    expect(container.firstChild).toBeNull();
  });

  it("falls back to the run budget before any snapshot arrives", () => {
    render(<RunBudgetStrip events={[]} mode="fast" budget={{ max_searches: 1 }} />);
    expect(screen.getByText("0/1")).toBeInTheDocument();
  });

  it("shows live searches and stage from the latest snapshot", () => {
    const events = [
      ev(1, "budget_snapshot", { mode: "multiple", searches_used: 2, searches_max: 4, stage: "lane_2" }),
    ];
    render(<RunBudgetStrip events={events} mode="multiple" />);
    expect(screen.getByText("2/4")).toBeInTheDocument();
    expect(screen.getByText("lane_2")).toBeInTheDocument();
  });

  it("shows remaining searches, fetched pages, and fallback tier when present", () => {
    const events = [
      ev(1, "budget_snapshot", {
        mode: "deep",
        searches_used: 2,
        searches_max: 5,
        remaining_searches: 3,
        pages_fetched: 4,
        search_tier: "firecrawl_search",
        degraded_fetch: true,
        stage: "analyst",
      }),
    ];
    render(<RunBudgetStrip events={events} mode="deep" />);

    expect(screen.getByText("2/5")).toBeInTheDocument();
    expect(screen.getByText(/3 left/)).toBeInTheDocument();
    expect(screen.getByText(/4 pages/)).toBeInTheDocument();
    expect(screen.getByText("firecrawl search")).toBeInTheDocument();
  });

  it("keeps conditional evidence recovery separate from the ordinary search budget", () => {
    const events = [
      ev(1, "budget_snapshot", {
        mode: "fast",
        searches_used: 1,
        searches_max: 1,
        recovery_searches_used: 1,
        recovery_searches_max: 1,
        stage: "source_recovery",
      }),
    ];
    render(<RunBudgetStrip events={events} mode="fast" />);

    const strip = screen.getByLabelText("Run budget");
    expect(strip).toHaveTextContent("Searches 1/1");
    expect(strip).toHaveTextContent("Recovery searches 1/1");
    expect(screen.getByText("Verifying better sources")).toBeInTheDocument();
  });

  it("renders council vote and synthesis stages with friendly labels", () => {
    const voteEvents = [
      ev(1, "budget_snapshot", { mode: "council", searches_used: 1, searches_max: 4, stage: "vote" }),
    ];
    const { rerender } = render(<RunBudgetStrip events={voteEvents} mode="council" />);
    expect(screen.getByText("Selecting best draft")).toBeInTheDocument();

    const synthesisEvents = [
      ev(2, "budget_snapshot", { mode: "council", searches_used: 1, searches_max: 4, stage: "synthesis" }),
    ];
    rerender(<RunBudgetStrip events={synthesisEvents} mode="council" />);
    expect(screen.getByText("Chair synthesis")).toBeInTheDocument();
  });

  it("renders council drafts and critique stages with friendly labels", () => {
    const draftsEvents = [
      ev(1, "budget_snapshot", { mode: "council", searches_used: 1, searches_max: 4, stage: "drafts" }),
    ];
    const { rerender } = render(<RunBudgetStrip events={draftsEvents} mode="council" />);
    expect(screen.getByText("Drafting member answers")).toBeInTheDocument();

    const critiqueEvents = [
      ev(2, "budget_snapshot", { mode: "council", searches_used: 1, searches_max: 4, stage: "critique" }),
    ];
    rerender(<RunBudgetStrip events={critiqueEvents} mode="council" />);
    expect(screen.getByText("Peer critique round")).toBeInTheDocument();
  });

  it("shows gap rounds for deep runs, preferring the final result value", () => {
    const events = [
      ev(1, "budget_snapshot", { mode: "deep", searches_used: 5, searches_max: 6, stage: "gap_1", gap_rounds: 1 }),
    ];
    render(<RunBudgetStrip events={events} mode="deep" gapRounds={2} />);
    expect(screen.getByText("5/6")).toBeInTheDocument();
    // result.gap_rounds (2) wins over the live snapshot (1).
    expect(screen.getByText("2")).toBeInTheDocument();
  });

  it("hides gap rounds for non-deep modes", () => {
    const events = [
      ev(1, "budget_snapshot", { mode: "multiple", searches_used: 1, searches_max: 4, stage: "lane_1", gap_rounds: 3 }),
    ];
    render(<RunBudgetStrip events={events} mode="multiple" />);
    expect(screen.queryByText(/Gap rounds/)).toBeNull();
  });
});
