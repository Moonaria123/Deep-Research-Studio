import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { ContextBudgetEditor } from "./ContextBudgetEditor";
import { api } from "@/lib/api";
import type { ContextBudgetConfig, ContextBudgetState } from "@/lib/types";

function config(overrides: Partial<ContextBudgetConfig> = {}): ContextBudgetConfig {
  return {
    preset: "balanced",
    max_chars: 6000,
    turns: 6,
    recent_chars: 3500,
    semantic_chars: 1500,
    memory_chars: 1000,
    recent_share: 58,
    semantic_share: 25,
    memory_share: 17,
    ...overrides,
  };
}

const state: ContextBudgetState = {
  effective: config(),
  saved: null,
  env: null,
  default: config(),
  source: "default",
};

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    api: {
      getContextBudget: vi.fn(),
      updateContextBudget: vi.fn(),
      recommendContextBudget: vi.fn(),
    },
  };
});

describe("ContextBudgetEditor", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.getContextBudget).mockResolvedValue(state);
    vi.mocked(api.updateContextBudget).mockResolvedValue(state);
  });

  it("loads and shows the effective preset as active", async () => {
    render(<ContextBudgetEditor />);
    await waitFor(() => expect(api.getContextBudget).toHaveBeenCalled());
    expect(screen.getByRole("button", { name: /Balanced/ })).toHaveAttribute("aria-pressed", "true");
  });

  it("saves a named preset selection", async () => {
    render(<ContextBudgetEditor />);
    await waitFor(() => expect(api.getContextBudget).toHaveBeenCalled());
    fireEvent.click(screen.getByRole("button", { name: /Follow-up focus/ }));
    fireEvent.click(screen.getByRole("button", { name: /Save budget/ }));
    await waitFor(() =>
      expect(api.updateContextBudget).toHaveBeenCalledWith(
        expect.objectContaining({ preset: "followup" }),
      ),
    );
  });

  it("switches to custom mode and saves shares when a slider moves", async () => {
    render(<ContextBudgetEditor />);
    await waitFor(() => expect(api.getContextBudget).toHaveBeenCalled());
    // Move the recent share to keep the sum at 100 (58 -> 60, others 25/17 => 102; nudge down memory).
    fireEvent.change(screen.getByLabelText("Long-term memory share"), { target: { value: "15" } });
    fireEvent.change(screen.getByLabelText("Recent conversation share"), { target: { value: "60" } });
    fireEvent.click(screen.getByRole("button", { name: /Save budget/ }));
    await waitFor(() =>
      expect(api.updateContextBudget).toHaveBeenCalledWith(
        expect.objectContaining({ preset: "custom", recent_share: 60, memory_share: 15 }),
      ),
    );
  });

  it("fetches and adopts a recommendation", async () => {
    vi.mocked(api.recommendContextBudget).mockResolvedValue({
      recommendation: config({ preset: "long_thread", recent_share: 50, semantic_share: 35, memory_share: 15 }),
      reasons: ["This session has 20 messages: favor semantic recall."],
    });
    render(<ContextBudgetEditor />);
    await waitFor(() => expect(api.getContextBudget).toHaveBeenCalled());
    fireEvent.click(screen.getByRole("button", { name: /Smart recommend/ }));
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("semantic recall"));
    fireEvent.click(screen.getByRole("button", { name: /Adopt/ }));
    expect(screen.getByRole("button", { name: /Long thread/ })).toHaveAttribute("aria-pressed", "true");
  });
});
