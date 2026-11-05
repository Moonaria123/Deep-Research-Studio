import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import type { CouncilMember, GroundedAnswer, Run } from "@/lib/types";
import { api } from "@/lib/api";

const mocks = vi.hoisted(() => ({
  useRun: vi.fn(),
}));

vi.mock("@/hooks/useRun", () => ({
  useRun: mocks.useRun,
}));

vi.mock("@/hooks/useGsapEntrance", () => ({
  useGsapEntrance: () => ({ current: null }),
}));

import { Answer, CouncilView, RunPanel, RunView } from "./RunView";

const baseAnswer: GroundedAnswer = {
  text: "Solar is renewable [c1].",
  citations: [{ id: "c1", document_id: "d1", url: "https://e.com/a", title: "Source A", snippet: "", score: 0.9 }],
  claim_map: { "0": ["c1"] },
  coverage: 0.9,
};

function renderRunView(run: Run | undefined, state: { status?: "streaming" | "pausing" | "paused" | "completed" | "error" | "cancelling" | "cancelled"; error?: string; errorCode?: string; draft?: string } = {}) {
  mocks.useRun.mockReturnValue({
    run,
    events: [],
    status: state.status ?? "completed",
    error: state.error,
    errorCode: state.errorCode,
    draft: state.draft,
    lastSeq: 0,
    cancel: vi.fn(),
    pause: vi.fn(),
  });
  return render(
    <MemoryRouter initialEntries={["/runs/r1"]}>
      <Routes>
        <Route path="/runs/:runId" element={<RunView />} />
      </Routes>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  vi.restoreAllMocks();
  mocks.useRun.mockReset();
});

describe("RunView Answer", () => {
  it("renders confidence tag and keeps limitations out of the body", () => {
    render(
      <Answer
        answer={{ ...baseAnswer, confidence_label: "established", limitations: "Data is from 2023 [c1]." }}
      />,
    );
    expect(screen.getByText("Established")).toBeInTheDocument();
    // Body shows the finding; the limitation text is collapsed (not inline).
    expect(screen.getByText(/Solar is renewable/)).toBeInTheDocument();
    expect(screen.queryByText(/Data is from 2023/)).toBeNull();
    // The dedicated panel toggle is present.
    expect(screen.getByRole("button", { name: /limitations/i })).toBeInTheDocument();
  });

  it("omits the panel and tag when absent", () => {
    render(<Answer answer={baseAnswer} />);
    expect(screen.queryByRole("button", { name: /limitations/i })).toBeNull();
    expect(screen.queryByText("Established")).toBeNull();
  });

  it("renders temporal scope before the body and keeps conflict/freshness notes after it", () => {
    const { container } = render(
      <Answer
        answer={{
          ...baseAnswer,
          confidence_label: "contested",
          conflicts: [
            {
              id: "conflict-1",
              claim_id: "claim-1",
              claim: "Solar cost moved in opposite directions",
              type: "peer_unresolved",
              claim_scope: "same market",
              evidence_ids: ["external:web_document:d1", "local:library_note:n1"],
              resolution: "contested",
              reason: "Same-scope evidence remains unresolved.",
            },
          ],
          limitations: "Coverage remains bounded.",
        }}
        temporalCoverage={{
          intent: "latest",
          checked_at: 10,
          status: "unknown",
          basis: "none",
          evidence_ids: [],
          reason_codes: ["temporal_coverage_unknown"],
          statement: "Temporal scope is not fully verifiable.",
          note_required: true,
        }}
      />,
    );

    const text = container.textContent ?? "";
    expect(text.indexOf("Temporal scope is not fully verifiable.")).toBeLessThan(text.indexOf("Solar is renewable"));
    expect(text.indexOf("Solar is renewable")).toBeLessThan(text.indexOf("证据冲突附注"));
    expect(text.indexOf("证据冲突附注")).toBeLessThan(text.indexOf("时效性附注"));
    expect(screen.getByText(/Solar cost moved in opposite directions/)).toBeInTheDocument();
  });
});

describe("CouncilView", () => {
  const members: CouncilMember[] = [
    { model: "model-x", perspective: "advocate", answer: baseAnswer },
    { model: "model-y", perspective: "skeptic", answer: baseAnswer },
  ];

  it("leads with the structured consensus and shows member perspectives", () => {
    const consensus: GroundedAnswer = {
      ...baseAnswer,
      text: "## Agreement\nSolar is renewable [c1].\n\n## Synthesis\nSolar is renewable [c1].",
      confidence_label: "contested",
    };
    render(
      <CouncilView
        consensus={consensus}
        members={members}
        confidence={0.62}
        breakdown={{ coverage: 0.9, agreement: 0.4 }}
        strength="mixed"
      />,
    );
    expect(screen.getByText("Council consensus")).toBeInTheDocument();
    expect(screen.getByText("Member perspectives")).toBeInTheDocument();
    expect(screen.getByText(/Mixed views/)).toBeInTheDocument();
    expect(screen.getByText("Contested")).toBeInTheDocument();
    // Each model appears in both the report header and the member perspectives list.
    expect(screen.getAllByText("model-x").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("model-y").length).toBeGreaterThanOrEqual(1);
  });

  it("renders parseable consensus as Agreement/Disagreement matrix tables", () => {
    const consensus: GroundedAnswer = {
      ...baseAnswer,
      text:
        "## Agreement\n- Solar is renewable [c1]\n\n## Disagreement\n- Cost trajectory differs [c1]\n\n" +
        "## Unique findings\n- One member flagged storage gaps [c1]\n\n## Synthesis\nSolar scales well [c1].",
      confidence_label: "contested",
    };
    render(<CouncilView consensus={consensus} members={members} strength="strong" />);
    expect(screen.getByRole("heading", { name: /Agreement/ })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /Disagreement/ })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /Unique findings/ })).toBeInTheDocument();
    expect(screen.getByText(/Cost trajectory differs/)).toBeInTheDocument();
    expect(screen.getByText("Strong consensus")).toBeInTheDocument();
  });

  it("falls back to the plain consensus when no matrix headings are present", () => {
    const consensus: GroundedAnswer = {
      ...baseAnswer,
      text: "Solar is renewable and scales well [c1].",
    };
    render(<CouncilView consensus={consensus} members={members} />);
    expect(screen.getByText(/Solar is renewable and scales well/)).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: /Agreement/ })).toBeNull();
  });

  it("does not fabricate an Executive summary block on the consensus", () => {
    const consensus: GroundedAnswer = {
      ...baseAnswer,
      text: "## Agreement\nSolar is renewable [c1].\n\n## Synthesis\nSolar is renewable [c1].",
      // A stray summary/takeaways must NOT surface in the council variant.
      summary: "Solar is renewable [c1].",
      key_takeaways: ["Solar is renewable [c1]"],
      confidence_label: "contested",
    };
    render(<CouncilView consensus={consensus} members={members} />);
    expect(screen.queryByText("Executive summary")).toBeNull();
    expect(screen.queryByText("Key takeaways")).toBeNull();
  });

  it("renders Council temporal and conflict disclosures in the locked order", () => {
    const consensus: GroundedAnswer = {
      ...baseAnswer,
      text:
        "## Agreement\n- Solar is renewable [c1]\n\n## Disagreement\n- Cost differs [c1]\n\n" +
        "## Unique findings\n- Storage matters [c1]\n\n## Synthesis\nSolar scales [c1].",
      confidence_label: "contested",
      conflicts: [
        {
          id: "conflict-council-1",
          claim_id: "claim-council-1",
          claim: "Council evidence disagrees on cost",
          type: "peer_unresolved",
          claim_scope: "same market and period",
          evidence_ids: ["c1", "c2"],
          resolution: "contested",
          reason: "Same-scope sources remain tied.",
        },
      ],
      limitations: "The evidence pack has a bounded limitation.",
    };
    const { container } = render(
      <CouncilView
        consensus={consensus}
        members={members}
        strength="mixed"
        temporalCoverage={{
          intent: "latest",
          checked_at: 10,
          status: "unknown",
          basis: "none",
          evidence_ids: [],
          reason_codes: ["temporal_coverage_unknown"],
          statement: "Council evidence is not fully current.",
          note_required: true,
        }}
      />,
    );

    const text = container.textContent ?? "";
    expect(text.indexOf("Council evidence is not fully current.")).toBeLessThan(
      text.indexOf("Solar is renewable"),
    );
    expect(text.indexOf("Solar is renewable")).toBeLessThan(text.indexOf("证据冲突附注"));
    expect(text.indexOf("证据冲突附注")).toBeLessThan(text.indexOf("时效性附注"));
    expect(text.indexOf("时效性附注")).toBeLessThan(text.indexOf("Limitations"));
    expect(text.indexOf("Limitations")).toBeLessThan(text.indexOf("sources cited"));
    expect(text.indexOf("sources cited")).toBeLessThan(text.indexOf("Member perspectives"));
  });

  it("does not turn low Council agreement into an evidence conflict note", () => {
    render(
      <CouncilView
        consensus={{
          ...baseAnswer,
          text: "Solar is renewable and scales well [c1].",
          confidence_label: "contested",
          conflicts: [],
        }}
        members={members}
        strength="divergent"
      />,
    );

    expect(screen.getByText("Contested")).toBeInTheDocument();
    expect(screen.queryByLabelText("证据冲突附注")).toBeNull();
  });
});

describe("RunView T10 states", () => {
  it("keeps the structured brief for historical results without research_intent or T49 diagnostics", () => {
    renderRunView({
      id: "r1",
      session_id: "s1",
      mode: "fast",
      query: "Solar outlook",
      status: "succeeded",
      result: {
        run_id: "r1",
        mode: "fast",
        status: "succeeded",
        answer: {
          ...baseAnswer,
          summary: "A concise solar summary.",
          key_takeaways: ["Solar remains renewable [c1]."],
        },
        sources: [],
      },
    } as Run);

    expect(screen.getByText("Executive summary")).toBeInTheDocument();
    expect(screen.getByText("Key takeaways")).toBeInTheDocument();
  });

  it("labels live answer text as an ungrounded draft and echoes effective search locale", () => {
    renderRunView(
      {
        id: "r1",
        session_id: "s1",
        mode: "fast",
        query: "Solar outlook",
        status: "running",
        metadata: { effective_locale: { country: "SG", search_lang: "zh", ui_language: "zh-Hans-SG" } },
      } as Run,
      { status: "streaming", draft: "Streaming solar answer" },
    );

    expect(screen.getByRole("status", { name: /streaming answer draft/i })).toHaveTextContent(/grounding in progress/i);
    expect(screen.getByText(/Streaming solar answer/)).toBeInTheDocument();
    expect(screen.getByLabelText("Search locale")).toHaveTextContent("SG · zh · zh-Hans-SG");
  });

  it("shows KG candidate diagnostics without presenting KG as a fourth evidence source", () => {
    renderRunView({
      id: "r1",
      session_id: "s1",
      mode: "deep",
      query: "Solar outlook",
      status: "succeeded",
      result: {
        run_id: "r1",
        mode: "deep",
        status: "succeeded",
        answer: baseAnswer,
        sources: [],
        kg_retrieval: {
          requested: true,
          effective: true,
          attempted: true,
          applied: true,
          snapshot_version_id: "kgv-1",
          candidate_count: 2,
          hydrated_count: 1,
          adopted_evidence_ids: ["external:web_document:d1"],
          reason_codes: [],
        },
      },
    } as Run);

    expect(screen.getByLabelText("Knowledge graph retrieval")).toHaveTextContent("evidence adopted");
    expect(screen.getByLabelText("Knowledge graph retrieval")).toHaveTextContent("2 candidates · 1 hydrated");
    expect(screen.getByText(/KG is not a fourth evidence source/i)).toBeInTheDocument();
  });

  it("shows a partial-timeout banner while rendering the available result", () => {
    renderRunView({
      id: "r1",
      session_id: "s1",
      mode: "deep",
      query: "Solar outlook",
      status: "partial_timeout",
      result: {
        run_id: "r1",
        mode: "deep",
        status: "partial_timeout",
        answer: baseAnswer,
        sources: [],
        sections: [],
        completion_reason: "timeout",
      },
    } as Run);

    expect(screen.getByRole("status", { name: /partial result/i })).toHaveTextContent(/timed out/i);
    expect(screen.getByText(/Solar is renewable/)).toBeInTheDocument();
  });

  it("renders friendly copy for T10 error codes", () => {
    renderRunView(undefined, {
      status: "error",
      errorCode: "RUN_TIMEOUT",
      error: "raw timeout message",
    });

    expect(screen.getByRole("alert")).toHaveTextContent(/timed out before it could complete/i);
  });

  it("renders a metadata-only timeout error without requiring a result body", () => {
    renderRunView(
      {
        id: "r1",
        session_id: "s1",
        mode: "fast",
        query: "Deadline-bound query",
        status: "error",
        metadata: {
          phase_metrics: [
            {
              plan_id: "p1",
              lane: "external",
              query: "Deadline-bound query",
              stage: "search",
              provider: "fake",
              attempt: 1,
              started_at: 10,
              finished_at: 11,
              duration_ms: 1000,
              status: "timeout",
              deadline_remaining_before_ms: 900,
              deadline_remaining_after_ms: 0,
            },
          ],
          deadline_summary: {
            requested_at: 9,
            deadline_at: 11,
            worker_started_at: 10,
            writer_started_at: null,
            terminal_at: 11,
            queue_delay_ms: 1000,
            writer_reserve_s: 8,
            finalize_reserve_s: 2,
            cleanup_reserve_s: 2,
            deadline_met: false,
            cleanup_overrun_ms: 0,
          },
          retrieval_budget: {
            results_per_query: 6,
            candidate_count: 0,
            fetches_max: 1,
            fetches_used: 0,
            fetch_concurrency: 1,
            writer_reserve_s: 8,
            finalize_reserve_s: 2,
            cleanup_reserve_s: 2,
            recovery_fetches_max: 1,
            recovery_fetches_used: 0,
          },
        },
      } as Run,
      { status: "error", errorCode: "RUN_TIMEOUT", error: "raw timeout message" },
    );

    expect(screen.getByRole("alert")).toHaveTextContent(/timed out before it could complete/i);
    expect(screen.queryByRole("status", { name: /streaming answer draft/i })).toBeNull();
    expect(screen.getByText("Deadline-bound query")).toBeInTheDocument();
  });

  it("shows secret-free writer diagnostics for an error without a result body", () => {
    renderRunView(
      {
        id: "r1",
        session_id: "s1",
        mode: "fast",
        query: "Poe empty output",
        status: "error",
        metadata: {
          writer_diagnostics: {
            classification: "length_without_visible_output",
            provider: "poe",
            model: "Claude-Sonnet",
            finish_reason: "length",
            visible_chars: 0,
            fallback_scope: "same_provider_sync",
            attempts: [
              {
                attempt: 0,
                provider: "poe",
                call_mode: "stream",
                classification: "length_without_visible_output",
                status: "empty",
              },
            ],
            call_options: {
              applied: { reasoning_effort: "extra_body" },
              ignored: { response_format: "unsupported_by_poe_chat" },
              visible_output_target: 600,
              reasoning_reserve_tokens: 4096,
              provider_output_cap: 4696,
            },
          },
        },
      } as Run,
      { status: "error", error: "The writer returned no visible answer." },
    );

    expect(screen.getByLabelText("Writer diagnostics")).toHaveTextContent(
      /length_without_visible_output/i,
    );
    expect(screen.getByText(/Parameters applied:/)).toHaveTextContent(/reasoning_effort/);
    expect(screen.getByText(/Parameters applied:/)).toHaveTextContent(/response_format/);
    expect(screen.getByText(/Output cap:/)).toHaveTextContent(
      "visible target 600 + reasoning reserve 4096 = provider cap 4696 tokens",
    );
  });

  it("does not render an empty writer diagnostic shell before the writer starts", () => {
    renderRunView(
      {
        id: "r1",
        session_id: "s1",
        mode: "fast",
        query: "No evidence",
        status: "error",
        metadata: { writer_diagnostics: {} },
      } as unknown as Run,
      { status: "error", errorCode: "GROUNDING_REQUIRED", error: "no evidence" },
    );

    expect(screen.queryByLabelText("Writer diagnostics")).toBeNull();
  });

  it("explains exhausted relevance recovery without exposing the raw grounding error", () => {
    renderRunView(undefined, {
      status: "error",
      errorCode: "GROUNDING_REQUIRED",
      error: "coverage_below_threshold: internal diagnostic",
    });

    expect(screen.getByRole("alert")).toHaveTextContent(/could not establish enough relevant/i);
    expect(screen.getByRole("alert")).not.toHaveTextContent(/internal diagnostic/i);
  });

  it("shows cancellation as a terminal state without retaining a draft", () => {
    renderRunView(
      {
        id: "r1",
        session_id: "s1",
        mode: "fast",
        query: "Cancelled query",
        status: "cancelled",
      } as Run,
      { status: "cancelled", draft: undefined },
    );

    expect(screen.getByRole("status", { name: /cancelled run/i })).toHaveTextContent(/cancelled/i);
    expect(screen.queryByRole("status", { name: /streaming answer draft/i })).toBeNull();
  });

  it("offers Pause only for a running Deep run and explains the safe-boundary wait", () => {
    const pause = vi.fn();
    mocks.useRun.mockReturnValue({
      run: {
        id: "r1",
        session_id: "s1",
        mode: "deep",
        query: "Deep query",
        status: "running",
      } as Run,
      events: [],
      status: "streaming",
      lastSeq: 0,
      cancel: vi.fn(),
      pause,
    });

    const rendered = render(
      <MemoryRouter initialEntries={["/runs/r1"]}>
        <Routes>
          <Route path="/runs/:runId" element={<RunView />} />
        </Routes>
      </MemoryRouter>,
    );
    fireEvent.click(screen.getByRole("button", { name: /^pause$/i }));
    expect(pause).toHaveBeenCalledTimes(1);

    mocks.useRun.mockReturnValue({
      run: {
        id: "r1",
        session_id: "s1",
        mode: "deep",
        query: "Deep query",
        status: "pausing",
      } as Run,
      events: [],
      status: "pausing",
      lastSeq: 0,
      cancel: vi.fn(),
      pause,
    });
    rendered.rerender(
      <MemoryRouter initialEntries={["/runs/r1"]}>
        <Routes>
          <Route path="/runs/:runId" element={<RunView />} />
        </Routes>
      </MemoryRouter>,
    );
    expect(screen.getByText(/pausing at the next safe checkpoint/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^pause$/i })).toBeNull();
  });

  it("continues a paused checkpoint into a child run with revision and lineage budget", async () => {
    vi.spyOn(api, "getRunCheckpoint").mockResolvedValue({
      run_id: "r1",
      lineage_id: "lineage-1",
      revision: 3,
      schema_version: 1,
      stage: "round_complete",
      disposition: "ready",
      integrity: "valid",
      resume_readiness: "degraded",
      reason_codes: ["source_tombstoned"],
      claimed_child_run_id: null,
      updated_at: 1_800_000_000,
      budget_usage: {
        searches_used: 4,
        searches_max: 8,
        pages_fetched: 2,
        recovery_searches_used: 1,
        recovery_searches_max: 1,
      },
    });
    vi.spyOn(api, "continueRun").mockResolvedValue({
      run_id: "child-1",
      session_id: "s1",
      status: "queued",
      parent_run_id: "r1",
      idempotent_replay: false,
    });
    const onChildRun = vi.fn();
    mocks.useRun.mockReturnValue({
      run: {
        id: "r1",
        session_id: "s1",
        mode: "deep",
        query: "Deep query",
        status: "paused",
      } as Run,
      events: [],
      status: "paused",
      lastSeq: 0,
      cancel: vi.fn(),
      pause: vi.fn(),
    });

    render(
      <MemoryRouter>
        <RunPanel runId="r1" onChildRun={onChildRun} />
      </MemoryRouter>,
    );

    expect(await screen.findByRole("status", { name: /deep checkpoint/i })).toHaveTextContent(
      /round_complete/i,
    );
    expect(screen.getByText(/source_tombstoned/i)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText(/additional search rounds/i), {
      target: { value: "2" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^continue$/i }));

    await waitFor(() =>
      expect(api.continueRun).toHaveBeenCalledWith(
        "r1",
        expect.objectContaining({
          expected_checkpoint_revision: 3,
          action: "continue",
          incremental_budget: { max_searches: 2 },
          idempotency_key: expect.stringMatching(/^checkpoint-r1-continue-/),
        }),
      ),
    );
    expect(onChildRun).toHaveBeenCalledWith("child-1");
  });

  it("revises only the remaining plan and sends the incremental lineage budget", async () => {
    vi.spyOn(api, "getRunCheckpoint").mockResolvedValue({
      run_id: "r1",
      lineage_id: "lineage-1",
      revision: 4,
      schema_version: 1,
      stage: "round_complete",
      disposition: "ready",
      integrity: "valid",
      resume_readiness: "ready",
      reason_codes: [],
      claimed_child_run_id: null,
      budget_usage: {
        searches_used: 3,
        searches_max: 6,
      },
    });
    vi.spyOn(api, "continueRun").mockResolvedValue({
      run_id: "child-revised",
      session_id: "s1",
      status: "queued",
      parent_run_id: "r1",
      idempotent_replay: false,
    });
    const onChildRun = vi.fn();
    mocks.useRun.mockReturnValue({
      run: {
        id: "r1",
        session_id: "s1",
        mode: "deep",
        query: "Deep query",
        status: "paused",
      } as Run,
      events: [],
      status: "paused",
      lastSeq: 0,
      cancel: vi.fn(),
      pause: vi.fn(),
    });

    render(
      <MemoryRouter>
        <RunPanel runId="r1" onChildRun={onChildRun} />
      </MemoryRouter>,
    );

    await screen.findByRole("status", { name: /deep checkpoint/i });
    fireEvent.change(screen.getByLabelText(/additional search rounds/i), {
      target: { value: "2" },
    });
    fireEvent.change(screen.getByLabelText(/revised remaining sections/i), {
      target: { value: "Evidence update\nImplications" },
    });
    fireEvent.click(screen.getByRole("button", { name: /revise remaining plan/i }));

    await waitFor(() =>
      expect(api.continueRun).toHaveBeenCalledWith(
        "r1",
        expect.objectContaining({
          expected_checkpoint_revision: 4,
          action: "revise",
          incremental_budget: { max_searches: 2 },
          remaining_plan: {
            sections: ["Evidence update", "Implications"],
          },
          idempotency_key: expect.stringMatching(/^checkpoint-r1-revise-/),
        }),
      ),
    );
    expect(onChildRun).toHaveBeenCalledWith("child-revised");
  });

  it("shows verification boundaries while preserving every report action", () => {
    renderRunView({
      id: "r1",
      session_id: "s1",
      mode: "fast",
      query: "Latest solar outlook",
      status: "completed_degraded",
      result: {
        run_id: "r1",
        mode: "fast",
        status: "completed_degraded",
        answer: baseAnswer,
        degraded_reasons: ["temporal_coverage_unknown"],
        unverified_items: [
          { question: "Does the outlook include the latest quarter?", reason: "evidence_insufficient" },
        ],
      },
    } as Run);

    expect(screen.getByRole("status", { name: /degraded result/i })).toHaveTextContent(/available for use/i);
    expect(screen.getByRole("list", { name: /unverified items/i })).toHaveTextContent(/latest quarter/i);
    expect(screen.getByRole("list", { name: /unverified items/i })).toHaveTextContent(/did not establish/i);
    expect(screen.getByText(/Solar is renewable/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^copy$/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /export/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /save to memory/i })).toBeEnabled();
  });

  it("allows copying a grounded partial result with its disclosure", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
    renderRunView({
      id: "r1",
      session_id: "s1",
      mode: "fast",
      query: "Latest solar outlook",
      status: "completed_degraded",
      result: {
        run_id: "r1",
        mode: "fast",
        status: "completed_degraded",
        answer: baseAnswer,
        unverified_items: [
          { question: "Does it cover the latest quarter?", reason: "verification_incomplete" },
        ],
      },
    } as Run);

    fireEvent.click(screen.getByRole("button", { name: /^copy$/i }));

    await waitFor(() => expect(writeText).toHaveBeenCalledTimes(1));
    expect(writeText.mock.calls[0][0]).toContain("## Items to verify");
    expect(writeText.mock.calls[0][0]).toContain("Does it cover the latest quarter?");
  });

  it("shows the report immediately with a spinning confidence indicator while verification runs", () => {
    renderRunView({
      id: "r1",
      session_id: "s1",
      mode: "fast",
      query: "Latest solar outlook",
      status: "completed",
      result: {
        run_id: "r1",
        mode: "fast",
        status: "completed",
        answer: baseAnswer,
        confidence: 0.8,
        verification: { status: "running" },
      },
    } as Run);

    expect(screen.getByText(/Solar is renewable/)).toBeInTheDocument();
    expect(screen.getByRole("status", { name: "正在核验报告" })).toHaveTextContent("正在核验报告置信度");
    expect(screen.getByRole("button", { name: /export/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /save to memory/i })).toBeEnabled();
    expect(screen.queryByText(/confidence 80%/i)).toBeNull();
  });

  it("hides Council numeric confidence until verification finishes", () => {
    renderRunView({
      id: "r1",
      session_id: "s1",
      mode: "council",
      query: "Council assessment",
      status: "completed",
      result: {
        run_id: "r1",
        mode: "council",
        status: "completed",
        answer: baseAnswer,
        members: [],
        confidence: 0.62,
        verification: { status: "pending" },
      },
    } as Run);

    expect(screen.getByRole("status", { name: "正在核验报告" })).toBeInTheDocument();
    expect(screen.queryByText(/Consensus strength/i)).toBeNull();
    expect(screen.queryByText("62%")).toBeNull();
  });

  it("polls a delivered report until its verification result is available", async () => {
    vi.useFakeTimers();
    try {
      const runningRun = {
        id: "r1",
        session_id: "s1",
        mode: "fast",
        query: "Latest solar outlook",
        status: "completed",
        result: {
          run_id: "r1",
          mode: "fast",
          status: "completed",
          answer: baseAnswer,
          verification: { status: "running" },
        },
      } as Run;
      vi.spyOn(api, "getRun").mockResolvedValue({
        ...runningRun,
        result: {
          ...runningRun.result!,
          answer: { ...baseAnswer, confidence_label: "established" },
          verification: { status: "completed", completed_at: 20 },
        },
      });
      renderRunView(runningRun);

      await act(async () => {
        await vi.advanceTimersByTimeAsync(1000);
      });

      expect(api.getRun).toHaveBeenCalledWith("r1");
      expect(screen.getByRole("status", { name: "核验完成" })).toBeInTheDocument();
      expect(screen.queryByRole("status", { name: "正在核验报告" })).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it("shows a non-modal completion notice and persists low-confidence acknowledgement", async () => {
    vi.spyOn(api, "acknowledgeRunConfidence").mockResolvedValue({ acknowledged: true, acknowledged_at: 10 });
    renderRunView({
      id: "r1",
      session_id: "s1",
      mode: "fast",
      query: "Latest solar outlook",
      status: "completed",
      result: {
        run_id: "r1",
        mode: "fast",
        status: "completed",
        answer: { ...baseAnswer, confidence_label: "uncertain" },
        verification: { status: "completed", completed_at: 10 },
        confidence_warning: { reasons: ["evidence_insufficient", "internal_new_reason"], acknowledged: false },
      },
    } as Run);

    expect(screen.getByRole("status", { name: "核验完成" })).toHaveTextContent("置信度信息已更新");
    fireEvent.click(screen.getByRole("button", { name: "查看低置信度警告" }));
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "低置信度警告" })).toBeInTheDocument();
    expect(screen.getByRole("list", { name: "低置信度原因" })).toHaveTextContent("部分内容尚缺乏充分证据支持");
    expect(screen.getByRole("list", { name: "低置信度原因" })).toHaveTextContent("请核查关键结论");
    expect(screen.queryByText("internal_new_reason")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "确认并忽略" }));

    await waitFor(() => expect(api.acknowledgeRunConfidence).toHaveBeenCalledWith("r1"));
    expect(screen.getByRole("button", { name: "低置信度风险已确认" })).toBeInTheDocument();
  });

  it("starts regeneration from the low-confidence warning and opens the new run", async () => {
    vi.spyOn(api, "regenerateRun").mockResolvedValue({ run_id: "r2", session_id: "s1", status: "queued" });
    mocks.useRun.mockReturnValue({
      run: {
        id: "r1",
        session_id: "s1",
        mode: "fast",
        query: "Latest solar outlook",
        status: "completed",
        result: {
          run_id: "r1",
          mode: "fast",
          status: "completed",
          answer: { ...baseAnswer, confidence_label: "uncertain" },
          verification: { status: "completed" },
          confidence_warning: { reasons: ["unsupported_claim"], acknowledged: false },
        },
      } as Run,
      events: [],
      status: "completed",
      lastSeq: 0,
      cancel: vi.fn(),
      pause: vi.fn(),
    });

    render(
      <MemoryRouter initialEntries={["/run/r1"]}>
        <Routes>
          <Route path="/run/:runId" element={<RunView />} />
        </Routes>
      </MemoryRouter>,
    );
    fireEvent.click(screen.getByRole("button", { name: "查看低置信度警告" }));
    fireEvent.click(screen.getByRole("button", { name: "重新出具" }));

    await waitFor(() => expect(api.regenerateRun).toHaveBeenCalledWith("r1"));
  });

  it("does not expose copy or export actions for an error without a final result", () => {
    renderRunView(
      {
        id: "r1",
        session_id: "s1",
        mode: "fast",
        query: "Failed query",
        status: "error",
      } as Run,
      { status: "error", errorCode: "GROUNDING_REQUIRED" },
    );

    expect(screen.getByRole("button", { name: /^copy$/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /export/i })).toBeDisabled();
  });

  it("treats an explicit empty final source snapshot as authoritative", () => {
    renderRunView({
      id: "r1",
      session_id: "s1",
      mode: "multiple",
      query: "Focused answer",
      status: "succeeded",
      result: {
        run_id: "r1",
        mode: "multiple",
        status: "succeeded",
        answer: baseAnswer,
        sources: [],
        lanes: [
          {
            query: "irrelevant lane",
            sources: [
              {
                chunk_id: 2,
                document_id: "rejected",
                url: "https://e.com/rejected",
                title: "Rejected evidence",
                text: "unrelated",
                score: 0.9,
              },
            ],
          },
        ],
      },
    } as Run);

    expect(screen.queryByText("Rejected evidence")).toBeNull();
    expect(screen.getByText("0 sources")).toBeInTheDocument();
  });

  it("shows completed recovery metadata without exposing rejected evidence text", () => {
    renderRunView({
      id: "r1",
      session_id: "s1",
      mode: "fast",
      query: "SQLite FTS5",
      status: "succeeded",
      result: {
        run_id: "r1",
        mode: "fast",
        status: "succeeded",
        answer: baseAnswer,
        sources: [],
        evidence_eligibility: [
          {
            evidence_id: "rejected",
            decision: "rejected",
            relevance: "unrelated",
            authority_status: "unsatisfied",
            reason_codes: ["focus_mismatch"],
            matched_anchors: [],
            provenance_family: "external:web_document:rejected",
          },
        ],
        remediation: {
          attempted: true,
          attempt_index: 1,
          trigger: "all_rejected",
          action: "retrieve_and_rewrite",
          focused_query: "SQLite FTS5",
          excluded_evidence_ids: ["rejected"],
          excluded_provenance_roots: ["external:web_document:rejected"],
          lane_ids: ["recovery-1-local", "recovery-1-memory", "recovery-1-external"],
          new_evidence_ids: [],
          status: "completed",
        },
        recovery_searches_max: 1,
        recovery_searches_used: 1,
      },
    } as Run);

    expect(screen.getByRole("status", { name: /evidence recovery/i })).toHaveTextContent(/local, memory, and external/i);
    expect(screen.getByRole("status", { name: /evidence recovery/i })).toHaveTextContent(/Rejected 1/i);
    expect(screen.queryByText(/focus_mismatch/i)).toBeNull();
  });
});
