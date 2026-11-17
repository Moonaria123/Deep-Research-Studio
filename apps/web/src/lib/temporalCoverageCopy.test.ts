import { describe, expect, it } from "vitest";
import { temporalCoverageCopy } from "./temporalCoverageCopy";

describe("temporalCoverageCopy", () => {
  it.each([
    [
      "unknown coverage",
      "时效范围：现有证据未提供可核验的事实生效或来源发布时间，无法确认其覆盖到当前时点。",
      "Temporal scope: The available evidence does not provide verifiable effective or source publication dates, so coverage through the present cannot be confirmed.",
    ],
    [
      "latest coverage",
      "时效范围：可核验证据支持至 2026-11-15；未使用检索时间替代事实时点。",
      "Temporal scope: Verifiable evidence supports coverage through 2026-11-15. Retrieval time is not used as a substitute for the dates of the facts.",
    ],
    [
      "a future requested date",
      "时效范围：请求时点 2027-01-01 尚未到达；正文仅使用截至 2026-11-15 可核验的最近资料。",
      "Temporal scope: The requested date, 2027-01-01, has not yet arrived. The report uses only the latest verifiable information available through 2026-11-15.",
    ],
    [
      "historical coverage",
      "时效范围：正文按 2024-03-01 的历史状态组织证据与结论。",
      "Temporal scope: The evidence and conclusions are organised around the historical state as of 2024-03-01.",
    ],
    [
      "incomplete historical coverage",
      "时效范围：现有证据不能完整还原 2024-03-01；正文仅陈述可核验的最近边界。",
      "Temporal scope: The available evidence cannot fully establish the state as of 2024-03-01. The report states only the latest verifiable limits of coverage.",
    ],
  ])("translates %s without changing its dates or uncertainty", (_name, statement, expected) => {
    expect(temporalCoverageCopy(statement)).toBe(expected);
  });

  it.each([
    "",
    "Custom coverage statement.",
    "这是一段用户提供的时效说明，保留原文。",
    "时效范围：正文按 2024-03-01 的历史状态组织证据与结论。附加说明。",
  ])("preserves custom content unchanged: %s", (statement) => {
    expect(temporalCoverageCopy(statement)).toBe(statement);
  });
});
