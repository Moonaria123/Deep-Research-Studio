// Translate only the backend's fixed system notices, including saved reports.
// Custom statements and report content must retain their original language.
export function temporalCoverageCopy(statement: string): string {
  if (statement === "时效范围：现有证据未提供可核验的事实生效或来源发布时间，无法确认其覆盖到当前时点。") {
    return "Temporal scope: The available evidence does not provide verifiable effective or source publication dates, so coverage through the present cannot be confirmed.";
  }

  const latest = statement.match(/^时效范围：可核验证据支持至 (\d{4}-\d{2}-\d{2})；未使用检索时间替代事实时点。$/);
  if (latest) {
    return `Temporal scope: Verifiable evidence supports coverage through ${latest[1]}. Retrieval time is not used as a substitute for the dates of the facts.`;
  }

  const future = statement.match(/^时效范围：请求时点 (\d{4}-\d{2}-\d{2}) 尚未到达；正文仅使用截至 (\d{4}-\d{2}-\d{2}) 可核验的最近资料。$/);
  if (future) {
    return `Temporal scope: The requested date, ${future[1]}, has not yet arrived. The report uses only the latest verifiable information available through ${future[2]}.`;
  }

  const historical = statement.match(/^时效范围：正文按 (\d{4}-\d{2}-\d{2}) 的历史状态组织证据与结论。$/);
  if (historical) {
    return `Temporal scope: The evidence and conclusions are organised around the historical state as of ${historical[1]}.`;
  }

  const incomplete = statement.match(/^时效范围：现有证据不能完整还原 (\d{4}-\d{2}-\d{2})；正文仅陈述可核验的最近边界。$/);
  if (incomplete) {
    return `Temporal scope: The available evidence cannot fully establish the state as of ${incomplete[1]}. The report states only the latest verifiable limits of coverage.`;
  }

  return statement;
}
