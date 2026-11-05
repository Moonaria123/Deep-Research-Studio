/** Resolve the Web API prefix from an optional public origin. */
export function resolveApiBase(publicOrigin: string | undefined): string {
  const origin = (publicOrigin ?? "").trim().replace(/\/+$/, "");
  return origin ? `${origin}/api/v1` : "/api/v1";
}

export const API_BASE = resolveApiBase(import.meta.env.VITE_DEEPRESEARCH_API_ORIGIN);
