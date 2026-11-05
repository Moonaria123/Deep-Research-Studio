import type { LLMRegistryStatus, ModelKind, ModelProfile } from "./types";

/** A selectable ``provider:model`` reference with display metadata. */
export interface ModelRef {
  ref: string;
  providerId: string;
  providerLabel: string;
  model: ModelProfile;
  label: string;
  kind: ModelKind;
}

function buildRefs(registry: LLMRegistryStatus | undefined, kind?: ModelKind): ModelRef[] {
  if (!registry) return [];
  const refs: ModelRef[] = [];
  registry.providers.forEach((p) => {
    if (p.enabled === false) return;
    p.models.forEach((m) => {
      if (m.enabled === false) return;
      if (kind && m.kind !== kind) return;
      const display = m.alias ? `${m.alias} (${m.id})` : m.id;
      refs.push({
        ref: `${p.id}:${m.id}`,
        providerId: p.id,
        providerLabel: p.label || p.id,
        model: m,
        label: `${p.label || p.id} · ${display}`,
        kind: m.kind,
      });
    });
  });
  return refs;
}

/** Flatten a registry into ``provider:model`` reference strings for dropdowns. */
export function modelRefs(registry?: LLMRegistryStatus): string[] {
  return buildRefs(registry).map((r) => r.ref);
}

/** Chat/reasoning model refs (excludes embedders) with display metadata. */
export function chatRefs(registry?: LLMRegistryStatus): ModelRef[] {
  return buildRefs(registry, "chat");
}

/** Embedding model refs (vector index sources) with display metadata. */
export function embedRefs(registry?: LLMRegistryStatus): ModelRef[] {
  return buildRefs(registry, "embed");
}
