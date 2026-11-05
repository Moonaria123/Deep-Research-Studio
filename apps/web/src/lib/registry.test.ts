import { describe, expect, it } from "vitest";
import { chatRefs, embedRefs, modelRefs } from "./registry";
import type { LLMRegistryStatus, ModelKind, ModelProfile } from "./types";

function model(id: string, kind: ModelKind = "chat", enabled = true, alias = ""): ModelProfile {
  return {
    id,
    alias,
    kind,
    context_window: 128000,
    capabilities: { vision: false, ocr: false, web_search: false, thinking: false },
    thinking_budget: null,
    temperature: null,
    dimensions: null,
    enabled,
  };
}

const registry: LLMRegistryStatus = {
  providers: [
    {
      id: "openai",
      label: "OpenAI",
      type: "openai",
      base_url: "https://api.openai.com/v1",
      enabled: true,
      has_key: true,
      models: [model("gpt-4o", "chat", true, "GPT-4o"), model("text-embedding-3-small", "embed")],
    },
    {
      id: "local",
      label: "Local",
      type: "ollama",
      base_url: "http://localhost:11434",
      enabled: false,
      has_key: false,
      models: [model("llama3", "chat")],
    },
  ],
  roles: { default: "openai:gpt-4o", embed: "openai:text-embedding-3-small", memory: "", council: [] },
};

describe("registry refs", () => {
  it("modelRefs flattens enabled providers and models", () => {
    expect(modelRefs(registry)).toEqual(["openai:gpt-4o", "openai:text-embedding-3-small"]);
  });

  it("chatRefs returns only chat models with display labels", () => {
    const refs = chatRefs(registry);
    expect(refs.map((r) => r.ref)).toEqual(["openai:gpt-4o"]);
    expect(refs[0].label).toContain("GPT-4o");
  });

  it("embedRefs returns only embedding models", () => {
    expect(embedRefs(registry).map((r) => r.ref)).toEqual(["openai:text-embedding-3-small"]);
  });

  it("skips disabled providers", () => {
    expect(modelRefs(registry)).not.toContain("local:llama3");
  });

  it("returns empty for undefined registry", () => {
    expect(chatRefs(undefined)).toEqual([]);
    expect(embedRefs(undefined)).toEqual([]);
  });
});
