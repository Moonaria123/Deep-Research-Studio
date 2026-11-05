import { describe, expect, it } from "vitest";
import { resolveGraphJumpTarget, resolveWikiEvidenceJumpTarget } from "./graphJump";
import type { KnowledgeGraphNode } from "./types";

function node(partial: Partial<KnowledgeGraphNode> & Pick<KnowledgeGraphNode, "type" | "key">): KnowledgeGraphNode {
  return {
    id: partial.id ?? partial.key,
    title: partial.title ?? "",
    summary: partial.summary ?? "",
    confidence: partial.confidence ?? 1,
    visible: partial.visible ?? true,
    attrs: partial.attrs ?? {},
    ...partial,
  };
}

describe("resolveGraphJumpTarget", () => {
  it("routes report nodes to the Library reports tab deep link", () => {
    const target = resolveGraphJumpTarget(
      node({
        type: "report",
        key: "report:r1",
        attrs: { report_id: "r1", jump_uri: "/library/reports/r1" },
      }),
    );
    expect(target).toBe("/library/reports/r1");
  });

  it("routes note nodes to the Library notes tab deep link", () => {
    const target = resolveGraphJumpTarget(
      node({
        type: "note",
        key: "note:n1",
        attrs: { note_id: "n1", jump_uri: "/library/notes/n1" },
      }),
    );
    expect(target).toBe("/library/notes/n1");
  });

  it("routes source nodes via jump_uri to Library sources deep link", () => {
    const target = resolveGraphJumpTarget(
      node({
        type: "source",
        key: "source:d1",
        attrs: { document_id: "d1", jump_uri: "/library/sources/d1" },
      }),
    );
    expect(target).toBe("/library/sources/d1");
  });

  it("falls back to tab index when report_id is missing", () => {
    expect(resolveGraphJumpTarget(node({ type: "report", key: "report:x" }))).toBe("/library/reports");
  });

  it("routes memory nodes to the cited Memory deep link", () => {
    expect(
      resolveGraphJumpTarget(
        node({ type: "memory", key: "memory:m1", attrs: { memory_id: "m1", jump_uri: "/memory/m1" } }),
      ),
    ).toBe("/memory/m1");
  });
});

describe("resolveWikiEvidenceJumpTarget", () => {
  it("routes source, note, report, message, and memory evidence to safe in-app surfaces", () => {
    expect(resolveWikiEvidenceJumpTarget({ type: "source", id: "source:d1" })).toBe("/library/sources/d1");
    expect(resolveWikiEvidenceJumpTarget({ type: "note", node_key: "note:n1" })).toBe("/library/notes/n1");
    expect(resolveWikiEvidenceJumpTarget({ type: "report", id: "report:r1" })).toBe("/library/reports/r1");
    expect(resolveWikiEvidenceJumpTarget({ type: "message", id: "msg-1" }, { sessionId: "sess-1" })).toBe("/session/sess-1");
    expect(resolveWikiEvidenceJumpTarget({ type: "memory", id: "memory:m1" })).toBe("/memory/m1");
  });

  it("returns null for unknown evidence and context-free messages", () => {
    expect(resolveWikiEvidenceJumpTarget({ type: "unknown", id: "x" })).toBeNull();
    expect(resolveWikiEvidenceJumpTarget({ type: "message", id: "msg-1" })).toBeNull();
  });
});
