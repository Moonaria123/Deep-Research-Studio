import { describe, expect, it } from "vitest";

import { evidenceArtifactLabel, evidenceHref, evidenceKey } from "./evidenceRoutes";
import type { EvidenceSource } from "./types";

describe("library document evidence routes", () => {
  it("uses a stable local key and an internal Library deep link", () => {
    const source: EvidenceSource = {
      chunk_id: 7,
      document_id: "doc-local",
      url: null,
      title: "Imported evidence",
      text: "Local evidence body",
      score: 1,
      origin: "local",
      artifact_type: "library_document",
      artifact_id: "doc-local",
      provenance_refs: ["local:library_document:doc-local"],
    };

    expect(evidenceKey(source)).toBe("local:library_document:doc-local");
    expect(evidenceHref(source)).toEqual({
      href: "/library/sources/doc-local",
      external: false,
    });
    expect(evidenceArtifactLabel(source)).toBe("document");
  });
});
