import { useMemo, useState } from "react";
import type {
  KnowledgeGraphNode,
  KnowledgeGraphNodeDetail,
  KnowledgeGraphWikiCrossLink,
  KnowledgeGraphWikiEvidenceRef,
  KnowledgeGraphWikiMap,
  KnowledgeGraphWikiTopic,
} from "@/lib/types";
import { stringAttr } from "@/components/graph/GraphNode";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { resolveWikiEvidenceJumpTarget } from "@/lib/graphJump";
import { formatTime } from "@/lib/utils";
import { summarizeWikiFreshness, type WikiPartialSummary } from "@/lib/wikiPartialSummary";
import { wikiErrorCodeHint, wikiFreshnessHint, wikiHiddenPeerHint, wikiPartialHint, type WikiOperatorHint } from "@/lib/wikiOperatorHints";

type WikiPeer = {
  topic: KnowledgeGraphWikiTopic;
  edge: KnowledgeGraphWikiCrossLink;
  visible: boolean;
};

export type WikiInspectorProps = {
  selectedNode?: KnowledgeGraphNode | KnowledgeGraphNodeDetail;
  selectedTopic?: KnowledgeGraphWikiTopic | null;
  selectedCrossLink?: KnowledgeGraphWikiCrossLink | null;
  wikiMap?: KnowledgeGraphWikiMap;
  wikiPartial: WikiPartialSummary | null;
  visibleTopicIds: Set<string>;
  onSelectTopic: (topicId: string) => void;
  onOpenSession: (sessionId: string) => void;
  focusedTopicId?: string;
  onFocusTopic?: (topicId: string) => void;
  onClearFocus?: () => void;
  onOpenEvidence?: (target: string) => void;
};

function evidenceRefs(value: unknown): KnowledgeGraphWikiEvidenceRef[] {
  return Array.isArray(value) ? (value.filter((item) => !!item && typeof item === "object") as KnowledgeGraphWikiEvidenceRef[]) : [];
}

function evidenceLabel(ref: KnowledgeGraphWikiEvidenceRef | Record<string, unknown>): string {
  const id = typeof ref.id === "string" && ref.id ? ref.id : "";
  const nodeKey = typeof ref.node_key === "string" && ref.node_key ? ref.node_key : "";
  return id || nodeKey || "evidence";
}

function stringValue(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function shortSessionId(sessionId: string): string {
  return sessionId.length > 16 ? `${sessionId.slice(0, 8)}...${sessionId.slice(-4)}` : sessionId;
}

function topicEvidenceRefs(topic?: KnowledgeGraphWikiTopic | null, node?: KnowledgeGraphNode | KnowledgeGraphNodeDetail): KnowledgeGraphWikiEvidenceRef[] {
  const topicRefs = evidenceRefs(topic?.attrs?.evidence_refs);
  if (topicRefs.length > 0) return topicRefs;
  return evidenceRefs(node?.attrs?.evidence_refs);
}

export function findWikiTopicForNode(
  selectedNode: KnowledgeGraphNode | KnowledgeGraphNodeDetail | undefined,
  wikiMap: KnowledgeGraphWikiMap | undefined,
): KnowledgeGraphWikiTopic | null {
  if (!selectedNode || selectedNode.type !== "wikiTopic") return null;
  return wikiMap?.topics.find((topic) => topic.id === selectedNode.id || topic.key === selectedNode.key) ?? null;
}

function peersForTopic(topic: KnowledgeGraphWikiTopic | null, wikiMap: KnowledgeGraphWikiMap | undefined, visibleTopicIds: Set<string>): WikiPeer[] {
  if (!topic || !wikiMap) return [];
  const topicsById = new Map(wikiMap.topics.map((item) => [item.id, item]));
  return wikiMap.cross_links
    .filter((edge) => edge.source === topic.id || edge.target === topic.id)
    .map((edge) => {
      const peerId = edge.source === topic.id ? edge.target : edge.source;
      const peerTopic = topicsById.get(peerId);
      return peerTopic ? { topic: peerTopic, edge, visible: visibleTopicIds.has(peerTopic.id) } : null;
    })
    .filter((item): item is WikiPeer => !!item);
}

function HintBlock({ hint }: { hint: WikiOperatorHint }) {
  return (
    <div className="rounded-md border border-warning/50 bg-warning/10 px-2 py-1.5 text-xs text-text">
      <span className="block font-medium text-warning">{hint.title}</span>
      <span className="mt-1 block text-text-muted">{hint.detail}</span>
    </div>
  );
}

export function WikiInspector({
  selectedNode,
  selectedTopic,
  selectedCrossLink,
  wikiMap,
  wikiPartial,
  visibleTopicIds,
  onSelectTopic,
  onOpenSession,
  focusedTopicId,
  onFocusTopic,
  onClearFocus,
  onOpenEvidence,
}: WikiInspectorProps) {
  const [peerHint, setPeerHint] = useState<WikiOperatorHint | null>(null);
  const topic = selectedTopic ?? findWikiTopicForNode(selectedNode, wikiMap);
  const nodeAttr = (key: string) => (selectedNode?.type === "wikiTopic" ? stringAttr(selectedNode as KnowledgeGraphNode, key) : "");
  const freshness = summarizeWikiFreshness(wikiMap);
  const freshnessHint = wikiFreshnessHint(freshness);
  const partialHint = wikiPartialHint(wikiPartial);
  const errorCodeHint = wikiErrorCodeHint(wikiPartial?.dominantErrorCode);
  const refs = topicEvidenceRefs(topic, selectedNode);
  const peers = useMemo(() => peersForTopic(topic, wikiMap, visibleTopicIds), [topic, wikiMap, visibleTopicIds]);
  const sessionId = stringValue(topic?.attrs?.session_id) || topic?.session_ids[0] || nodeAttr("session_id");
  const sessionTitle = stringValue(topic?.attrs?.session_title) || nodeAttr("session_title");
  const sessionLabel = sessionTitle || (sessionId ? shortSessionId(sessionId) : "");
  const quality = stringValue(topic?.attrs?.quality) || nodeAttr("quality");
  const confidence = stringValue(topic?.attrs?.confidence) || nodeAttr("confidence");
  const conceptKey = stringValue(topic?.attrs?.concept_key) || nodeAttr("concept_key");
  const crossLinkTopics = selectedCrossLink && wikiMap
    ? {
        source: wikiMap.topics.find((item) => item.id === selectedCrossLink.source),
        target: wikiMap.topics.find((item) => item.id === selectedCrossLink.target),
      }
    : null;

  if (!selectedNode || selectedNode.type !== "wikiTopic") {
    return (
      <div className="rounded-md border border-dashed border-border p-4 text-sm text-text-muted">
        Select a Wiki topic to inspect evidence, freshness, and cross-session links.
      </div>
    );
  }

  const title = topic?.title || selectedNode.title || "Untitled topic";
  const summary = topic?.summary || selectedNode.summary || "";

  return (
    <div aria-label="Wiki Inspector" className="space-y-3">
      <div>
        <div className="mb-2 flex flex-wrap items-center gap-2">
          <Badge>Topic</Badge>
          {quality ? <Badge variant="accent">{quality}</Badge> : null}
          {confidence ? <Badge>{confidence} confidence</Badge> : null}
        </div>
        <h3 className="break-words text-base font-semibold text-text">{title}</h3>
        {summary ? <p className="mt-2 text-sm text-text-muted">{summary}</p> : null}
        {conceptKey ? (
          <p className="mt-2 break-words rounded-md border border-border bg-bg px-2 py-1.5 text-xs text-text-muted">
            Concept key: <span className="font-medium text-text">{conceptKey}</span>
          </p>
        ) : null}
        {topic && (onFocusTopic || (focusedTopicId && onClearFocus)) ? (
          <div className="mt-2 flex flex-wrap gap-2">
            {focusedTopicId && onClearFocus ? (
              <Button variant="outline" size="sm" onClick={onClearFocus}>
                Return to full map
              </Button>
            ) : onFocusTopic ? (
              <Button variant="outline" size="sm" onClick={() => onFocusTopic(topic.id)}>
                Focus branch
              </Button>
            ) : null}
          </div>
        ) : null}
      </div>

      <div className="space-y-2 rounded-md border border-border bg-bg px-3 py-2 text-xs text-text">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h4 className="font-semibold uppercase text-text-faint">Freshness</h4>
          {freshness ? <Badge variant={freshness.stale || freshness.graphVersionMismatch || freshness.status === "partial" ? "warning" : "accent"}>{freshness.status}</Badge> : null}
        </div>
        {freshness?.lastRefreshedAt ? (
          <p className="text-text-muted">Last refreshed {formatTime(freshness.lastRefreshedAt)}</p>
        ) : (
          <p className="text-text-muted">No refresh timestamp is available.</p>
        )}
        {freshness?.snapshotGraphVersionId || freshness?.activeGraphVersionId ? (
          <p className="break-words text-text-muted">
            Snapshot graph {freshness.snapshotGraphVersionId || "unknown"} · Active graph {freshness.activeGraphVersionId || "unknown"}
          </p>
        ) : null}
        {freshnessHint ? <HintBlock hint={freshnessHint} /> : null}
        {partialHint ? <HintBlock hint={partialHint} /> : null}
        {errorCodeHint ? <HintBlock hint={errorCodeHint} /> : null}
        {wikiPartial?.repairText ? <p className="text-text-muted">{wikiPartial.repairText}</p> : null}
      </div>

      <div>
        <h4 className="mb-2 text-xs font-semibold uppercase text-text-faint">Topic evidence</h4>
        {refs.length > 0 ? (
          <ul className="space-y-1.5">
            {refs.slice(0, 8).map((ref, index) => (
              <li key={`${evidenceLabel(ref)}-${index}`} className="rounded-md border border-border bg-bg px-2 py-1.5 text-xs text-text-body">
                {(() => {
                  const target = resolveWikiEvidenceJumpTarget(ref, { sessionId });
                  const label = (
                    <>
                      <span className="font-medium">{ref.type || "evidence"}</span>
                      <span className="ml-1 text-text-muted">{evidenceLabel(ref)}</span>
                    </>
                  );
                  return target && onOpenEvidence ? (
                    <button type="button" className="text-left text-accent" onClick={() => onOpenEvidence(target)}>
                      {label}
                    </button>
                  ) : (
                    <>
                      {label}
                      {!target ? <span className="ml-1 text-text-faint">(unavailable)</span> : null}
                    </>
                  );
                })()}
                {ref.excerpt ? <span className="mt-1 block text-text-muted">{ref.excerpt}</span> : null}
              </li>
            ))}
          </ul>
        ) : (
          <p className="rounded-md border border-border bg-bg px-2 py-1.5 text-xs text-text-muted">
            Evidence unavailable for this snapshot.
          </p>
        )}
      </div>

      {selectedCrossLink && crossLinkTopics ? (
        <div aria-label="Cross-link detail" className="space-y-2 rounded-md border border-border bg-bg px-3 py-2 text-xs text-text">
          <h4 className="font-semibold uppercase text-text-faint">Cross-link detail</h4>
          <p className="font-medium text-text">
            {crossLinkTopics.source?.title || selectedCrossLink.source} → {crossLinkTopics.target?.title || selectedCrossLink.target}
          </p>
          {typeof selectedCrossLink.attrs?.reason === "string" && selectedCrossLink.attrs.reason ? (
            <p className="text-text-muted">{selectedCrossLink.attrs.reason}</p>
          ) : null}
          {selectedCrossLink.evidence_refs.length > 0 ? (
            <p className="text-text-faint">{selectedCrossLink.evidence_refs.slice(0, 8).map(evidenceLabel).join(" · ")}</p>
          ) : (
            <p className="text-text-muted">No bounded evidence reference is available for this link.</p>
          )}
        </div>
      ) : null}

      {peers.length > 0 ? (
        <div>
          <h4 className="mb-2 text-xs font-semibold uppercase text-text-faint">Cross-session topics</h4>
          <ul className="space-y-1.5">
            {peers.slice(0, 8).map((peer) => (
              <li key={`${peer.edge.id}-${peer.topic.id}`} className="rounded-md border border-border bg-bg px-2 py-1.5 text-xs text-text-body">
                <button
                  type="button"
                  className="text-left font-medium text-accent disabled:text-text-muted"
                  onClick={() => {
                    if (peer.visible) {
                      setPeerHint(null);
                      onSelectTopic(peer.topic.id);
                    } else {
                      setPeerHint(wikiHiddenPeerHint(peer.topic.title));
                    }
                  }}
                >
                  {peer.topic.title || "Untitled topic"}
                </button>
                {!peer.visible ? <span className="ml-1 text-text-faint">(hidden)</span> : null}
                {typeof peer.edge.attrs?.reason === "string" && peer.edge.attrs.reason ? (
                  <span className="mt-1 block text-text-muted">{peer.edge.attrs.reason}</span>
                ) : null}
                {peer.edge.evidence_refs.length > 0 ? (
                  <span className="mt-1 block text-text-faint">
                    {peer.edge.evidence_refs.slice(0, 3).map(evidenceLabel).join(" · ")}
                  </span>
                ) : null}
              </li>
            ))}
          </ul>
          {peerHint ? <div className="mt-2"><HintBlock hint={peerHint} /></div> : null}
        </div>
      ) : null}

      {sessionId ? (
        <div className="space-y-2 rounded-md border border-border bg-bg px-3 py-2 text-xs text-text">
          <h4 className="font-semibold uppercase text-text-faint">Source session</h4>
          <p className="break-words text-sm font-medium text-text">{sessionLabel}</p>
          <Button variant="outline" size="sm" onClick={() => onOpenSession(sessionId)}>
            Open source session
          </Button>
        </div>
      ) : null}
    </div>
  );
}
