"""T25 Wiki Map store contracts."""

from __future__ import annotations

from deepresearch_core.baseline.knowledge_graph.wiki_map import WikiMapBuilder
from deepresearch_core.baseline.knowledge_graph.wiki_models import (
    WikiSnapshotDraft,
    WikiTopicDraft,
    WikiTreeEdgeDraft,
)
from deepresearch_core.baseline.knowledge_graph.wiki_store import KnowledgeGraphWikiStore
from deepresearch_core.baseline.store import Store


def test_wiki_store_staged_active_partial_and_stale_roundtrip() -> None:
    store = Store(":memory:")
    wiki = KnowledgeGraphWikiStore(store)
    snapshot_id = wiki.create_staging_snapshot(workspace_id="default")
    stats = wiki.replace_snapshot_contents(
        snapshot_id,
        WikiSnapshotDraft(
            workspace_id="default",
            graph_version_id="",
            topics=(
                WikiTopicDraft(topic_key="wiki:root", title="Workspace Wiki", depth=0),
                WikiTopicDraft(topic_key="wiki:session:s1", title="Session 1", depth=1, parent_key="wiki:root", session_ids=("s1",)),
            ),
            tree_edges=(WikiTreeEdgeDraft(src_key="wiki:root", dst_key="wiki:session:s1"),),
            stats={"succeeded_sessions": 1},
        ),
    )
    wiki.activate_snapshot(snapshot_id, status="partial", stats=stats)

    active = wiki.get_active_snapshot("default")
    assert active
    assert active["id"] == snapshot_id
    assert active["status"] == "partial"
    assert active["stats"]["topics"] == 2

    wiki.mark_active_stale("default", reason="graph_version_changed")
    stale = wiki.get_active_snapshot("default")
    assert stale
    assert stale["stale"] is True
    assert stale["stale_reason"] == "graph_version_changed"

    view = WikiMapBuilder(store).build(workspace_id="default")
    assert view["active_snapshot"]["id"] == snapshot_id
    assert view["stale"] is True
    assert len(view["topics"]) == 2
    store.close()


def test_failed_wiki_snapshot_does_not_replace_previous_active() -> None:
    store = Store(":memory:")
    wiki = KnowledgeGraphWikiStore(store)
    active_id = wiki.create_staging_snapshot(workspace_id="default")
    stats = wiki.replace_snapshot_contents(
        active_id,
        WikiSnapshotDraft(
            workspace_id="default",
            graph_version_id="",
            topics=(WikiTopicDraft(topic_key="wiki:root", title="Workspace Wiki", depth=0),),
        ),
    )
    wiki.activate_snapshot(active_id, stats=stats)

    failed_id = wiki.create_staging_snapshot(workspace_id="default")
    wiki.mark_snapshot_failed(failed_id, error_code="NO_WIKI_TOPICS", error_message="empty")

    active = wiki.get_active_snapshot("default")
    assert active
    assert active["id"] == active_id
    assert active["status"] == "succeeded"
    store.close()
