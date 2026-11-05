"""DR-ITER-062 memory duplicate optimizer."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.baseline.indexing import Indexer
from deepresearch_core.baseline.library_index import LibraryIndexer
from deepresearch_core.baseline.memory import MemoryEngine, MemoryLayer
from deepresearch_core.baseline.memory_lifecycle import MemoryLifecycleService
from deepresearch_core.baseline.memory_optimizer import MemoryOptimizer
from deepresearch_core.baseline.service import BaselineService
from deepresearch_core.baseline.store import Store
from deepresearch_core.settings import ProviderSettings
from tests.baseline.fakes import FakeLLM


def test_optimizer_preview_finds_lexical_duplicates_without_embedder(tmp_path):
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", snapshot_root=tmp_path)
    service.add_memory({"layer": "workspace", "scope_key": "default", "content": "Prefer concise local-first answers"})
    service.add_memory({"layer": "workspace", "scope_key": "default", "content": "Prefer concise local-first responses"})

    preview = service.preview_memory_optimization({"threshold": 0.7})

    assert len(preview["groups"]) == 1
    assert preview["groups"][0]["suggestions"][0]["type"] == "merge"


def test_optimizer_preview_omits_pinned_memories(tmp_path):
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", snapshot_root=tmp_path)
    first = service.add_memory({"layer": "workspace", "scope_key": "default", "content": "Keep pinned local-first answers"})["memory_id"]
    service.add_memory({"layer": "workspace", "scope_key": "default", "content": "Keep pinned local-first responses"})
    service.patch_memory(first, {"pinned": True})

    preview = service.preview_memory_optimization({"threshold": 0.7})

    assert preview["groups"] == []


def test_optimizer_apply_rejects_pinned_memory(tmp_path):
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", snapshot_root=tmp_path)
    memory_id = service.add_memory({"layer": "workspace", "scope_key": "default", "content": "Pinned optimizer guard"})["memory_id"]
    service.patch_memory(memory_id, {"pinned": True})

    with pytest.raises(BaselineError) as exc:
        service.apply_memory_optimization({"strategy": "archive", "memory_ids": [memory_id]})

    assert exc.value.code is ErrorCode.INVALID_REQUEST
    assert exc.value.details["pinned_ids"] == [memory_id]
    assert service.store.get_memory(memory_id)["archived_at"] is None


def test_optimizer_merge_requires_memory_role(tmp_path):
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", snapshot_root=tmp_path)
    first = service.add_memory({"layer": "workspace", "scope_key": "default", "content": "Use local tools"})["memory_id"]
    second = service.add_memory({"layer": "workspace", "scope_key": "default", "content": "Use local tooling"})["memory_id"]

    with pytest.raises(BaselineError) as exc:
        service.apply_memory_optimization({"strategy": "merge", "memory_ids": [first, second]})

    assert exc.value.code is ErrorCode.PROVIDER_NOT_CONFIGURED


def test_optimizer_apply_merge_archives_sources_and_purges_legacy_library_artifacts():
    store = Store(":memory:")
    llm = FakeLLM(scripted={"Merge these memory entries": "User prefers concise local-first research answers."})
    memory = MemoryEngine(store, embedder=llm)
    library = LibraryIndexer(store, indexer=Indexer(store, embedder=llm))

    def index_memory(**kwargs):
        library.index_memory(**kwargs)

    lifecycle = MemoryLifecycleService(store, index_memory=index_memory)
    optimizer = MemoryOptimizer(
        store,
        memory=memory,
        lifecycle=lifecycle,
        index_memory=index_memory,
        memory_llm=lambda: llm,
    )
    first = memory.remember(MemoryLayer.WORKSPACE, "default", "Prefer concise local-first answers", salience=2.0)
    second = memory.remember(MemoryLayer.WORKSPACE, "default", "Prefer concise local-first responses", salience=1.0)
    for memory_id in (first, second):
        row = store.get_memory(memory_id)
        index_memory(
            memory_id=memory_id,
            namespace=row["namespace"],
            scope_key=row["scope_key"],
            content=row["content"],
            metadata={},
        )

    result = optimizer.apply({"strategy": "merge", "memory_ids": [first, second]})

    assert result["summary"] == "User prefers concise local-first research answers."
    assert set(result["archived_ids"]) == {first, second}
    assert store.get_memory(first)["archived_at"] is not None
    assert store.list_library_provenance(artifact_type="memory", artifact_id=first) == []
    assert store.list_library_provenance(artifact_type="memory", artifact_id=result["memory_id"]) == []
    assert result["library_index_warning"] is None


def test_optimizer_merge_rolls_back_when_source_purge_fails():
    store = Store(":memory:")
    llm = FakeLLM(scripted={"Merge these memory entries": "Merged durable local preference."})
    memory = MemoryEngine(store, embedder=llm)
    library = LibraryIndexer(store, indexer=Indexer(store, embedder=llm))

    def index_memory(**kwargs):
        library.index_memory(**kwargs)

    lifecycle = MemoryLifecycleService(store, index_memory=index_memory)
    optimizer = MemoryOptimizer(
        store,
        memory=memory,
        lifecycle=lifecycle,
        index_memory=index_memory,
        memory_llm=lambda: llm,
    )
    first = memory.remember(MemoryLayer.WORKSPACE, "default", "Rollback merge first", salience=2.0)
    second = memory.remember(MemoryLayer.WORKSPACE, "default", "Rollback merge second", salience=1.0)
    for memory_id in (first, second):
        row = store.get_memory(memory_id)
        index_memory(
            memory_id=memory_id,
            namespace=row["namespace"],
            scope_key=row["scope_key"],
            content=row["content"],
            metadata={},
        )

    with patch.object(store, "_delete_library_artifact_locked", side_effect=RuntimeError("purge failed")):
        with pytest.raises(BaselineError) as exc:
            optimizer.apply({"strategy": "merge", "memory_ids": [first, second]})

    assert exc.value.code is ErrorCode.INVALID_REQUEST
    assert exc.value.details["memory_ids"] == [first, second]
    active_ids = {row["id"] for row in store.list_memories(namespace="workspace", scope_key="default", limit=10)}
    assert active_ids == {first, second}
    assert all(store.get_memory(memory_id)["archived_at"] is None for memory_id in (first, second))
    assert not [
        row
        for row in store.list_all_memories(include_archived=True)
        if row.get("metadata", {}).get("source") == "memory_optimizer"
    ]
    assert store.list_library_provenance(artifact_type="memory", artifact_id=first)


def test_optimizer_merge_keeps_warning_null_when_legacy_merged_index_would_fail():
    store = Store(":memory:")
    llm = FakeLLM(scripted={"Merge these memory entries": "Merged memory with degraded index."})
    memory = MemoryEngine(store, embedder=llm)
    lifecycle = MemoryLifecycleService(store, index_memory=lambda **_kwargs: None)

    optimizer = MemoryOptimizer(
        store,
        memory=memory,
        lifecycle=lifecycle,
        index_memory=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("legacy index should not run")),
        memory_llm=lambda: llm,
    )
    first = memory.remember(MemoryLayer.WORKSPACE, "default", "Merge warning first", salience=2.0)
    second = memory.remember(MemoryLayer.WORKSPACE, "default", "Merge warning second", salience=1.0)

    result = optimizer.apply({"strategy": "merge", "memory_ids": [first, second]})

    assert result["library_index_warning"] is None
    assert store.get_memory(first)["archived_at"] is not None
    assert store.get_memory(second)["archived_at"] is not None


def test_optimizer_apply_archive_archives_selected_memories_and_purges_library_artifacts():
    store = Store(":memory:")
    llm = FakeLLM()
    memory = MemoryEngine(store, embedder=llm)
    library = LibraryIndexer(store, indexer=Indexer(store, embedder=llm))

    def index_memory(**kwargs):
        library.index_memory(**kwargs)

    lifecycle = MemoryLifecycleService(store, index_memory=index_memory)
    optimizer = MemoryOptimizer(
        store,
        memory=memory,
        lifecycle=lifecycle,
        index_memory=index_memory,
        memory_llm=lambda: llm,
    )
    keep = memory.remember(MemoryLayer.WORKSPACE, "default", "Keep durable memory", salience=2.0)
    duplicate = memory.remember(MemoryLayer.WORKSPACE, "default", "Archive duplicate memory", salience=1.0)
    row = store.get_memory(duplicate)
    index_memory(
        memory_id=duplicate,
        namespace=row["namespace"],
        scope_key=row["scope_key"],
        content=row["content"],
        metadata={},
    )
    assert store.search_chunks_lexical("Archive duplicate", limit=5)

    result = optimizer.apply({"strategy": "archive", "memory_ids": [duplicate]})

    assert result == {"strategy": "archive", "archived_ids": [duplicate]}
    assert store.get_memory(duplicate)["archived_at"] is not None
    assert {row["id"] for row in store.list_memories(namespace="workspace", scope_key="default")} == {keep}
    assert store.list_library_provenance(artifact_type="memory", artifact_id=duplicate) == []
    assert store.search_chunks_lexical("Archive duplicate", limit=5) == []


def test_optimizer_apply_downrank_reduces_salience_and_keeps_memory_active():
    store = Store(":memory:")
    llm = FakeLLM()
    memory = MemoryEngine(store, embedder=llm)
    lifecycle = MemoryLifecycleService(store, index_memory=lambda **_kwargs: None)
    optimizer = MemoryOptimizer(
        store,
        memory=memory,
        lifecycle=lifecycle,
        index_memory=lambda **_kwargs: None,
        memory_llm=lambda: llm,
    )
    memory_id = memory.remember(MemoryLayer.USER, "default", "Lower priority duplicate", salience=2.0)

    result = optimizer.apply({"strategy": "downrank", "memory_ids": [memory_id], "factor": 0.25})

    assert result == {"strategy": "downrank", "memory_ids": [memory_id], "factor": 0.25}
    assert store.get_memory(memory_id)["archived_at"] is None
    assert store.get_memory(memory_id)["salience"] == pytest.approx(0.5)
    assert [row["id"] for row in store.list_memories(namespace="user", scope_key="default")] == [memory_id]
