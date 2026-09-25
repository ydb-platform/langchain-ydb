"""Integration tests for native fulltext/vector search."""

from types import SimpleNamespace
from typing import Any, List

import pytest
import ydb
from langchain_core.embeddings import Embeddings

from langchain_ydb.vectorstores import YDB, AsyncYDB, YDBSettings


class HybridEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[1.0, 0.0] if "semantic" in text else [0.0, 1.0] for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return [1.0, 0.0]


def _settings(table: str, **kwargs: Any) -> YDBSettings:
    return YDBSettings(table=table, vector_dimension=2, **kwargs)


def _index_names(store: YDB) -> set[str]:
    description = store.connection._driver.table_client.describe_table(
        store._table_path()
    )
    assert all(index.status == ydb.IndexStatus.READY for index in description.indexes)
    return {index.name for index in description.indexes}


@pytest.mark.parametrize("vector_pass_as_bytes", [True, False])
def test_hybrid_search_on_new_store(vector_pass_as_bytes: bool) -> None:
    config = _settings(
        f"test_hybrid_new_{vector_pass_as_bytes}",
        drop_existing_table=True,
        hybrid_search_enabled=True,
        vector_pass_as_bytes=vector_pass_as_bytes,
        column_map=(
            {
                "id": "custom_id",
                "document": "custom_document",
                "embedding": "custom_embedding",
                "metadata": "custom_metadata",
            }
            if not vector_pass_as_bytes
            else YDBSettings().column_map
        ),
    )
    store = YDB(HybridEmbeddings(), config=config)
    try:
        assert {config.index_name, config.fulltext_index_name} <= _index_names(store)
        store.add_texts(
            ["semantic result", "needle exact term"],
            ids=["semantic", "lexical"],
            metadatas=[{"source": "vector"}, {"source": "text"}],
        )

        results = store.hybrid_search(
            "needle", k=2, weights=(2.0, 1.0), candidate_limits=(1, 1)
        )
        assert [doc.id for doc in results] == ["lexical", "semantic"]
        assert results[0].metadata == {"source": "text"}
        assert len(store.hybrid_search("needle", k=2, mode="linear")) == 2
        assert [doc.id for doc in store.as_hybrid_retriever(k=2).invoke("needle")] == [
            "lexical",
            "semantic",
        ]
        with pytest.raises(ValueError, match="hybrid metadata filters"):
            store.hybrid_search("needle", filter={"source": "text"})
    finally:
        store.drop()


@pytest.mark.parametrize("existing_vector_index", [False, True])
def test_hybrid_search_on_existing_store(existing_vector_index: bool) -> None:
    table = f"test_hybrid_existing_{existing_vector_index}"
    initial = YDB(
        HybridEmbeddings(),
        config=_settings(
            table, drop_existing_table=True, index_enabled=existing_vector_index
        ),
    )
    initial.add_texts(
        ["semantic result", "needle exact term"],
        ids=["semantic", "lexical"],
    )
    config = _settings(table, hybrid_search_enabled=True)
    try:
        store = YDB(HybridEmbeddings(), config=config)
        assert {config.index_name, config.fulltext_index_name} <= _index_names(store)
        assert {doc.id for doc in store.hybrid_search("needle", k=2)} == {
            "semantic",
            "lexical",
        }
        store.add_texts(["new needle"], ids=["new"])
        reopened = YDB(HybridEmbeddings(), config=config)
        assert "new" in {doc.id for doc in reopened.hybrid_search("needle", k=3)}
    finally:
        initial.drop()


@pytest.mark.parametrize("incompatible_index", ["fulltext", "vector"])
def test_reject_incompatible_existing_index(incompatible_index: str) -> None:
    table = f"test_hybrid_incompatible_{incompatible_index}"
    initial = YDB(
        HybridEmbeddings(), config=_settings(table, drop_existing_table=True)
    )
    config = _settings(table, hybrid_search_enabled=True)
    try:
        if incompatible_index == "fulltext":
            initial._execute_query(
                initial._format_add_named_vector_index_query(2, config.index_name),
                ddl=True,
            )
            initial._execute_query(
                f"ALTER TABLE `{table}` ADD INDEX `{config.fulltext_index_name}` "
                "GLOBAL USING fulltext_plain ON (`document`) "
                "WITH (tokenizer=standard);",
                ddl=True,
            )
        else:
            initial._execute_query(initial._format_add_fulltext_index_query(), ddl=True)
            initial._execute_query(
                f"ALTER TABLE `{table}` ADD INDEX `{config.index_name}` "
                "GLOBAL USING vector_kmeans_tree ON (embedding) "
                "WITH (distance=euclidean, vector_type=\"Float\", "
                "vector_dimension=2, levels=2, clusters=128);",
                ddl=True,
            )

        with pytest.raises(ValueError, match="HybridRank validation failed"):
            YDB(HybridEmbeddings(), config=config)
    finally:
        initial.drop()


def test_hybrid_index_wait_has_deadline() -> None:
    config = _settings(
        "test_hybrid_never_ready",
        hybrid_search_enabled=True,
        hybrid_index_ready_timeout=0.01,
    )
    indexes = [
        SimpleNamespace(
            name=config.index_name,
            index_columns=[config.column_map["embedding"]],
            status=ydb.IndexStatus.BUILDING,
        ),
        SimpleNamespace(
            name=config.fulltext_index_name,
            index_columns=[config.column_map["document"]],
            status=ydb.IndexStatus.READY,
        ),
    ]
    table_client = SimpleNamespace(
        describe_table=lambda _: SimpleNamespace(indexes=indexes)
    )
    store = object.__new__(YDB)
    store.config = config
    store.connection = SimpleNamespace(  # type: ignore[assignment]
        _driver=SimpleNamespace(table_client=table_client)
    )

    with pytest.raises(TimeoutError, match="ydb_vector_index: BUILDING"):
        store._ensure_hybrid_indexes()


@pytest.mark.asyncio
async def test_async_hybrid_search_on_new_store() -> None:
    config = _settings(
        "test_async_hybrid_new", drop_existing_table=True, hybrid_search_enabled=True
    )
    store = await AsyncYDB.create(HybridEmbeddings(), config=config)
    try:
        await store.aadd_texts(
            ["semantic result", "needle exact term"],
            ids=["semantic", "lexical"],
        )
        results = await store.ahybrid_search("needle", k=2, candidate_limits=(1, 1))
        assert {doc.id for doc in results} == {"semantic", "lexical"}
        retrieved = await store.as_hybrid_retriever(k=2).ainvoke("needle")
        assert {doc.id for doc in retrieved} == {"semantic", "lexical"}
    finally:
        await store.adrop()
        await store.aclose()


@pytest.mark.asyncio
async def test_async_hybrid_search_on_existing_store() -> None:
    table = "test_async_hybrid_existing"
    initial = await AsyncYDB.create(
        HybridEmbeddings(), config=_settings(table, drop_existing_table=True)
    )
    try:
        await initial.aadd_texts(
            ["semantic result", "needle exact term"],
            ids=["semantic", "lexical"],
        )
        store = await AsyncYDB.create(
            HybridEmbeddings(), config=_settings(table, hybrid_search_enabled=True)
        )
        try:
            assert {doc.id for doc in await store.ahybrid_search("needle", k=2)} == {
                "semantic",
                "lexical",
            }
        finally:
            await store.aclose()
    finally:
        await initial.adrop()
        await initial.aclose()


@pytest.mark.asyncio
async def test_async_reject_incompatible_existing_fulltext_index() -> None:
    table = "test_async_hybrid_incompatible_fulltext"
    initial = YDB(
        HybridEmbeddings(), config=_settings(table, drop_existing_table=True)
    )
    config = _settings(table, hybrid_search_enabled=True)
    try:
        initial._execute_query(
            initial._format_add_named_vector_index_query(2, config.index_name),
            ddl=True,
        )
        initial._execute_query(
            f"ALTER TABLE `{table}` ADD INDEX `{config.fulltext_index_name}` "
            "GLOBAL USING fulltext_plain ON (`document`) "
            "WITH (tokenizer=standard);",
            ddl=True,
        )
        with pytest.raises(ValueError, match="HybridRank validation failed"):
            await AsyncYDB.create(HybridEmbeddings(), config=config)
    finally:
        initial.drop()


@pytest.mark.asyncio
async def test_async_hybrid_index_wait_has_deadline() -> None:
    config = _settings(
        "test_async_hybrid_never_ready",
        hybrid_search_enabled=True,
        hybrid_index_ready_timeout=0.01,
    )
    indexes = [
        SimpleNamespace(
            name=config.index_name,
            index_columns=[config.column_map["embedding"]],
            status=ydb.IndexStatus.BUILDING,
        ),
        SimpleNamespace(
            name=config.fulltext_index_name,
            index_columns=[config.column_map["document"]],
            status=ydb.IndexStatus.READY,
        ),
    ]

    async def describe_table(_: str) -> SimpleNamespace:
        return SimpleNamespace(indexes=indexes)

    table_client = SimpleNamespace(describe_table=describe_table)
    store = object.__new__(AsyncYDB)
    store.config = config
    store.connection = SimpleNamespace(  # type: ignore[assignment]
        _driver=SimpleNamespace(table_client=table_client)
    )

    with pytest.raises(TimeoutError, match="ydb_vector_index: BUILDING"):
        await store._ensure_hybrid_indexes()
