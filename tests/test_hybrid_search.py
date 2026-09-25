"""Integration tests for native fulltext/vector search."""

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
        with pytest.raises(ValueError, match="Metadata filters"):
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
