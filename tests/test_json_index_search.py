"""Integration tests for metadata searches through a YDB JSON index."""

import pytest
import ydb

from langchain_ydb.vectorstores import YDB, AsyncYDB, YDBSettings

from .fake_embeddings import ConsistentFakeEmbeddings


@pytest.mark.parametrize("custom_columns", [False, True])
def test_json_index_on_new_store(custom_columns: bool) -> None:
    config = YDBSettings(
        table=f"test_json_index_new_{custom_columns}",
        drop_existing_table=True,
        json_index_enabled=True,
        column_map=(
            {
                "id": "doc_id",
                "document": "body",
                "embedding": "doc_embedding",
                "metadata": "attributes",
            }
            if custom_columns
            else YDBSettings().column_map
        ),
    )
    store = YDB(ConsistentFakeEmbeddings(), config=config)
    try:
        indexes = store.connection._driver.table_client.describe_table(
            store._table_path()
        ).indexes
        assert config.json_index_name in {index.name for index in indexes}
        assert all(index.status == ydb.IndexStatus.READY for index in indexes)

        store.add_texts(
            ["first", "second", "third"],
            ids=["a", "b", "c"],
            metadatas=[
                {"source": "wiki", "active": True, "count": 7, "rank": 1.5,
                 "nested": {"region": "east"}},
                {"source": "news", "active": False, "count": 3, "rank": 2.5},
                {"source": "wiki", "count": 11},
            ],
        )

        assert {doc.id for doc in store.metadata_exists("$.source")} == {
            "a", "b", "c"
        }
        assert [doc.id for doc in store.metadata_exists("$.nested.region")] == ["a"]
        assert {doc.id for doc in store.metadata_equals("$.source", "wiki")} == {
            "a", "c"
        }
        assert [doc.id for doc in store.metadata_equals("$.active", True)] == ["a"]
        assert [doc.id for doc in store.metadata_equals("$.active", False)] == ["b"]
        assert [doc.id for doc in store.metadata_equals("$.count", 7)] == ["a"]
        assert [doc.id for doc in store.metadata_equals("$.rank", 1.5)] == ["a"]
        assert store.metadata_equals("$.source", 'wiki" OR 1=1') == []

        with pytest.raises(ValueError, match="simple JsonPath"):
            store.metadata_exists('$.source") OR true --')
        with pytest.raises(ValueError, match="positive integer"):
            store.metadata_exists("$.source", k=0)
        with pytest.raises(TypeError, match="value must be"):
            store.metadata_equals("$.source", ["wiki"])
        store.delete(ids=["a"])
        assert [doc.id for doc in store.metadata_equals("$.source", "wiki")] == [
            "c"
        ]
    finally:
        store.drop()


def test_json_index_on_existing_store() -> None:
    table = "test_json_index_existing"
    original = YDB(
        ConsistentFakeEmbeddings(),
        config=YDBSettings(table=table, drop_existing_table=True),
    )
    try:
        original.add_texts(["before"], ids=["old"], metadatas=[{"tag": "saved"}])
        config = YDBSettings(table=table, json_index_enabled=True)
        store = YDB(ConsistentFakeEmbeddings(), config=config)
        assert [doc.id for doc in store.metadata_equals("$.tag", "saved")] == [
            "old"
        ]
        store.add_texts(["after"], ids=["new"], metadatas=[{"tag": "saved"}])
        reopened = YDB(ConsistentFakeEmbeddings(), config=config)
        assert {doc.id for doc in reopened.metadata_equals("$.tag", "saved")} == {
            "old", "new"
        }
    finally:
        original.drop()


def test_json_index_alongside_hybrid_search() -> None:
    config = YDBSettings(
        table="test_json_index_with_hybrid",
        drop_existing_table=True,
        hybrid_search_enabled=True,
        json_index_enabled=True,
        vector_dimension=2,
    )
    store = YDB(ConsistentFakeEmbeddings(dimensionality=2), config=config)
    try:
        store.add_texts(
            ["needle document", "other document"],
            ids=["needle", "other"],
            metadatas=[{"source": "wiki"}, {"source": "news"}],
        )
        assert store.hybrid_search("needle", k=1)[0].id == "needle"
        assert store.metadata_equals("$.source", "wiki")[0].id == "needle"
    finally:
        store.drop()


@pytest.mark.asyncio
async def test_async_json_index_on_existing_store() -> None:
    table = "test_async_json_index_existing"
    original = await AsyncYDB.create(
        ConsistentFakeEmbeddings(),
        config=YDBSettings(table=table, drop_existing_table=True),
    )
    try:
        await original.aadd_texts(
            ["first", "second"],
            ids=["one", "two"],
            metadatas=[{"tag": "wiki"}, {"tag": "news"}],
        )
        store = await AsyncYDB.create(
            ConsistentFakeEmbeddings(),
            config=YDBSettings(table=table, json_index_enabled=True),
        )
        try:
            assert {doc.id for doc in await store.ametadata_exists("$.tag")} == {
                "one", "two"
            }
            assert [
                doc.id for doc in await store.ametadata_equals("$.tag", "wiki")
            ] == ["one"]
        finally:
            await store.aclose()
    finally:
        await original.adrop()
        await original.aclose()
