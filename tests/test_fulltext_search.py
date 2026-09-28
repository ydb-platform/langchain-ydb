"""Integration tests for standalone YDB fulltext retrieval."""

import pytest
import ydb

from langchain_ydb.vectorstores import YDB, AsyncYDB, YDBSettings

from .fake_embeddings import ConsistentFakeEmbeddings


@pytest.mark.parametrize("custom_columns", [False, True])
def test_fulltext_search_on_new_store(custom_columns: bool) -> None:
    config = YDBSettings(
        table=f"test_fulltext_new_{custom_columns}",
        drop_existing_table=True,
        fulltext_index_enabled=True,
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
        assert config.fulltext_index_name in {index.name for index in indexes}
        assert config.index_name not in {index.name for index in indexes}
        assert all(index.status == ydb.IndexStatus.READY for index in indexes)

        store.add_texts(
            ["red apple", "red banana", "blue ocean"],
            ids=["apple", "banana", "ocean"],
            metadatas=[{"kind": "fruit"}, {"kind": "fruit"}, {"kind": "water"}],
        )
        assert {doc.id for doc in store.fulltext_match("red")} == {
            "apple",
            "banana",
        }
        ranked = store.fulltext_search_with_score("apple")
        assert ranked[0][0].id == "apple"
        assert ranked[0][0].metadata == {"kind": "fruit"}
        assert ranked[0][1] > 0
        assert store.fulltext_search("apple")[0].id == "apple"
        assert store.as_fulltext_retriever(k=1).invoke("apple")[0].id == "apple"
    finally:
        store.drop()


def test_fulltext_search_on_existing_store() -> None:
    table = "test_fulltext_existing"
    original = YDB(
        ConsistentFakeEmbeddings(),
        config=YDBSettings(table=table, drop_existing_table=True),
    )
    try:
        original.add_texts(["early keyword"], ids=["early"])
        config = YDBSettings(table=table, fulltext_index_enabled=True)
        store = YDB(ConsistentFakeEmbeddings(), config=config)
        assert store.fulltext_search("early")[0].id == "early"
        store.add_texts(["later keyword"], ids=["later"])
        reopened = YDB(ConsistentFakeEmbeddings(), config=config)
        assert {doc.id for doc in reopened.fulltext_match("keyword")} == {
            "early",
            "later",
        }
    finally:
        original.drop()


def test_reject_non_relevance_fulltext_index() -> None:
    table = "test_fulltext_plain_reuse"
    original = YDB(
        ConsistentFakeEmbeddings(),
        config=YDBSettings(table=table, drop_existing_table=True),
    )
    try:
        original.add_texts(["index keyword"], ids=["existing"])
        original._execute_query(
            f"ALTER TABLE `{table}` ADD INDEX ydb_fulltext_index "
            "GLOBAL USING fulltext_plain ON (document) "
            "WITH (tokenizer=standard);",
            ddl=True,
        )
        with pytest.raises(ValueError, match="fulltext_relevance"):
            YDB(
                ConsistentFakeEmbeddings(),
                config=YDBSettings(table=table, fulltext_index_enabled=True),
            )
    finally:
        original.drop()


@pytest.mark.asyncio
async def test_async_fulltext_search_on_existing_store() -> None:
    table = "test_async_fulltext_existing"
    original = await AsyncYDB.create(
        ConsistentFakeEmbeddings(),
        config=YDBSettings(table=table, drop_existing_table=True),
    )
    try:
        await original.aadd_texts(
            ["red apple", "blue ocean"], ids=["apple", "ocean"]
        )
        store = await AsyncYDB.create(
            ConsistentFakeEmbeddings(),
            config=YDBSettings(table=table, fulltext_index_enabled=True),
        )
        try:
            assert (await store.afulltext_search("apple"))[0].id == "apple"
            assert (await store.afulltext_match("red"))[0].id == "apple"
            assert (await store.afulltext_search_with_score("apple"))[0][1] > 0
            retrieved = await store.as_fulltext_retriever(k=1).ainvoke("apple")
            assert retrieved[0].id == "apple"
        finally:
            await store.aclose()
    finally:
        await original.adrop()
        await original.aclose()
