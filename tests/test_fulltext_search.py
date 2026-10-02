"""Integration tests for standalone YDB fulltext retrieval."""

import pytest
import ydb
import ydb_dbapi

from langchain_ydb.vectorstores import YDB, AsyncYDB, YDBSettings

from .fake_embeddings import ConsistentFakeEmbeddings


@pytest.mark.parametrize(
    "enabled_modes",
    [
        {"index_enabled": True, "fulltext_index_enabled": True},
        {"hybrid_search_enabled": True},
    ],
)
def test_reject_colliding_index_names_before_connect(
    monkeypatch: pytest.MonkeyPatch, enabled_modes: dict
) -> None:
    def unexpected_connect(**kwargs: object) -> None:
        raise AssertionError("YDB connection must not be opened")

    monkeypatch.setattr(ydb_dbapi, "connect", unexpected_connect)
    config = YDBSettings(
        index_name="shared_index", fulltext_index_name="shared_index", **enabled_modes
    )
    with pytest.raises(ValueError, match="index names must be different"):
        YDB(ConsistentFakeEmbeddings(), config=config)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "enabled_modes",
    [
        {"index_enabled": True, "fulltext_index_enabled": True},
        {"hybrid_search_enabled": True},
    ],
)
async def test_async_reject_colliding_index_names_before_connect(
    monkeypatch: pytest.MonkeyPatch, enabled_modes: dict
) -> None:
    async def unexpected_connect(**kwargs: object) -> None:
        raise AssertionError("YDB connection must not be opened")

    monkeypatch.setattr(ydb_dbapi, "async_connect", unexpected_connect)
    config = YDBSettings(
        index_name="shared_index", fulltext_index_name="shared_index", **enabled_modes
    )
    with pytest.raises(ValueError, match="index names must be different"):
        await AsyncYDB.create(ConsistentFakeEmbeddings(), config=config)


def test_vector_rebuild_rejects_later_index_name_collision() -> None:
    config = YDBSettings(index_enabled=True, fulltext_index_enabled=True)
    config.fulltext_index_name = config.index_name
    store = object.__new__(YDB)
    store.config = config
    with pytest.raises(ValueError, match="index names must be different"):
        store.update_vector_index_if_needed()


@pytest.mark.asyncio
async def test_async_vector_rebuild_rejects_later_index_name_collision() -> None:
    config = YDBSettings(index_enabled=True, fulltext_index_enabled=True)
    config.fulltext_index_name = config.index_name
    store = object.__new__(AsyncYDB)
    store.config = config
    with pytest.raises(ValueError, match="index names must be different"):
        await store.update_vector_index_if_needed()


@pytest.mark.parametrize("mode", ["GLOBAL", "GLOBAL SYNC", "GLOBAL ASYNC"])
def test_reused_fulltext_index_type_accepts_global_modes(mode: str) -> None:
    store = object.__new__(YDB)
    store.config = YDBSettings(fulltext_index_name="custom_fulltext")
    store._validate_fulltext_index_type(
        "CREATE TABLE docs ("
        f"INDEX `custom_fulltext` {mode} USING fulltext_relevance "
        "ON (`document`));"
    )


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_reject_invalid_fulltext_index_ready_timeout(timeout: float) -> None:
    store = object.__new__(YDB)
    store.config = YDBSettings(fulltext_index_ready_timeout=timeout)
    with pytest.raises(ValueError, match="fulltext_index_ready_timeout"):
        store._validate_fulltext_settings()


@pytest.mark.parametrize(
    "method_name",
    ["fulltext_match", "fulltext_search", "fulltext_search_with_score"],
)
def test_fulltext_methods_require_index_flag(method_name: str) -> None:
    store = object.__new__(YDB)
    store.config = YDBSettings()
    with pytest.raises(ValueError, match="fulltext_index_enabled"):
        getattr(store, method_name)("query")


def test_fulltext_retriever_requires_index_flag() -> None:
    store = object.__new__(YDB)
    store.config = YDBSettings()
    with pytest.raises(ValueError, match="fulltext_index_enabled"):
        store.as_fulltext_retriever()


@pytest.mark.asyncio
async def test_async_fulltext_requires_index_flag() -> None:
    store = object.__new__(AsyncYDB)
    store.config = YDBSettings()
    with pytest.raises(ValueError, match="fulltext_index_enabled"):
        await store.afulltext_search("query")


@pytest.mark.parametrize(
    "method_name",
    ["fulltext_match", "fulltext_search", "fulltext_search_with_score"],
)
def test_async_store_rejects_sync_fulltext_methods(method_name: str) -> None:
    store = object.__new__(AsyncYDB)
    with pytest.raises(NotImplementedError, match="await afulltext"):
        getattr(store, method_name)("query")


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
