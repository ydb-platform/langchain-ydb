"""Test YDB functionality."""

from typing import Optional

import pytest
import ydb
from langchain_core.documents import Document
from ydb_dbapi.utils import prepare_credentials

from langchain_ydb.vectorstores import (
    YDB,
    AsyncYDB,
    YDBSearchStrategy,
    YDBSettings,
)

from .fake_embeddings import ConsistentFakeEmbeddings


def document_eq(doc1: Document, doc2: Document, check_id: bool = False) -> bool:
    """Compare two documents, optionally checking the id."""
    return (
        doc1.page_content == doc2.page_content
        and doc1.metadata == doc2.metadata
        and (not check_id or doc1.id == doc2.id)
    )


def _driver_config_for_tests(config: YDBSettings) -> ydb.DriverConfig:
    protocol = "grpcs" if config.secure else "grpc"
    endpoint = f"{protocol}://{config.host}:{config.port}"
    creds = prepare_credentials(
        {"username": config.username, "password": config.password}
        if config.username
        else None
    )
    root_certificates = ydb.load_ydb_root_certificate(None)
    return ydb.DriverConfig(
        endpoint=endpoint,
        database=config.database,
        credentials=creds,
        root_certificates=root_certificates,
    )


def make_sync_driver_and_session_pool(
    config: YDBSettings,
) -> tuple[ydb.Driver, ydb.QuerySessionPool]:
    driver = ydb.Driver(_driver_config_for_tests(config))
    driver.wait(timeout=10, fail_fast=True)
    pool = ydb.QuerySessionPool(driver, size=5)
    return driver, pool


async def make_async_driver_and_session_pool(
    config: YDBSettings,
) -> tuple[ydb.aio.Driver, ydb.aio.QuerySessionPool]:
    driver = ydb.aio.Driver(_driver_config_for_tests(config))
    await driver.wait(timeout=10, fail_fast=True)
    pool = ydb.aio.QuerySessionPool(driver, size=5)
    return driver, pool


class TestYDBVectorStore:
    """Sync :class:`YDB` plus LangChain ``a*`` APIs (executor wraps sync I/O)."""

    @pytest.mark.parametrize("vector_pass_as_bytes", [True, False])
    def test_from_texts_and_similarity_search(self, vector_pass_as_bytes: bool) -> None:
        """Test end to end construction and search."""
        texts = ["foo", "bar", "baz"]
        config = YDBSettings(
            drop_existing_table=True,
            vector_pass_as_bytes=vector_pass_as_bytes,
        )
        config.table = "test_ydb"
        docsearch = YDB.from_texts(texts, ConsistentFakeEmbeddings(), config=config)
        output = docsearch.similarity_search("foo", k=1)
        assert document_eq(output[0], Document(page_content="foo"))
        docsearch.drop()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("vector_pass_as_bytes", [True, False])
    async def test_asimilarity_search_uses_executor(
        self, vector_pass_as_bytes: bool
    ) -> None:
        """Default async search on sync YDB uses a thread pool (LangChain)."""
        texts = ["foo", "bar", "baz"]
        config = YDBSettings(
            drop_existing_table=True,
            vector_pass_as_bytes=vector_pass_as_bytes,
        )
        config.table = "test_ydb_executor_async"
        docsearch = YDB.from_texts(texts, ConsistentFakeEmbeddings(), config=config)
        output = await docsearch.asimilarity_search("foo", k=1)
        assert document_eq(output[0], Document(page_content="foo"))
        docsearch.drop()

    def test_with_custom_column_names(self) -> None:
        """Test end to end construction and search with custom col names."""
        texts = ["foo", "bar", "baz"]
        config = YDBSettings(
            drop_existing_table=True,
            column_map={
                "id": "custom_id",
                "document": "custom_document",
                "embedding": "custom_embedding",
                "metadata": "custom_metadata",
            },
        )
        config.table = "test_ydb_custom_col_names"
        docsearch = YDB.from_texts(texts, ConsistentFakeEmbeddings(), config=config)
        output = docsearch.similarity_search("bar", k=1)
        assert document_eq(output[0], Document(page_content="bar"))
        docsearch.drop()

    def test_no_texts_loss_with_batches(self) -> None:
        """Test end to end construction and search with custom col names."""
        n = 50
        texts = [f"{i}" for i in range(n)]
        config = YDBSettings(
            drop_existing_table=True,
        )
        config.table = "test_ydb_batches"
        docsearch = YDB.from_texts(texts, ConsistentFakeEmbeddings(), config=config)
        output = docsearch.similarity_search("text", k=n + 1)
        assert len(output) == n
        docsearch.drop()

    def test_create_with_metadatas(self) -> None:
        """Test end to end construction with metadatas."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i)} for i in range(len(texts))]
        config = YDBSettings(drop_existing_table=True)
        config.table = "test_ydb_with_metadatas"
        docsearch = YDB.from_texts(
            texts=texts,
            embedding=ConsistentFakeEmbeddings(),
            config=config,
            metadatas=metadatas,
        )
        output = docsearch.similarity_search("foo", k=1)
        assert document_eq(
            output[0], Document(page_content="foo", metadata={"page": "0"})
        )
        docsearch.drop()

    def test_create_with_metadatas_different_len_raises(self) -> None:
        """Test end to end construction with metadatas different len raises."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i)} for i in range(10)]
        config = YDBSettings(drop_existing_table=True)
        config.table = "test_ydb_with_metadatas"

        with pytest.raises(ValueError):
            YDB.from_texts(
                texts=texts,
                embedding=ConsistentFakeEmbeddings(),
                config=config,
                metadatas=metadatas,
            )

    def test_create_with_empty_metadatas(self) -> None:
        """Test end to end construction with empty metadatas."""
        texts = ["foo", "bar", "baz"]
        metadatas: list[dict] = [{} for _ in range(len(texts))]
        config = YDBSettings(drop_existing_table=True)
        config.table = "test_ydb_with_metadatas"
        docsearch = YDB.from_texts(
            texts=texts,
            embedding=ConsistentFakeEmbeddings(),
            config=config,
            metadatas=metadatas,
        )
        output = docsearch.similarity_search("foo", k=1)
        assert document_eq(output[0], Document(page_content="foo"))
        docsearch.drop()

    def test_text_with_escape_chars(self) -> None:
        """Test end to end construction with empty metadatas."""
        texts = [
            """
        Some text \\that 'should' "have" escape chars.
        One more line.
    """
        ]
        config = YDBSettings(drop_existing_table=True)
        config.table = "test_ydb_with_metadatas"
        docsearch = YDB.from_texts(
            texts=texts,
            embedding=ConsistentFakeEmbeddings(),
            config=config,
        )

        output = docsearch.similarity_search("foo", k=1)
        assert len(output) == 1

        docsearch.drop()

    def test_delete_all(self) -> None:
        """Test delete without specified ids."""
        texts = ["foo", "bar", "baz"]
        config = YDBSettings(drop_existing_table=True)
        config.table = "test_delete"
        docsearch = YDB(
            embedding=ConsistentFakeEmbeddings(),
            config=config,
        )

        docsearch.add_texts(texts)

        docsearch.delete()

        output = docsearch.similarity_search("sometext", k=1)
        assert output == []

        docsearch.drop()

    def test_id_persistence(self) -> None:
        """Test id persistence."""
        texts = ["foo", "bar", "baz"]
        ids = ["1", "2", "3"]
        config = YDBSettings(drop_existing_table=True)
        config.table = "test_id_persistence"
        docsearch = YDB.from_texts(
            texts=texts,
            embedding=ConsistentFakeEmbeddings(),
            config=config,
            ids=ids,
        )
        output = docsearch.similarity_search("foo", k=1)
        assert document_eq(
            output[0], Document(page_content="foo", id="1"), check_id=True
        )
        output = docsearch.similarity_search("bar", k=1)
        assert document_eq(
            output[0], Document(page_content="bar", id="2"), check_id=True
        )
        output = docsearch.similarity_search("baz", k=1)
        assert document_eq(
            output[0], Document(page_content="baz", id="3"), check_id=True
        )
        docsearch.drop()

    def test_delete_with_ids(self) -> None:
        """Test delete with specified ids."""
        texts = ["foo", "bar", "baz"]
        config = YDBSettings(drop_existing_table=True)
        config.table = "test_delete"
        docsearch = YDB(
            embedding=ConsistentFakeEmbeddings(),
            config=config,
        )

        ids = docsearch.add_texts(texts)

        docsearch.delete(ids[:2])

        output = docsearch.similarity_search("sometext", k=1)
        assert document_eq(output[0], Document(page_content="baz"))

        docsearch.drop()

    def test_search_with_filter(self) -> None:
        """Test end to end construction search with filter."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i)} for i in range(len(texts))]
        config = YDBSettings(drop_existing_table=True)
        config.table = "test_ydb_with_metadatas"
        docsearch = YDB.from_texts(
            texts=texts,
            embedding=ConsistentFakeEmbeddings(),
            config=config,
            metadatas=metadatas,
        )

        output = docsearch.similarity_search("sometext", filter={"page": "0"}, k=1)
        assert document_eq(
            output[0], Document(page_content="foo", metadata={"page": "0"})
        )

        output = docsearch.similarity_search("sometext", filter={"page": "1"}, k=1)
        assert document_eq(
            output[0], Document(page_content="bar", metadata={"page": "1"})
        )

        output = docsearch.similarity_search("sometext", filter={"page": "2"}, k=1)
        assert document_eq(
            output[0], Document(page_content="baz", metadata={"page": "2"})
        )

        docsearch.drop()

    def test_search_with_complex_filter(self) -> None:
        """Test end to end construction search with filter."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i), "index": str(i)} for i in range(len(texts))]
        config = YDBSettings(drop_existing_table=True)
        config.table = "test_ydb_with_complex_metadatas"
        docsearch = YDB.from_texts(
            texts=texts,
            embedding=ConsistentFakeEmbeddings(),
            config=config,
            metadatas=metadatas,
        )

        output = docsearch.similarity_search("sometext", filter={"page": "0"}, k=1)
        assert len(output) == 1
        assert output[0].page_content == "foo"

        output = docsearch.similarity_search(
            "sometext", filter={"page": "1", "index": "1"}, k=1
        )
        assert len(output) == 1
        assert output[0].page_content == "bar"

        output = docsearch.similarity_search(
            "sometext", filter={"page": "1", "index": "2"}, k=1
        )
        assert len(output) == 0

        docsearch.drop()

    @pytest.mark.parametrize(
        "strategy",
        [
            (YDBSearchStrategy.COSINE_DISTANCE),
            (YDBSearchStrategy.COSINE_SIMILARITY),
            (YDBSearchStrategy.EUCLIDEAN_DISTANCE),
            (YDBSearchStrategy.INNER_PRODUCT_SIMILARITY),
            (YDBSearchStrategy.MANHATTAN_DISTANCE),
        ],
    )
    def test_different_search_strategies(self, strategy: YDBSearchStrategy) -> None:
        """Test end to end construction and search with specified strategy."""
        texts = ["foo", "bar", "baz"]
        config = YDBSettings(
            drop_existing_table=True,
            strategy=strategy,
        )
        config.table = "test_ydb_with_different_search_strategies"
        docsearch = YDB.from_texts(
            texts=texts,
            embedding=ConsistentFakeEmbeddings(),
            config=config,
        )

        output = docsearch.similarity_search("foo", k=1)
        assert document_eq(output[0], Document(page_content="foo"))

        docsearch.drop()

    def test_search_with_score(self) -> None:
        """Test end to end construction with search with score."""
        texts = ["foo", "bar", "baz"]
        config = YDBSettings(drop_existing_table=True)
        config.table = "test_ydb"
        docsearch = YDB.from_texts(texts, ConsistentFakeEmbeddings(), config=config)
        output = docsearch.similarity_search_with_score("foo", k=1)
        assert document_eq(output[0][0], Document(page_content="foo"))
        docsearch.drop()

    def test_persistence(self) -> None:
        """Test YDB with persistence."""

        texts = ["foo", "bar", "baz"]
        config = YDBSettings(drop_existing_table=True)
        config.table = "test_ydb_with_persistence"
        embeddings = ConsistentFakeEmbeddings()
        docsearch = YDB.from_texts(texts, embeddings, config=config)
        output = docsearch.similarity_search("foo", k=1)
        assert document_eq(output[0], Document(page_content="foo"))

        config = YDBSettings()
        config.table = "test_ydb_with_persistence"
        docsearch = YDB(embedding=embeddings, config=config)
        output = docsearch.similarity_search("foo", k=1)
        assert document_eq(output[0], Document(page_content="foo"))

        docsearch.drop()

    def test_retriever_interface(self) -> None:
        """Test end to end construction with search from retriever interface."""
        texts = ["foo", "bar", "baz"]
        config = YDBSettings(drop_existing_table=True)
        config.table = "test_ydb"
        docsearch = YDB.from_texts(texts, ConsistentFakeEmbeddings(), config=config)

        retriever = docsearch.as_retriever(search_kwargs={"k": 1})

        output = retriever.invoke("foo")
        assert document_eq(output[0], Document(page_content="foo"))
        docsearch.drop()

    def test_retriever_interface_with_filter(self) -> None:
        """Test search with filter from retriever interface."""
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i)} for i in range(len(texts))]
        config = YDBSettings(drop_existing_table=True)
        config.table = "test_ydb_with_metadatas"
        docsearch = YDB.from_texts(
            texts=texts,
            embedding=ConsistentFakeEmbeddings(),
            config=config,
            metadatas=metadatas,
        )

        retriever = docsearch.as_retriever(search_kwargs={"k": 1})

        output = retriever.invoke("sometext", filter={"page": "1"})
        assert document_eq(
            output[0], Document(page_content="bar", metadata={"page": "1"})
        )

        docsearch.drop()

    @pytest.mark.parametrize(
        "n, batch_embeddings",
        [
            (10, False),
            (50, False),
            (100, False),
            (10, True),
            (50, True),
            (100, True),
        ],
    )
    def test_batch_insertion(self, n: int, batch_embeddings: bool) -> None:
        """Test batch insertion with different document counts."""
        texts = [f"text_{i}" for i in range(n)]
        metadatas = [{"index": str(i)} for i in range(n)]

        config = YDBSettings(drop_existing_table=True)
        config.table = f"test_ydb_batch_{n}"
        docsearch = YDB.from_texts(
            texts=texts,
            embedding=ConsistentFakeEmbeddings(),
            config=config,
            metadatas=metadatas,
            batch_embeddings=batch_embeddings,
        )

        all_results = docsearch.similarity_search("text", k=n + 1)
        assert len(all_results) == n

        docsearch.drop()

    @pytest.mark.parametrize("n,batch_size", [(25, None), (50, 10), (100, 50)])
    def test_batch_insertion_with_add_texts(
        self, n: int, batch_size: Optional[int]
    ) -> None:
        """Test add_texts with different document counts and batch sizes."""
        config = YDBSettings(drop_existing_table=True)
        config.table = f"test_ydb_add_texts_batch_{n}_{batch_size}"
        docsearch = YDB(
            embedding=ConsistentFakeEmbeddings(),
            config=config,
        )

        texts = [f"text_{i}" for i in range(n)]
        metadatas = [{"index": str(i)} for i in range(n)]

        with pytest.MonkeyPatch.context() as mp:
            processed_batches = []

            def mock_embed_documents(texts):
                return [[0.1] * 5 for _ in range(len(texts))]

            def mock_execute_query(query, params=None, ddl=False):
                processed_batches.append(len(params["$documents"][0]))
                return None

            mp.setattr(
                docsearch.embedding_function, "embed_documents", mock_embed_documents
            )
            mp.setattr(docsearch, "_execute_query", mock_execute_query)

            if batch_size is not None:
                ids = docsearch.add_texts(
                    texts=texts,
                    metadatas=metadatas,
                    batch_size=batch_size,
                )
            else:
                ids = docsearch.add_texts(
                    texts=texts,
                    metadatas=metadatas,
                )

            assert len(ids) == n

            expected_batch_size = batch_size if batch_size is not None else 32
            expected_num_batches = (n + expected_batch_size - 1) // expected_batch_size

            assert len(processed_batches) == expected_num_batches

            assert sum(processed_batches) == n

            for i, bs in enumerate(processed_batches):
                if i < len(processed_batches) - 1:
                    assert bs == expected_batch_size
                else:
                    assert bs <= expected_batch_size

        docsearch.drop()

    @pytest.mark.parametrize(
        "strategy",
        [
            (YDBSearchStrategy.COSINE_DISTANCE),
            (YDBSearchStrategy.COSINE_SIMILARITY),
            (YDBSearchStrategy.EUCLIDEAN_DISTANCE),
            (YDBSearchStrategy.INNER_PRODUCT_SIMILARITY),
            (YDBSearchStrategy.MANHATTAN_DISTANCE),
        ],
    )
    def test_basic_vector_index(self, strategy: YDBSearchStrategy) -> None:
        """Test end to end construction and search with specified strategy."""
        texts = ["foo", "bar", "baz"]
        config = YDBSettings(
            drop_existing_table=True,
            strategy=strategy,
            index_enabled=True,
        )
        config.table = "test_ydb_with_vector_index"
        docsearch = YDB.from_texts(
            texts=texts,
            embedding=ConsistentFakeEmbeddings(),
            config=config,
        )

        output = docsearch.similarity_search("foo", k=1)
        assert document_eq(output[0], Document(page_content="foo"))

        docsearch.drop()

    def test_reindex(self) -> None:
        """Test end to end construction and search with specified strategy."""
        texts = ["foo", "bar", "baz"]
        config = YDBSettings(
            drop_existing_table=True,
            index_enabled=True,
        )
        config.table = "test_ydb_with_vector_index"
        docsearch = YDB.from_texts(
            texts=texts,
            embedding=ConsistentFakeEmbeddings(),
            config=config,
        )

        output = docsearch.similarity_search("foo", k=1)
        assert document_eq(output[0], Document(page_content="foo"))

        docsearch.add_texts(["qwe", "asd", "zxc"])

        output = docsearch.similarity_search("foo", k=1)
        assert document_eq(output[0], Document(page_content="foo"))

        output = docsearch.similarity_search("zxc", k=1)
        assert document_eq(output[0], Document(page_content="zxc"))

        docsearch.drop()

    def test_with_external_query_session_pool(self) -> None:
        """``ydb_session_pool`` is passed through to ydb-dbapi (shared driver/pool)."""
        texts = ["foo", "bar", "baz"]
        config = YDBSettings(drop_existing_table=True)
        config.table = "test_ydb_external_session_pool"
        driver, pool = make_sync_driver_and_session_pool(config)
        docsearch: Optional[YDB] = None
        try:
            docsearch = YDB(
                ConsistentFakeEmbeddings(),
                config=config,
                ydb_session_pool=pool,
            )
            docsearch.add_texts(texts)
            output = docsearch.similarity_search("foo", k=1)
            assert document_eq(output[0], Document(page_content="foo"))
        finally:
            if docsearch is not None:
                docsearch.drop()
                docsearch.connection.close()
            pool.stop()
            driver.stop()


class TestAsyncYDBVectorStore:
    """Parity with :class:`TestYDBVectorStore` using native asyncio I/O."""

    @staticmethod
    async def _finish(store: AsyncYDB) -> None:
        await store.adrop()
        await store.aclose()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("vector_pass_as_bytes", [True, False])
    async def test_from_texts_and_similarity_search(
        self, vector_pass_as_bytes: bool
    ) -> None:
        texts = ["foo", "bar", "baz"]
        config = YDBSettings(
            drop_existing_table=True,
            vector_pass_as_bytes=vector_pass_as_bytes,
        )
        config.table = "test_async_ydb"
        docsearch = await AsyncYDB.afrom_texts(
            texts=texts, embedding=ConsistentFakeEmbeddings(), config=config
        )
        output = await docsearch.asimilarity_search("foo", k=1)
        assert document_eq(output[0], Document(page_content="foo"))
        await self._finish(docsearch)

    @pytest.mark.asyncio
    async def test_with_custom_column_names(self) -> None:
        texts = ["foo", "bar", "baz"]
        config = YDBSettings(
            drop_existing_table=True,
            column_map={
                "id": "custom_id",
                "document": "custom_document",
                "embedding": "custom_embedding",
                "metadata": "custom_metadata",
            },
        )
        config.table = "test_async_custom_col_names"
        docsearch = await AsyncYDB.afrom_texts(
            texts=texts, embedding=ConsistentFakeEmbeddings(), config=config
        )
        output = await docsearch.asimilarity_search("bar", k=1)
        assert document_eq(output[0], Document(page_content="bar"))
        await self._finish(docsearch)

    @pytest.mark.asyncio
    async def test_no_texts_loss_with_batches(self) -> None:
        n = 50
        texts = [f"{i}" for i in range(n)]
        config = YDBSettings(drop_existing_table=True)
        config.table = "test_async_batches"
        docsearch = await AsyncYDB.afrom_texts(
            texts=texts, embedding=ConsistentFakeEmbeddings(), config=config
        )
        output = await docsearch.asimilarity_search("text", k=n + 1)
        assert len(output) == n
        await self._finish(docsearch)

    @pytest.mark.asyncio
    async def test_create_with_metadatas(self) -> None:
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i)} for i in range(len(texts))]
        config = YDBSettings(drop_existing_table=True)
        config.table = "test_async_metadatas"
        docsearch = await AsyncYDB.afrom_texts(
            texts=texts,
            embedding=ConsistentFakeEmbeddings(),
            config=config,
            metadatas=metadatas,
        )
        output = await docsearch.asimilarity_search("foo", k=1)
        assert document_eq(
            output[0], Document(page_content="foo", metadata={"page": "0"})
        )
        await self._finish(docsearch)

    @pytest.mark.asyncio
    async def test_create_with_metadatas_different_len_raises(self) -> None:
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i)} for i in range(10)]
        config = YDBSettings(drop_existing_table=True)
        config.table = "test_async_metadatas_mismatch"
        store = await AsyncYDB.create(
            ConsistentFakeEmbeddings(),
            config=config,
        )
        try:
            with pytest.raises(ValueError):
                await store.aadd_texts(texts=texts, metadatas=metadatas)
        finally:
            await self._finish(store)

    @pytest.mark.asyncio
    async def test_create_with_empty_metadatas(self) -> None:
        texts = ["foo", "bar", "baz"]
        metadatas: list[dict] = [{} for _ in range(len(texts))]
        config = YDBSettings(drop_existing_table=True)
        config.table = "test_async_empty_metadatas"
        docsearch = await AsyncYDB.afrom_texts(
            texts=texts,
            embedding=ConsistentFakeEmbeddings(),
            config=config,
            metadatas=metadatas,
        )
        output = await docsearch.asimilarity_search("foo", k=1)
        assert document_eq(output[0], Document(page_content="foo"))
        await self._finish(docsearch)

    @pytest.mark.asyncio
    async def test_text_with_escape_chars(self) -> None:
        texts = [
            """
        Some text \\that 'should' "have" escape chars.
        One more line.
    """
        ]
        config = YDBSettings(drop_existing_table=True)
        config.table = "test_async_escape"
        docsearch = await AsyncYDB.afrom_texts(
            texts=texts,
            embedding=ConsistentFakeEmbeddings(),
            config=config,
        )
        output = await docsearch.asimilarity_search("foo", k=1)
        assert len(output) == 1
        await self._finish(docsearch)

    @pytest.mark.asyncio
    async def test_delete_all(self) -> None:
        texts = ["foo", "bar", "baz"]
        config = YDBSettings(drop_existing_table=True)
        config.table = "test_async_delete_all"
        docsearch = await AsyncYDB.create(
            embedding=ConsistentFakeEmbeddings(),
            config=config,
        )
        await docsearch.aadd_texts(texts)
        await docsearch.adelete()
        output = await docsearch.asimilarity_search("sometext", k=1)
        assert output == []
        await self._finish(docsearch)

    @pytest.mark.asyncio
    async def test_id_persistence(self) -> None:
        texts = ["foo", "bar", "baz"]
        ids = ["1", "2", "3"]
        config = YDBSettings(drop_existing_table=True)
        config.table = "test_async_id_persistence"
        docsearch = await AsyncYDB.afrom_texts(
            texts=texts,
            embedding=ConsistentFakeEmbeddings(),
            config=config,
            ids=ids,
        )
        output = await docsearch.asimilarity_search("foo", k=1)
        assert document_eq(
            output[0], Document(page_content="foo", id="1"), check_id=True
        )
        output = await docsearch.asimilarity_search("bar", k=1)
        assert document_eq(
            output[0], Document(page_content="bar", id="2"), check_id=True
        )
        output = await docsearch.asimilarity_search("baz", k=1)
        assert document_eq(
            output[0], Document(page_content="baz", id="3"), check_id=True
        )
        await self._finish(docsearch)

    @pytest.mark.asyncio
    async def test_delete_with_ids(self) -> None:
        texts = ["foo", "bar", "baz"]
        config = YDBSettings(drop_existing_table=True)
        config.table = "test_async_delete_ids"
        docsearch = await AsyncYDB.create(
            embedding=ConsistentFakeEmbeddings(),
            config=config,
        )
        ids = await docsearch.aadd_texts(texts)
        await docsearch.adelete(ids[:2])
        output = await docsearch.asimilarity_search("sometext", k=1)
        assert document_eq(output[0], Document(page_content="baz"))
        await self._finish(docsearch)

    @pytest.mark.asyncio
    async def test_search_with_filter(self) -> None:
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i)} for i in range(len(texts))]
        config = YDBSettings(drop_existing_table=True)
        config.table = "test_async_filter"
        docsearch = await AsyncYDB.afrom_texts(
            texts=texts,
            embedding=ConsistentFakeEmbeddings(),
            config=config,
            metadatas=metadatas,
        )
        output = await docsearch.asimilarity_search(
            "sometext", filter={"page": "0"}, k=1
        )
        assert document_eq(
            output[0], Document(page_content="foo", metadata={"page": "0"})
        )
        output = await docsearch.asimilarity_search(
            "sometext", filter={"page": "1"}, k=1
        )
        assert document_eq(
            output[0], Document(page_content="bar", metadata={"page": "1"})
        )
        output = await docsearch.asimilarity_search(
            "sometext", filter={"page": "2"}, k=1
        )
        assert document_eq(
            output[0], Document(page_content="baz", metadata={"page": "2"})
        )
        await self._finish(docsearch)

    @pytest.mark.asyncio
    async def test_search_with_complex_filter(self) -> None:
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i), "index": str(i)} for i in range(len(texts))]
        config = YDBSettings(drop_existing_table=True)
        config.table = "test_async_complex_filter"
        docsearch = await AsyncYDB.afrom_texts(
            texts=texts,
            embedding=ConsistentFakeEmbeddings(),
            config=config,
            metadatas=metadatas,
        )
        output = await docsearch.asimilarity_search(
            "sometext", filter={"page": "0"}, k=1
        )
        assert len(output) == 1
        assert output[0].page_content == "foo"
        output = await docsearch.asimilarity_search(
            "sometext", filter={"page": "1", "index": "1"}, k=1
        )
        assert len(output) == 1
        assert output[0].page_content == "bar"
        output = await docsearch.asimilarity_search(
            "sometext", filter={"page": "1", "index": "2"}, k=1
        )
        assert len(output) == 0
        await self._finish(docsearch)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "strategy",
        [
            (YDBSearchStrategy.COSINE_DISTANCE),
            (YDBSearchStrategy.COSINE_SIMILARITY),
            (YDBSearchStrategy.EUCLIDEAN_DISTANCE),
            (YDBSearchStrategy.INNER_PRODUCT_SIMILARITY),
            (YDBSearchStrategy.MANHATTAN_DISTANCE),
        ],
    )
    async def test_different_search_strategies(
        self, strategy: YDBSearchStrategy
    ) -> None:
        texts = ["foo", "bar", "baz"]
        config = YDBSettings(
            drop_existing_table=True,
            strategy=strategy,
        )
        config.table = f"test_async_strategies_{strategy.value}"
        docsearch = await AsyncYDB.afrom_texts(
            texts=texts,
            embedding=ConsistentFakeEmbeddings(),
            config=config,
        )
        output = await docsearch.asimilarity_search("foo", k=1)
        assert document_eq(output[0], Document(page_content="foo"))
        await self._finish(docsearch)

    @pytest.mark.asyncio
    async def test_search_with_score(self) -> None:
        texts = ["foo", "bar", "baz"]
        config = YDBSettings(drop_existing_table=True)
        config.table = "test_async_score"
        docsearch = await AsyncYDB.afrom_texts(
            texts=texts, embedding=ConsistentFakeEmbeddings(), config=config
        )
        output = await docsearch.asimilarity_search_with_score("foo", k=1)
        assert document_eq(output[0][0], Document(page_content="foo"))
        await self._finish(docsearch)

    @pytest.mark.asyncio
    async def test_persistence(self) -> None:
        texts = ["foo", "bar", "baz"]
        config = YDBSettings(drop_existing_table=True)
        config.table = "test_async_persistence"
        embeddings = ConsistentFakeEmbeddings()
        docsearch = await AsyncYDB.afrom_texts(texts, embeddings, config=config)
        output = await docsearch.asimilarity_search("foo", k=1)
        assert document_eq(output[0], Document(page_content="foo"))
        await docsearch.aclose()

        config = YDBSettings()
        config.table = "test_async_persistence"
        docsearch = await AsyncYDB.create(embeddings, config=config)
        output = await docsearch.asimilarity_search("foo", k=1)
        assert document_eq(output[0], Document(page_content="foo"))
        await self._finish(docsearch)

    @pytest.mark.asyncio
    async def test_retriever_interface(self) -> None:
        texts = ["foo", "bar", "baz"]
        config = YDBSettings(drop_existing_table=True)
        config.table = "test_async_retriever"
        docsearch = await AsyncYDB.afrom_texts(
            texts=texts, embedding=ConsistentFakeEmbeddings(), config=config
        )
        retriever = docsearch.as_retriever(search_kwargs={"k": 1})
        output = await retriever.ainvoke("foo")
        assert document_eq(output[0], Document(page_content="foo"))
        await self._finish(docsearch)

    @pytest.mark.asyncio
    async def test_retriever_interface_with_filter(self) -> None:
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": str(i)} for i in range(len(texts))]
        config = YDBSettings(drop_existing_table=True)
        config.table = "test_async_retriever_filter"
        docsearch = await AsyncYDB.afrom_texts(
            texts=texts,
            embedding=ConsistentFakeEmbeddings(),
            config=config,
            metadatas=metadatas,
        )
        retriever = docsearch.as_retriever(search_kwargs={"k": 1})
        output = await retriever.ainvoke("sometext", filter={"page": "1"})
        assert document_eq(
            output[0], Document(page_content="bar", metadata={"page": "1"})
        )
        await self._finish(docsearch)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "n, batch_embeddings",
        [
            (10, False),
            (50, False),
            (100, False),
            (10, True),
            (50, True),
            (100, True),
        ],
    )
    async def test_batch_insertion(self, n: int, batch_embeddings: bool) -> None:
        texts = [f"text_{i}" for i in range(n)]
        metadatas = [{"index": str(i)} for i in range(n)]
        config = YDBSettings(drop_existing_table=True)
        config.table = f"test_async_batch_{n}_{batch_embeddings}"
        docsearch = await AsyncYDB.afrom_texts(
            texts=texts,
            embedding=ConsistentFakeEmbeddings(),
            config=config,
            metadatas=metadatas,
            batch_embeddings=batch_embeddings,
        )
        all_results = await docsearch.asimilarity_search("text", k=n + 1)
        assert len(all_results) == n
        await self._finish(docsearch)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("n,batch_size", [(25, None), (50, 10), (100, 50)])
    async def test_batch_insertion_with_aadd_texts(
        self, n: int, batch_size: Optional[int]
    ) -> None:
        config = YDBSettings(drop_existing_table=True)
        config.table = f"test_async_aadd_batch_{n}_{batch_size}"
        docsearch = await AsyncYDB.create(
            embedding=ConsistentFakeEmbeddings(),
            config=config,
        )
        texts = [f"text_{i}" for i in range(n)]
        metadatas = [{"index": str(i)} for i in range(n)]

        with pytest.MonkeyPatch.context() as mp:
            processed_batches = []

            async def mock_aembed_documents(texts):
                return [[0.1] * 5 for _ in range(len(texts))]

            async def mock_execute_query_async(query, params=None, ddl=False):
                processed_batches.append(len(params["$documents"][0]))
                return []

            mp.setattr(
                docsearch.embedding_function,
                "aembed_documents",
                mock_aembed_documents,
            )
            mp.setattr(docsearch, "_execute_query_async", mock_execute_query_async)

            if batch_size is not None:
                ids = await docsearch.aadd_texts(
                    texts=texts,
                    metadatas=metadatas,
                    batch_size=batch_size,
                )
            else:
                ids = await docsearch.aadd_texts(
                    texts=texts,
                    metadatas=metadatas,
                )

            assert len(ids) == n
            expected_batch_size = batch_size if batch_size is not None else 32
            expected_num_batches = (n + expected_batch_size - 1) // expected_batch_size
            assert len(processed_batches) == expected_num_batches
            assert sum(processed_batches) == n
            for i, bs in enumerate(processed_batches):
                if i < len(processed_batches) - 1:
                    assert bs == expected_batch_size
                else:
                    assert bs <= expected_batch_size

        await self._finish(docsearch)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "strategy",
        [
            (YDBSearchStrategy.COSINE_DISTANCE),
            (YDBSearchStrategy.COSINE_SIMILARITY),
            (YDBSearchStrategy.EUCLIDEAN_DISTANCE),
            (YDBSearchStrategy.INNER_PRODUCT_SIMILARITY),
            (YDBSearchStrategy.MANHATTAN_DISTANCE),
        ],
    )
    async def test_basic_vector_index(self, strategy: YDBSearchStrategy) -> None:
        texts = ["foo", "bar", "baz"]
        config = YDBSettings(
            drop_existing_table=True,
            strategy=strategy,
            index_enabled=True,
        )
        config.table = f"test_async_vector_index_{strategy.value}"
        docsearch = await AsyncYDB.afrom_texts(
            texts=texts,
            embedding=ConsistentFakeEmbeddings(),
            config=config,
        )
        output = await docsearch.asimilarity_search("foo", k=1)
        assert document_eq(output[0], Document(page_content="foo"))
        await self._finish(docsearch)

    @pytest.mark.asyncio
    async def test_reindex(self) -> None:
        texts = ["foo", "bar", "baz"]
        config = YDBSettings(
            drop_existing_table=True,
            index_enabled=True,
        )
        config.table = "test_async_reindex"
        docsearch = await AsyncYDB.afrom_texts(
            texts=texts,
            embedding=ConsistentFakeEmbeddings(),
            config=config,
        )
        output = await docsearch.asimilarity_search("foo", k=1)
        assert document_eq(output[0], Document(page_content="foo"))
        await docsearch.aadd_texts(["qwe", "asd", "zxc"])
        output = await docsearch.asimilarity_search("foo", k=1)
        assert document_eq(output[0], Document(page_content="foo"))
        output = await docsearch.asimilarity_search("zxc", k=1)
        assert document_eq(output[0], Document(page_content="zxc"))
        await self._finish(docsearch)

    @pytest.mark.asyncio
    async def test_with_external_query_session_pool(self) -> None:
        """``ydb_session_pool`` for async connection shares aio driver/pool."""
        texts = ["foo", "bar", "baz"]
        config = YDBSettings(drop_existing_table=True)
        config.table = "test_async_external_session_pool"
        driver, pool = await make_async_driver_and_session_pool(config)
        docsearch: Optional[AsyncYDB] = None
        try:
            docsearch = await AsyncYDB.create(
                ConsistentFakeEmbeddings(),
                config=config,
                ydb_session_pool=pool,
            )
            await docsearch.aadd_texts(texts)
            output = await docsearch.asimilarity_search("foo", k=1)
            assert document_eq(output[0], Document(page_content="foo"))
        finally:
            if docsearch is not None:
                await docsearch.adrop()
                await docsearch.aclose()
            await pool.stop()
            await driver.stop()
