"""Test YDB functionality."""

import pytest
from langchain_core.documents import Document

from langchain_ydb.vectorstores import YDB, YDBSearchStrategy, YDBSettings

from .fake_embeddings import ConsistentFakeEmbeddings


def test_ydb() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    config = YDBSettings(drop_existing_table=True)
    config.table = "test_ydb"
    docsearch = YDB.from_texts(texts, ConsistentFakeEmbeddings(), config=config)
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]
    docsearch.drop()


@pytest.mark.asyncio
async def test_ydb_async() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    config = YDBSettings(drop_existing_table=True)
    config.table = "test_ydb_async"
    docsearch = YDB.from_texts(
        texts=texts, embedding=ConsistentFakeEmbeddings(), config=config
    )
    output = await docsearch.asimilarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]
    docsearch.drop()


def test_ydb_with_custom_column_names() -> None:
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
    assert output == [Document(page_content="bar")]
    docsearch.drop()


def test_no_texts_loss_with_batches() -> None:
    """Test end to end construction and search with custom col names."""
    n = 50
    texts = [f"{i}" for i in range(n)]
    config = YDBSettings(
        drop_existing_table=True,
    )
    config.table = "test_ydb_batches"
    docsearch = YDB.from_texts(texts, ConsistentFakeEmbeddings(), config=config)
    output = docsearch.similarity_search("text", k = n + 1)
    assert len(output) == n
    docsearch.drop()


def test_create_ydb_with_metadatas() -> None:
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
    assert output == [Document(page_content="foo", metadata={"page": "0"})]
    docsearch.drop()


def test_create_ydb_with_metadatas_different_len_raises() -> None:
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


def test_create_ydb_with_empty_metadatas() -> None:
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
    assert output == [Document(page_content="foo")]
    docsearch.drop()


def test_create_ydb_with_text_to_have_escape_chars() -> None:
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


def test_delete() -> None:
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


def test_delete_with_ids() -> None:
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
    assert output == [Document(page_content="baz")]

    docsearch.drop()


def test_search_with_filter() -> None:
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
    assert output == [Document(page_content="foo", metadata={"page": "0"})]

    output = docsearch.similarity_search("sometext", filter={"page": "1"}, k=1)
    assert output == [Document(page_content="bar", metadata={"page": "1"})]

    output = docsearch.similarity_search("sometext", filter={"page": "2"}, k=1)
    assert output == [Document(page_content="baz", metadata={"page": "2"})]

    docsearch.drop()


def test_search_with_complex_filter() -> None:
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
    assert output == []

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
def test_different_search_strategies(strategy: YDBSearchStrategy) -> None:
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
    assert output == [Document(page_content="foo")]

    docsearch.drop()


def test_search_with_score() -> None:
    """Test end to end construction with search with score."""
    texts = ["foo", "bar", "baz"]
    config = YDBSettings(drop_existing_table=True)
    config.table = "test_ydb"
    docsearch = YDB.from_texts(texts, ConsistentFakeEmbeddings(), config=config)
    output = docsearch.similarity_search_with_score("foo", k=1)
    assert output[0][0] == Document(page_content="foo")
    docsearch.drop()


def test_ydb_with_persistence() -> None:
    """Test YDB with persistence."""

    texts = ["foo", "bar", "baz"]
    config = YDBSettings(drop_existing_table=True)
    config.table = "test_ydb_with_persistence"
    embeddings = ConsistentFakeEmbeddings()
    docsearch = YDB.from_texts(texts, embeddings, config=config)
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    config = YDBSettings()
    config.table = "test_ydb_with_persistence"
    docsearch = YDB(embedding=embeddings, config=config)
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    docsearch.drop()


def test_search_from_retriever_interface() -> None:
    """Test end to end construction with search from retriever interface."""
    texts = ["foo", "bar", "baz"]
    config = YDBSettings(drop_existing_table=True)
    config.table = "test_ydb"
    docsearch = YDB.from_texts(texts, ConsistentFakeEmbeddings(), config=config)

    retriever = docsearch.as_retriever(search_kwargs={"k": 1})

    output = retriever.invoke("foo")
    assert output == [Document(page_content="foo")]
    docsearch.drop()


def test_search_from_retriever_interface_with_filter() -> None:
    """Test end to end construction with search with filter from retriever interface."""
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
    assert output == [Document(page_content="bar", metadata={"page": "1"})]

    docsearch.drop()


@pytest.mark.parametrize("n", [10, 50, 100])
def test_batch_insertion(n: int) -> None:
    """Test batch insertion with different document counts."""
    # Create documents
    texts = [f"text_{i}" for i in range(n)]
    metadatas = [{"index": str(i)} for i in range(n)]

    # Create vectorstore
    config = YDBSettings(drop_existing_table=True)
    config.table = f"test_ydb_batch_{n}"
    docsearch = YDB.from_texts(
        texts=texts,
        embedding=ConsistentFakeEmbeddings(),
        config=config,
        metadatas=metadatas,
    )

    # Verify total count matches expected
    all_results = docsearch.similarity_search("text", k = n + 1)
    assert len(all_results) == n

    # Clean up
    docsearch.drop()

@pytest.mark.parametrize("n,batch_size", [(25, None), (50, 10), (100, 50)])
def test_batch_insertion_with_add_texts(n: int, batch_size: int) -> None:
    """Test add_texts with different document counts and batch sizes."""
    # Setup
    config = YDBSettings(drop_existing_table=True)
    config.table = f"test_ydb_add_texts_batch_{n}_{batch_size}"
    docsearch = YDB(
        embedding=ConsistentFakeEmbeddings(),
        config=config,
    )

    # Create test data
    texts = [f"text_{i}" for i in range(n)]
    metadatas = [{"index": str(i)} for i in range(n)]

    # Mock the embedding and execute functions to verify batch behavior
    with pytest.MonkeyPatch.context() as mp:
        # Track batches
        processed_batches = []

        # Mock embedding function to track batch sizes
        def mock_embed_documents(texts):
            return [[0.1] * 5 for _ in range(len(texts))]

        # Mock execute query to avoid actual database operations
        def mock_execute_query(query, params=None, ddl=False):
            processed_batches.append(len(params["$documents"][0]))
            return None

        # Apply mocks
        mp.setattr(
            docsearch.embedding_function, "embed_documents", mock_embed_documents
        )
        mp.setattr(docsearch, "_execute_query", mock_execute_query)

        # Execute add_texts with specified batch size
        kwargs = {}
        if batch_size is not None:
            kwargs["batch_size"] = batch_size

        ids = docsearch.add_texts(
            texts=texts,
            metadatas=metadatas,
            **kwargs
        )

        # Verify results
        assert len(ids) == n  # Correct number of IDs returned

        # Verify correct batch sizes were used
        expected_batch_size = batch_size if batch_size is not None else 32
        expected_num_batches = (n + expected_batch_size - 1) // expected_batch_size

        assert len(processed_batches) == expected_num_batches

        # Verify all texts were processed in total
        assert sum(processed_batches) == n

        # Verify most batches are of the expected size (except possibly the last one)
        for i, batch_size in enumerate(processed_batches):
            if i < len(processed_batches) - 1:
                # All but the last batch should be full
                assert batch_size == expected_batch_size
            else:
                # Last batch can be smaller
                assert batch_size <= expected_batch_size

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
def test_basic_vector_index(strategy: YDBSearchStrategy) -> None:
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
    assert output == [Document(page_content="foo")]

    docsearch.drop()


def test_reindex() -> None:
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
    assert output == [Document(page_content="foo")]

    docsearch.add_texts(["qwe", "asd", "zxc"])

    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

    output = docsearch.similarity_search("zxc", k=1)
    assert output == [Document(page_content="zxc")]

    docsearch.drop()
