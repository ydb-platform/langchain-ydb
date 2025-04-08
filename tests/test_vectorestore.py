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
    output = docsearch.similarity_search("text", k=n + 1)
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


def test_batch_insertion() -> None:
    """Test batch insertion with different batch sizes."""
    # Create a large number of documents
    n = 100
    texts = [f"text_{i}" for i in range(n)]
    metadatas = [{"index": str(i)} for i in range(n)]
    
    # Test with default batch size
    config = YDBSettings(drop_existing_table=True)
    config.table = "test_ydb_batch_default"
    docsearch1 = YDB.from_texts(
        texts=texts,
        embedding=ConsistentFakeEmbeddings(),
        config=config,
        metadatas=metadatas,
    )
    
    # Test with small batch size
    config = YDBSettings(drop_existing_table=True)
    config.table = "test_ydb_batch_small"
    docsearch2 = YDB.from_texts(
        texts=texts,
        embedding=ConsistentFakeEmbeddings(),
        config=config,
        metadatas=metadatas,
        batch_size=10,  # Small batch size
    )
    
    # Test with large batch size
    config = YDBSettings(drop_existing_table=True)
    config.table = "test_ydb_batch_large"
    docsearch3 = YDB.from_texts(
        texts=texts,
        embedding=ConsistentFakeEmbeddings(),
        config=config,
        metadatas=metadatas,
        batch_size=n,  # Entire dataset in one batch
    )
    
    # Verify all documents are properly inserted with all batch sizes
    for i, text in enumerate(texts):
        # Test default batch size
        results1 = docsearch1.similarity_search(
            text, k=1, filter={"index": str(i)}
        )
        assert len(results1) == 1
        assert results1[0].page_content == text
        assert results1[0].metadata["index"] == str(i)
        
        # Test small batch size
        results2 = docsearch2.similarity_search(
            text, k=1, filter={"index": str(i)}
        )
        assert len(results2) == 1
        assert results2[0].page_content == text
        assert results2[0].metadata["index"] == str(i)
        
        # Test large batch size
        results3 = docsearch3.similarity_search(
            text, k=1, filter={"index": str(i)}
        )
        assert len(results3) == 1
        assert results3[0].page_content == text
        assert results3[0].metadata["index"] == str(i)
    
    # Clean up
    docsearch1.drop()
    docsearch2.drop()
    docsearch3.drop()


def test_batch_insertion_with_add_texts() -> None:
    """Test add_texts with batch insertion."""
    # Create a vector store
    config = YDBSettings(drop_existing_table=True)
    config.table = "test_ydb_add_texts_batch"
    docsearch = YDB(
        embedding=ConsistentFakeEmbeddings(),
        config=config,
    )
    
    # Add texts in batches
    n = 50
    texts = [f"text_{i}" for i in range(n)]
    metadatas = [{"index": str(i)} for i in range(n)]
    
    # Add first half with default batch size
    ids1 = docsearch.add_texts(
        texts=texts[:n//2],
        metadatas=metadatas[:n//2],
    )
    
    # Add second half with custom batch size
    ids2 = docsearch.add_texts(
        texts=texts[n//2:],
        metadatas=metadatas[n//2:],
        batch_size=5,
    )
    
    # Verify all documents are inserted
    assert len(ids1) + len(ids2) == n
    
    # Test retrieval for all documents
    for i, text in enumerate(texts):
        results = docsearch.similarity_search(
            text, k=1, filter={"index": str(i)}
        )
        assert len(results) == 1
        assert results[0].page_content == text
        assert results[0].metadata["index"] == str(i)
    
    # Clean up
    docsearch.drop()
