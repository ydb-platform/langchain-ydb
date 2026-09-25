# langchain-ydb
---
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/ydb-platform/langchain-ydb/blob/main/LICENSE)
[![PyPI version](https://badge.fury.io/py/langchain-ydb.svg)](https://badge.fury.io/py/langchain-ydb)
[![Functional tests](https://github.com/ydb-platform/langchain-ydb/actions/workflows/tests.yml/badge.svg)](https://github.com/ydb-platform/langchain-ydb/actions/workflows/tests.yml)
[![Lint checks](https://github.com/ydb-platform/langchain-ydb/actions/workflows/lint.yml/badge.svg)](https://github.com/ydb-platform/langchain-ydb/actions/workflows/lint.yml)

LangChain's YDB integration (langchain-ydb) provides vector capabilities for working with [YDB](https://ydb.tech/).

## Getting Started

### Setting Up YDB

Launch a YDB Docker container with:

```shell
docker run -d -p 2136:2136 --name ydb-langchain -e YDB_USE_IN_MEMORY_PDISKS=true -h localhost ydbplatform/local-ydb:trunk
```

### Installing the Package

Install `langchain-ydb` package with:

```bash
pip install -U langchain-ydb
```

VectorStore works along with an embedding model, here using `langchain-openai` as example.

```shell
pip install langchain-openai
export OPENAI_API_KEY=...
```

## Work with YDB Vector Store

### Creating a Vector Store

```python
from langchain_openai import OpenAIEmbeddings
from langchain_ydb.vectorstores import YDB, YDBSearchStrategy, YDBSettings


settings = YDBSettings(
    host="localhost",
    port=2136,
    database="/local",
    table="ydb_example",
    strategy=YDBSearchStrategy.COSINE_SIMILARITY,
)

vector_store = YDB(
    OpenAIEmbeddings(),
    config=settings,
)
```

### Async vector store (`AsyncYDB`)

For native asyncio I/O (`ydb.aio` via `ydb-dbapi`), use `AsyncYDB` instead of `YDB`:

```python
from langchain_ydb.vectorstores import AsyncYDB, YDBSettings

store = await AsyncYDB.afrom_texts(
    ["hello", "world"],
    embeddings,
    config=YDBSettings(table="my_async_table"),
)
docs = await store.asimilarity_search("hello", k=1)
await store.aclose()
```

Sync methods on `AsyncYDB` are not supported; use `a*` APIs.

### Hybrid search

On a YDB server that supports [HybridRank](https://ydb.tech/docs/en/dev/hybrid-search?version=main),
enable hybrid search to combine fulltext relevance with vector similarity. The
store uses the existing `document` and `embedding` columns. It creates missing
`fulltext_relevance` and `vector_kmeans_tree` indexes and waits for them to be
ready. Opening a populated table with the same settings builds the missing indexes
over its existing documents; no data migration or re-embedding is needed.

```python
from langchain_ydb.vectorstores import YDB, YDBSettings

settings = YDBSettings(
    table="my_documents",
    hybrid_search_enabled=True,
    vector_dimension=1536,  # optional if the embedding model can be probed
)
store = YDB(embeddings, config=settings)
store.add_texts(["A document about databases"])

documents = store.hybrid_search("database", k=4)
retriever = store.as_hybrid_retriever(k=4)
documents = retriever.invoke("database")
```

Use the same settings and table name to open an existing store. Set `index_name`
and `fulltext_index_name` if those indexes already exist under custom names.
`hybrid_index_build_timeout` controls how long opening the store waits for index
backfill. For asynchronous I/O, use `await AsyncYDB.create(embeddings, config=settings)`,
`await store.ahybrid_search(...)`, and `await retriever.ainvoke(...)`.

The default fusion is RRF. `hybrid_search` and `ahybrid_search` also accept
`mode="linear"`, `weights=(text_weight, vector_weight)`, and
`candidate_limits=(text_candidates, vector_candidates)`. The existing
`similarity_search` methods remain vector-only. Metadata filters and a fused
numeric score are not available through the native hybrid query.

#### How to use Credentials

To use `YDB` credentials pass a `credentials` value into `YDBSettings`.

There are several ways to use credentials:

**Static Credentials**:

```python
settings = YDBSettings(credentials={"username": "name", "password": "pass"})

vector_store = YDB(embeddings, config=settings)
```

**Access Token Credentials**:

```python
settings = YDBSettings(credentials={"token": "zxc123"})

vector_store = YDB(embeddings, config=settings)
```

**Service Account Credentials**:

```python
settings = YDBSettings(credentials={
    "service_account_json": {
        "id": "...",
        "service_account_id": "...",
        "created_at": "...",
        "key_algorithm": "...",
        "public_key": "...",
        "private_key": "..."
    }
})

vector_store = YDB(embeddings, config=settings)
```

**Credentials Object From YDB SDK**:

Additionally, you can use any credentials that comes with `ydb` package. Example:

```python
import ydb.iam

settings = YDBSettings(credentials=ydb.iam.MetadataUrlCredentials())

vector_store = YDB(embeddings, config=settings)
```

### Add items to vector store

Once you have created your vector store, you can interact with it by adding and deleting different items.

Prepare documents to work with:

```python
from uuid import uuid4

from langchain_core.documents import Document

document_1 = Document(
    page_content="I had chocalate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
)

document_4 = Document(
    page_content="Robbers broke into the city bank and stole $1 million in cash.",
    metadata={"source": "news"},
)

document_5 = Document(
    page_content="Wow! That was an amazing movie. I can't wait to see it again.",
    metadata={"source": "tweet"},
)

document_6 = Document(
    page_content="Is the new iPhone worth the price? Read this review to find out.",
    metadata={"source": "website"},
)

document_7 = Document(
    page_content="The top 10 soccer players in the world right now.",
    metadata={"source": "website"},
)

document_8 = Document(
    page_content="LangGraph is the best framework for building stateful, agentic applications!",
    metadata={"source": "tweet"},
)

document_9 = Document(
    page_content="The stock market is down 500 points today due to fears of a recession.",
    metadata={"source": "news"},
)

document_10 = Document(
    page_content="I have a bad feeling I am going to get deleted :(",
    metadata={"source": "tweet"},
)

documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
]
uuids = [str(uuid4()) for _ in range(len(documents))]
```

You can add items to your vector store by using the `add_documents` function.

```python
vector_store.add_documents(documents=documents, ids=uuids)
```

### Delete items from vector store

You can delete items from your vector store by ID using the `delete` function.

```python
vector_store.delete(ids=[uuids[-1]])
```

### Query vector store

Once your vector store has been created and relevant documents have been added, you will likely want to query it during the execution of your chain or agent.

#### Query directly

**Similarity search**:

A simple similarity search can be performed as follows:

```python
results = vector_store.similarity_search(
    "LangChain provides abstractions to make working with LLMs easy", k=2
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")
```

**Similarity search with score**

You can also perform a search with a score:

```python
results = vector_store.similarity_search_with_score("Will it be hot tomorrow?", k=3)
for res, score in results:
    print(f"* [SIM={score:.3f}] {res.page_content} [{res.metadata}]")
```

#### Filtering

You can search with filters as described below:

```python
results = vector_store.similarity_search_with_score(
    "What did I eat for breakfast?",
    k=4,
    filter={"source": "tweet"},
)
for res, _ in results:
    print(f"* {res.page_content} [{res.metadata}]")
```

#### Query by turning into retriever

You can also transform the vector store into a retriever for easier usage in your chains.

Here's how to transform your vector store into a retriever and then invoke the retriever with a simple query and filter.

```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2},
)
results = retriever.invoke(
    "Stealing from the bank is a crime", filter={"source": "news"}
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")
```
