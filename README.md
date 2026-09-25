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
hybrid search combines fulltext relevance and vector similarity over the **same
table**. The text branch indexes `column_map["document"]`; the vector branch
indexes `column_map["embedding"]`. Enabling the feature creates either missing
index and waits for both indexes to be ready. This works for a new table and for
a table that already contains documents. Existing documents are indexed in place;
their embeddings are not recomputed.

#### Configure the store

These `YDBSettings` fields control hybrid search and its indexes:

| Field | Default | Effect |
| --- | --- | --- |
| `hybrid_search_enabled` | `False` | Enables the hybrid API and prepares both indexes when the store opens. It also enables the vector index for ordinary vector searches, so `index_enabled=True` is not required. |
| `fulltext_index_name` | `"ydb_fulltext_index"` | Name of the `fulltext_relevance` index on the document column. Set it to the name of an existing index to reuse that index. Otherwise the store creates an index with this name. |
| `hybrid_index_build_timeout` | `300.0` seconds | Time to wait for index readiness after issuing any creation statements. The statements themselves may take additional time. Increase it if backfilling a large table takes longer. A timeout raises `TimeoutError`; the indexes remain in the table. |
| `database` | `"/local"` | Database containing the table. Use the same database as the existing store. |
| `table` | `"ydb_langchain_store"` | Table to create or open in `database`. Use the existing table name when enabling hybrid search over stored documents. |
| `drop_existing_table` | `False` | If `True`, deletes the table before opening it. Keep it `False` when enabling hybrid search on existing data. |
| `index_enabled` | `False` | Enables indexed vector search independently of hybrid search. `hybrid_search_enabled=True` already implies it, so it can remain `False` for hybrid use. |
| `index_name` | `"ydb_vector_index"` | Name of the `vector_kmeans_tree` index on the embedding column. Set it to the existing index name to reuse that index. |
| `strategy` | `YDBSearchStrategy.COSINE_SIMILARITY` | Vector scoring function and index metric. Use the same strategy as an existing vector index. |
| `index_config_levels` | `2` | K-means tree depth when a vector index is created or rebuilt. |
| `index_config_clusters` | `128` | Number of clusters when a vector index is created or rebuilt. |
| `index_tree_search_top_size` | `1` | `ydb.KMeansTreeSearchTopSize` for the vector branch at query time. Increase it to search more tree candidates. |
| `vector_dimension` | `None` | Dimension for vector index creation or rebuild. When omitted, the store obtains it by calling the embedding model with `embed_query("index")` (or `aembed_query` for `AsyncYDB`). |
| `vector_pass_as_bytes` | `True` | Passes document and query vectors to YDB as binary `String` values. With `False`, passes `List<Float>` and converts them to the binary format in YQL. |
| `column_map` | `id`, `document`, `embedding`, `metadata` | Maps these four roles to columns in an existing table. The fulltext index uses the mapped document column and the vector index uses the mapped embedding column. |

If an existing table uses different column names, pass all four role mappings:

```python
settings = YDBSettings(
    table="my_documents",
    hybrid_search_enabled=True,
    column_map={
        "id": "doc_id",
        "document": "body",
        "embedding": "body_vector",
        "metadata": "attributes",
    },
)
```

The automatically created fulltext index uses
`GLOBAL USING fulltext_relevance` with
`WITH (tokenizer=standard, use_filter_lowercase=true)`. The tokenizer, lowercase
filter, and other fulltext index options are **not** `YDBSettings` fields. To use
different tokenization or normalization options, create a `fulltext_relevance`
index directly on the document column yourself and pass its name as
`fulltext_index_name`. The store reuses an index with that name; it checks the
indexed column and readiness but does not check the index type or tokenizer
settings.

For a new table:

```python
from langchain_openai import OpenAIEmbeddings
from langchain_ydb.vectorstores import YDB, YDBSettings

settings = YDBSettings(
    table="my_documents",
    hybrid_search_enabled=True,
    vector_dimension=1536,  # set this to your model's embedding size
)
store = YDB(OpenAIEmbeddings(), config=settings)
store.add_texts(["A document about databases"])

documents = store.hybrid_search("database", k=4)
retriever = store.as_hybrid_retriever(k=4)
documents = retriever.invoke("database")
```

To enable hybrid search on an existing table, open it with
`hybrid_search_enabled=True` and **leave `drop_existing_table=False`** (the
default). Set `index_name` and `fulltext_index_name` if the existing indexes
have custom names. A missing index is built over the existing rows; a ready
index with the configured name is reused. For example:

```python
settings = YDBSettings(
    table="my_documents",
    hybrid_search_enabled=True,
    index_name="existing_vector_index",
    fulltext_index_name="document_relevance_index",
    vector_dimension=1536,
)
store = YDB(OpenAIEmbeddings(), config=settings)
documents = store.hybrid_search("database", k=4)
```

If you add documents after opening the store, the fulltext index follows those
writes automatically and the vector index is rebuilt using the configured
`strategy`, `index_config_levels`, `index_config_clusters`, and
`vector_dimension`. Those creation settings do not alter an existing ready
vector index merely by opening the store.

For asynchronous I/O, use `await AsyncYDB.create(embeddings, config=settings)`,
`await store.ahybrid_search(...)`, and
`await store.as_hybrid_retriever(k=4).ainvoke(...)`. Close the store with
`await store.aclose()`.

#### Configure each query

`hybrid_search` and `ahybrid_search` take the same text query for fulltext
matching and embedding. Their query-time arguments are:

| Argument | Default | Effect |
| --- | --- | --- |
| `k` | `4` | Number of returned documents. Must be a positive integer. |
| `mode` | `"rrf"` | Fusion mode: reciprocal rank fusion (`"rrf"`) or normalized weighted scores (`"linear"`). |
| `weights` | `(1.0, 1.0)` | Non-negative weights in **(fulltext, vector)** order. |
| `candidate_limits` | `None` | Optional positive candidate counts in **(fulltext, vector)** order. By default, YDB uses `k * 10` candidates per branch. |

This gives the vector branch twice the weight:

```python
store.hybrid_search(
    "database", k=5, weights=(1.0, 2.0), candidate_limits=(50, 100)
)
retriever = store.as_hybrid_retriever(k=5, weights=(1.0, 2.0))
```

`similarity_search` and the ordinary `as_retriever()` remain vector-only;
use `as_hybrid_retriever()` for hybrid retrieval. Metadata `filter` is not
supported by the native hybrid query and raises `ValueError`. The API returns
documents in fused order, without a numeric fused score.

The [local RAG wiki notebook](examples/local_rag_wiki/Example.ipynb) compares
vector and hybrid results and uses the hybrid retriever in a RAG chain.

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
