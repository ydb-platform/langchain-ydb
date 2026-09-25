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

### Configuration

Pass a `YDBSettings` instance to `YDB` or `AsyncYDB`. The fields below control
connections, table creation, and search. Defaults apply when a field is omitted.

#### Connection and table

| Field | Default | Purpose |
| --- | --- | --- |
| `host` | `"localhost"` | YDB host. |
| `port` | `2136` | gRPC port. |
| `credentials` | `None` | Authentication; see [Credentials](#how-to-use-credentials) below. |
| `secure` | `False` | Use `grpcs` instead of `grpc`. |
| `database` | `"/local"` | Database containing the table. |
| `table` | `"ydb_langchain_store"` | Table to create or open. |
| `column_map` | `id`, `document`, `embedding`, `metadata` | Maps these four roles to table columns. Supply all four names for a custom schema. |
| `drop_existing_table` | `False` | Drop and recreate the table when the store opens. Leave `False` to reuse existing data. |

For a table with custom column names:

```python
settings = YDBSettings(
    table="my_documents",
    column_map={
        "id": "doc_id",
        "document": "body",
        "embedding": "body_vector",
        "metadata": "attributes",
    },
)
```

#### Vector search and index

| Field | Default | Purpose |
| --- | --- | --- |
| `strategy` | `YDBSearchStrategy.COSINE_SIMILARITY` | Vector scoring function and index metric. Match the metric of an existing index. |
| `index_enabled` | `False` | Use a vector index for similarity search and rebuild it after writes. On a new table, the index is first created after adding documents. Hybrid search creates missing indexes when the store opens. |
| `index_name` | `"ydb_vector_index"` | Vector index name; set it to the existing name when reusing an index. |
| `index_config_levels` | `2` | K-means tree depth when a vector index is created or rebuilt. |
| `index_config_clusters` | `128` | Number of clusters when a vector index is created or rebuilt. |
| `index_tree_search_top_size` | `1` | `ydb.KMeansTreeSearchTopSize` for indexed vector queries, including the vector branch of hybrid search. Higher values search more tree candidates. |
| `vector_dimension` | `None` | Embedding dimension for index creation or rebuild. If omitted when needed, it is inferred from `embed_query("index")` or `aembed_query("index")`. |
| `vector_pass_as_bytes` | `True` | Pass document and query vectors as binary `String` values. With `False`, pass `List<Float>` and convert in YQL. |

Index creation settings affect a new or rebuilt vector index. Opening a table
with a ready index does not change that index. Adding documents rebuilds the
vector index when `index_enabled` or `hybrid_search_enabled` is set.

#### Hybrid index setup

| Field | Default | Purpose |
| --- | --- | --- |
| `hybrid_search_enabled` | `False` | Enable hybrid search and create any missing fulltext and vector indexes when opening a new or existing table. Implies indexed vector search even if `index_enabled=False`. |
| `fulltext_index_name` | `"ydb_fulltext_index"` | Fulltext relevance index name. Set it to an existing index name to reuse that index. |

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

## Hybrid search

On a YDB server that supports [HybridRank](https://ydb.tech/docs/en/dev/hybrid-search?version=main),
hybrid search combines fulltext relevance over the document column with vector
similarity over the embedding column. Both indexes belong to the same table.
Set `hybrid_search_enabled=True` to create missing indexes and wait until they
are ready. Existing documents are indexed in place without recomputing their
embeddings.

### New table

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

### Existing table

Open the same table with `hybrid_search_enabled=True` and leave
`drop_existing_table=False` (the default). Set `index_name` and
`fulltext_index_name` to the names of any indexes you want to reuse. Missing
indexes are built over the existing rows; ready indexes with those names are
reused.

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

The automatically created fulltext index is `GLOBAL USING fulltext_relevance`
on `column_map["document"]` with
`WITH (tokenizer=standard, use_filter_lowercase=true)`. These tokenizer and
normalization options are fixed in the integration. For different options,
create a `fulltext_relevance` index directly on the document column and pass
its name as `fulltext_index_name`. The store checks the indexed column and
readiness of an existing index, but does not check its type or tokenizer
settings. The fulltext index follows subsequent writes automatically; the
vector index is rebuilt after documents are added.

### Query options

`hybrid_search` and `ahybrid_search` use the same input text for fulltext
matching and query embedding. Their query-time options are:

| Argument | Default | Purpose |
| --- | --- | --- |
| `k` | `4` | Number of returned documents; must be a positive integer. |
| `mode` | `"rrf"` | Fusion by reciprocal rank (`"rrf"`) or normalized weighted scores (`"linear"`). |
| `weights` | `(1.0, 1.0)` | Non-negative weights in **(fulltext, vector)** order. |
| `candidate_limits` | `None` | Positive candidate counts in **(fulltext, vector)** order. By default, YDB uses `k * 10` candidates per branch. |

For example, this gives the vector branch twice the weight:

```python
store.hybrid_search(
    "database", k=5, weights=(1.0, 2.0), candidate_limits=(50, 100)
)
retriever = store.as_hybrid_retriever(k=5, weights=(1.0, 2.0))
```

For asynchronous I/O, use `await AsyncYDB.create(embeddings, config=settings)`,
`await store.ahybrid_search(...)`, and
`await store.as_hybrid_retriever(k=4).ainvoke(...)`. Close the store with
`await store.aclose()`.

The ordinary `similarity_search` and `as_retriever()` remain vector-only. Use
`as_hybrid_retriever()` for hybrid retrieval. Metadata `filter` is unsupported
by the native hybrid query and raises `ValueError`; the API returns documents
in fused order without a numeric fused score.

The [basic example notebook](examples/basic_example.ipynb) compares vector and
hybrid results on an existing table.
