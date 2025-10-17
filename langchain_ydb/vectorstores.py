from __future__ import annotations

import enum
import json
import logging
import struct
from dataclasses import dataclass, field
from hashlib import sha1
from typing import Any, Dict, Iterable, List, Optional

import ydb
import ydb_dbapi
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)


class YDBSearchStrategy(str, enum.Enum):
    """Enumerator of the search strategies."""

    INNER_PRODUCT_SIMILARITY = "InnerProductSimilarity"
    COSINE_SIMILARITY = "CosineSimilarity"
    COSINE_DISTANCE = "CosineDistance"
    MANHATTAN_DISTANCE = "ManhattanDistance"
    EUCLIDEAN_DISTANCE = "EuclideanDistance"

    def __str__(self) -> str:
        return self.value


DEFAULT_SEARCH_STRATEGY = YDBSearchStrategy.COSINE_SIMILARITY


def _get_default_column_map_dict() -> Dict[str, str]:
    return {
        "id": "id",
        "document": "document",
        "embedding": "embedding",
        "metadata": "metadata",
    }


@dataclass
class YDBSettings:
    """`YDB` client configuration.

    Attribute:
        host (str) : An URL to connect to YDB. Defaults to 'localhost'.
        port (int) : URL port to connect with GRPC. Defaults to 2136.
        username (str) : Username to login. Defaults to None.
        password (str) : Password to login. Defaults to None.
        secure (bool) : Connect to server over secure connection. Defaults to False.
        database (str) : Database name to find the table. Defaults to '/local'.
        table (str) : Table name to operate on. Defaults to 'ydb_langchain_store'.
        column_map (Dict) : Column type map to project column name onto langchain
                            semantics. Must have keys: `text`, `id`, `vector`,
                            must be same size to number of columns. For example:
                            .. code-block:: python

                                {
                                    'id': 'text_id',
                                    'embedding': 'text_embedding',
                                    'document': 'text_plain',
                                    'metadata': 'metadata_dictionary_in_json',
                                }

                            Defaults to identity map.
        strategy (str) : Strategy to perform search,
                         supported are ('InnerProductSimilarity', 'CosineSimilarity',
                         'CosineDistance', 'ManhattanDistance',
                         'EuclideanDistance'). Defaults to 'CosineSimilarity'.
                         Enum `YDBSearchStrategy` contains all of them.
        index_enabled (bool) : Enables usage of vector index. Default is False.
        index_name (str) : Name of vector index. Default is 'ydb_vector_index'.
        index_config_levels (int) : The number of levels in the tree, which determines
                                    the search depth (recommended 1–3). Default is 2.
        index_config_clusters (int) : The number of clusters in k-means, which defines
                                      the search breadth (recommended 64–512).
                                      Default is 128.
        index_tree_search_top_size (int) : Completeness of the indexed vector search.
                                           Default is 1.
        drop_existing_table (bool) : Flag to drop existing table while init.
                                     Defaults to False.
        vector_pass_as_bytes (bool) : Flag to pass vectors as bytes to YDB.
                                      Defaults to True.
    """

    host: str = "localhost"
    port: int = 2136

    username: Optional[str] = None
    password: Optional[str] = None

    secure: bool = False

    database: str = "/local"
    table: str = "ydb_langchain_store"

    column_map: Dict[str, str] = field(default_factory=_get_default_column_map_dict)

    strategy: str = DEFAULT_SEARCH_STRATEGY


    index_enabled: bool = False
    index_name: str = "ydb_vector_index"
    index_config_levels: int = 2
    index_config_clusters: int = 128
    index_tree_search_top_size: int = 1

    drop_existing_table: bool = False
    vector_pass_as_bytes: bool = True


class YDB(VectorStore):
    """`YDB` vector store.

    To use, you should have the ``ydb-dbapi`` python package installed.
    """

    def __init__(
        self,
        embedding: Embeddings,
        config: Optional[YDBSettings] = None,
        **kwargs: Any,
    ) -> None:
        """YDB Wrapper to LangChain

        Args:
            embedding (Embeddings): embedding function to use
            config (YDBSettings): Configuration to YDB DBAPI
            kwargs (any): Other keyword arguments will pass into ydb-dbapi
        """

        try:
            from tqdm import tqdm

            self.pgbar = tqdm
        except ImportError:
            # Just in case if tqdm is not installed
            self.pgbar = lambda x, **kwargs: x

        super().__init__()
        if config is not None:
            self.config = config
        else:
            self.config = YDBSettings()

        assert self.config
        assert self.config.host and self.config.port
        assert self.config.database and self.config.table
        assert self.config.column_map and self.config.strategy

        self.sort_order = (
            "DESC" if self.config.strategy.endswith("Similarity") else "ASC"
        )

        self.embedding_function = embedding

        # Create a connection to ydb
        self.connection = ydb_dbapi.connect(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            username=self.config.username,
            password=self.config.password,
            protocol="grpcs" if self.config.secure else "grpc",
            **kwargs,
        )

        if self.config.drop_existing_table:
            self.drop()

        self._execute_query(self._prepare_scheme_query(), ddl=True)

        self._insert_query = self._prepare_insert_query()

        self._add_index_query = self._prepare_add_index_query()

        self._batch_ydb_type = self._prepare_document_type()

    def _prepare_document_type(self) -> ydb.ListType:
        document_struct_type = ydb.StructType()
        document_struct_type.add_member('id', ydb.PrimitiveType.Utf8)
        document_struct_type.add_member('document', ydb.PrimitiveType.Utf8)
        document_struct_type.add_member('embedding', self._get_sdk_vector_type())
        document_struct_type.add_member('metadata', ydb.PrimitiveType.Json)
        return ydb.ListType(document_struct_type)

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """Access the query embedding object if available."""
        return self.embedding_function

    def _convert_vector_to_bytes_if_needed(
        self, vector: list[float]
    ) -> bytes | list[float]:
        if self.config.vector_pass_as_bytes:
            b = struct.pack("f" * len(vector), *vector)
            return b + b'\x01'
        return vector

    def _get_vector_type(self) -> str:
        if self.config.vector_pass_as_bytes:
            return "String"
        return "List<Float>"

    def _get_sdk_vector_type(self) -> ydb.PrimitiveType | ydb.ListType:
        if self.config.vector_pass_as_bytes:
            return ydb.PrimitiveType.String
        return ydb.ListType(ydb.PrimitiveType.Float)

    def _execute_query(
        self,
        query: str,
        params: Optional[dict] = None,
        ddl: bool = False,
    ) -> List:
        with self.connection.cursor() as cursor:
            if ddl:
                cursor.execute_scheme(query, params)
            else:
                cursor.execute(query, params)

            if cursor.description is None:
                return []

            columns = [col[0] for col in cursor.description]

            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def _prepare_scheme_query(self) -> str:
        """Create table schema
        :param dim: dimension of embeddings
        :param index_params: parameters used for index

        This function returns a `CREATE TABLE` statement based on the value of
        `self.config.index_type`.
        If an index type is specified that index will be created, otherwise
        no index will be created.
        In the case of there being no index, a linear scan will be performed
        when the embedding field is queried.
        """
        return f"""
        CREATE TABLE IF NOT EXISTS `{self.config.table}` (
            {self.config.column_map["id"]} Utf8,
            {self.config.column_map["document"]} Utf8,
            {self.config.column_map["embedding"]} String,
            {self.config.column_map["metadata"]} Json,
            PRIMARY KEY ({self.config.column_map["id"]})
        );"""

    def _escape_str(self, text: str) -> str:
        escape = "\\"
        chars_to_escape = ["\\", '"', "'"]
        return "".join(
            [ch if ch not in chars_to_escape else escape + ch for ch in text]
        )

    def _get_index_strategy(self) -> str:
        if self.config.strategy == YDBSearchStrategy.COSINE_SIMILARITY:
            return "similarity=cosine"
        if self.config.strategy == YDBSearchStrategy.INNER_PRODUCT_SIMILARITY:
            return "similarity=inner_product"
        if self.config.strategy == YDBSearchStrategy.COSINE_DISTANCE:
            return "distance=cosine"
        if self.config.strategy == YDBSearchStrategy.EUCLIDEAN_DISTANCE:
            return "distance=euclidean"
        if self.config.strategy == YDBSearchStrategy.MANHATTAN_DISTANCE:
            return "distance=manhattan"
        raise ValueError(f"Unsupported strategy: {self.config.strategy}")


    def _prepare_add_index_query(self) -> str:

        vector_dim = len(self.embedding_function.embed_query("index"))
        return f"""
        ALTER TABLE `{self.config.table}`
        ADD INDEX {self.config.index_name}__temp
        GLOBAL USING vector_kmeans_tree
        ON ({self.config.column_map["embedding"]})
        WITH (
            {self._get_index_strategy()},
            vector_type="Float",
            vector_dimension={vector_dim},
            levels={self.config.index_config_levels},
            clusters={self.config.index_config_clusters}
        );
        """

    def update_vector_index_if_needed(self) -> None:
        if not self.config.index_enabled:
            return

        logger.info("Updating vector index...")

        query = self._prepare_add_index_query()
        self._execute_query(query, ddl=True)
        self.connection._driver.table_client.alter_table(
            f"{self.config.database}/{self.config.table}",
            rename_indexes=[
                ydb.RenameIndexItem(
                    source_name=f"{self.config.index_name}__temp",
                    destination_name=f"{self.config.index_name}",
                    replace_destination=True,
                ),
            ],
        )

        logger.info("Vector index updated")


    def _prepare_insert_query(self) -> str:
        embedding_select = "embedding" if self.config.vector_pass_as_bytes \
             else "Untag(Knn::ToBinaryStringFloat(embedding), 'FloatVector')"

        return f"""
        DECLARE $documents AS List<Struct<
            id: Utf8,
            document: Utf8,
            embedding: {self._get_vector_type()},
            metadata: Json>>;

        UPSERT INTO `{self.config.table}`
        (
        {self.config.column_map["id"]},
        {self.config.column_map["document"]},
        {self.config.column_map["embedding"]},
        {self.config.column_map["metadata"]}
        )
        SELECT
            id,
            document,
            {embedding_select},
            metadata
        FROM AS_TABLE($documents);
        """

    def _prepare_search_query(
        self,
        k: int,
        filter: Optional[dict],
    ) -> str:
        where_statement = ""
        if filter:
            if self.config.index_enabled:
                raise ValueError("Unable to use filter with enabled vector index.")

            where_statement = "WHERE "
            metadata_col = self.config.column_map["metadata"]
            stmts = []
            for key, value in filter.items():
                stmts.append(f'JSON_VALUE({metadata_col}, "$.{key}") = "{value}"')

            where_statement = f"WHERE {' AND '.join(stmts)}"

        strategy = self.config.strategy
        embedding_col = self.config.column_map["embedding"]

        pragma_statement = ""
        if self.config.index_enabled:
            size = self.config.index_tree_search_top_size
            pragma_statement = f"""
            PRAGMA ydb.KMeansTreeSearchTopSize="{size}";
            """

        view_index = ""
        if self.config.index_enabled:
            view_index = f"VIEW {self.config.index_name}"

        if self.config.vector_pass_as_bytes:
            declare_embedding = """
            DECLARE $embedding as String;

            $TargetEmbedding = $embedding;
            """
        else:
            declare_embedding = """
            DECLARE $embedding as List<Float>;

            $TargetEmbedding = Knn::ToBinaryStringFloat($embedding);
            """

        return f"""
        {pragma_statement}

        {declare_embedding}

        SELECT
            {self.config.column_map["id"]} as id,
            {self.config.column_map["document"]} as document,
            {self.config.column_map["metadata"]} as metadata,
        Knn::{strategy}({embedding_col}, $TargetEmbedding) as score
        FROM {self.config.table} {view_index}
        {where_statement}
        ORDER BY score
        {self.sort_order}
        LIMIT {k};
        """

    def _prepare_delete_query(self, ids: Optional[list[str]]) -> str:
        query = f"DELETE FROM {self.config.table}"
        if ids:
            query += f" WHERE {self.config.column_map['id']} IN {str(ids)}"
        return query

    def add_embeddings(
        self,
        texts: Iterable[str],
        embeddings: list[list[float]],
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> list[str]:
        """Add prepared embeddings to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            embeddings: List of embedding vectors.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of IDs associated with the texts.
            batch_size: Number of texts to process in a single batch. Defaults to 32.
            **kwargs: vectorstore specific parameters.
                One of the kwargs should be `ids` which is a list of ids
                associated with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        texts_ = texts if isinstance(texts, (list, tuple)) else list(texts)

        if ids is None:
            ids = [sha1(t.encode("utf-8")).hexdigest() for t in texts_]

        if metadatas and len(metadatas) != len(texts_):
            msg = (
                "The number of metadatas must match the number of texts."
                f"Got {len(metadatas)} metadatas and {len(texts_)} texts."
            )
            raise ValueError(msg)

        metadatas = metadatas if metadatas else [{} for _ in range(len(texts_))]

        # Process in batches
        batch_ranges = range(0, len(texts_), batch_size)
        for i in self.pgbar(
            batch_ranges,
            desc="Processing batches...",
            total=len(batch_ranges)
        ):
            batch_texts = texts_[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]

            # Create a list of document structs
            documents = []
            for doc_id, doc_text, doc_embedding, doc_metadata in zip(
                batch_ids, batch_texts, batch_embeddings, batch_metadatas
            ):
                # Use dictionary format for struct values - YDB will convert them
                document = {
                    'id': doc_id,
                    'document': doc_text,
                    'embedding': self._convert_vector_to_bytes_if_needed(doc_embedding),
                    'metadata': json.dumps(doc_metadata)
                }
                documents.append(document)

            # Execute the batch insert
            self._execute_query(
                self._insert_query,
                {
                    "$documents": (documents, self._batch_ydb_type)
                },
            )

        self.update_vector_index_if_needed()

        return ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        batch_size: int = 32,
        batch_embeddings: bool = True,
        **kwargs: Any,
    ) -> list[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of IDs associated with the texts.
            batch_size: Number of texts to process in a single batch. Defaults to 32.
            batch_embeddings: If False, embeddings will be calculated
                              in a single batch. Defaults to True.
            **kwargs: vectorstore specific parameters.
                One of the kwargs should be `ids` which is a list of ids
                associated with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """

        texts_ = texts if isinstance(texts, (list, tuple)) else list(texts)

        if not batch_embeddings:
            embeddings = self.embedding_function.embed_documents(texts_) # type: ignore

            return self.add_embeddings(
                texts=texts_,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
                batch_size=batch_size,
                **kwargs,
            )

        if ids is None:
            ids = [sha1(t.encode("utf-8")).hexdigest() for t in texts_]

        if metadatas and len(metadatas) != len(texts_):
            msg = (
                "The number of metadatas must match the number of texts."
                f"Got {len(metadatas)} metadatas and {len(texts_)} texts."
            )
            raise ValueError(msg)

        metadatas = metadatas if metadatas else [{} for _ in range(len(texts_))]

        # Process in batches
        batch_ranges = range(0, len(texts_), batch_size)
        for i in self.pgbar(
            batch_ranges,
            desc="Processing batches...",
            total=len(batch_ranges)
        ):
            batch_texts = texts_[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            batch_embeddings_ = self.embedding_function.embed_documents(batch_texts) # type: ignore

            # Create a list of document structs
            documents = []
            for doc_id, doc_text, doc_embedding, doc_metadata in zip(
                batch_ids, batch_texts, batch_embeddings_, batch_metadatas
            ):
                # Use dictionary format for struct values - YDB will convert them
                document = {
                    'id': doc_id,
                    'document': doc_text,
                    'embedding': self._convert_vector_to_bytes_if_needed(doc_embedding),
                    'metadata': json.dumps(doc_metadata)
                }
                documents.append(document)

            # Execute the batch insert
            self._execute_query(
                self._insert_query,
                {
                    "$documents": (documents, self._batch_ydb_type)
                },
            )

        self.update_vector_index_if_needed()

        return ids

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
        *,
        config: Optional[YDBSettings] = None,
        ids: Optional[list[str]] = None,
        batch_size: int = 32,
        batch_embeddings: bool = True,
        **kwargs: Any,
    ) -> YDB:
        """Return YDB VectorStore initialized from texts and embeddings.

        Args:
            texts: Texts to add to the vectorstore.
            embedding: Embedding function to use.
            metadatas: Optional list of metadatas associated with the texts.
                Default is None.
            ids: Optional list of IDs associated with the texts.
            batch_size: Number of texts to process in a single batch. Defaults to 32.
            batch_embeddings: If False, embeddings will be calculated
                              in a single batch. Defaults to True.
            kwargs: Additional keyword arguments.

        Returns:
            VectorStore: VectorStore initialized from texts and embeddings.
        """
        vs = cls(embedding, config, **kwargs)
        vs.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            batch_size=batch_size,
            batch_embeddings=batch_embeddings,
        )
        return vs

    def delete(self, ids: Optional[list[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete. If None, delete all. Default is None.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        query = self._prepare_delete_query(ids)
        self._execute_query(query)
        return True

    def similarity_search(
        self, query: str, k: int = 4, filter: Optional[dict] = None, **kwargs: Any
    ) -> list[Document]:
        """Return docs most similar to query.

        Args:
            query: Input text.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents most similar to the query.
        """
        embedding = self.embedding_function.embed_query(query)
        return self.similarity_search_by_vector(embedding, k, filter=filter, **kwargs)

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents most similar to the query vector.
        """

        query = self._prepare_search_query(k, filter=filter)
        res = self._execute_query(
            query,
            params={
                "$embedding": (
                    self._convert_vector_to_bytes_if_needed(embedding),
                    self._get_sdk_vector_type()
                )
            },
        )
        return [
            Document(
                page_content=row["document"],
                metadata=json.loads(row["metadata"]),
            )
            for row in res
        ]

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: Optional[dict] = None, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Run similarity search with distance.

        Args:
            *args: Arguments to pass to the search method.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Tuples of (doc, similarity_score).
        """

        embedding = self.embedding_function.embed_query(query)
        query = self._prepare_search_query(k, filter=filter)
        res = self._execute_query(
            query,
            params={
                "$embedding": (
                    self._convert_vector_to_bytes_if_needed(embedding),
                    self._get_sdk_vector_type()
                )
            },
        )
        return [
            (
                Document(
                    page_content=row["document"],
                    metadata=json.loads(row["metadata"]),
                ),
                row["score"],
            )
            for row in res
        ]

    def similarity_search_by_vector_with_score(
        self,
        embedding: list[float],
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents most similar to the query vector.
        """

        query = self._prepare_search_query(k, filter=filter)
        res = self._execute_query(
            query,
            params={
                "$embedding": (
                    self._convert_vector_to_bytes_if_needed(embedding),
                    self._get_sdk_vector_type()
                )
            },
        )
        return [
            (
                Document(
                    page_content=row["document"],
                    metadata=json.loads(row["metadata"]),
                ),
                row["score"],
            )
            for row in res
        ]

    def drop(self) -> None:
        """
        Helper function: Drop data
        """
        self._execute_query(
            f"DROP TABLE IF EXISTS `{self.config.table}`",
            ddl=True,
        )
